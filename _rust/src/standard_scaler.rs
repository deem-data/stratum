//! Parallel dense float64 StandardScaler kernels.
//!
//! Python validates estimator semantics before dispatch. Fit uses a corrected
//! two-pass variance rather than E[x²] - E[x]² to avoid cancellation for offsets.
use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::threads::get_bounded_thread_pool;

const MIN_BLOCK_LEN: usize = 10_000;

#[pyfunction]
pub fn standard_scale_fit(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    worker_budget: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let x = x.as_array();
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(PyValueError::new_err("nonempty X required"));
    }
    let pool = get_bounded_thread_pool(worker_budget).map_err(PyValueError::new_err)?;
    // Chunk sizing is internal; the bounded pool enforces the actual worker cap.
    let n_chunks = (x.nrows() / MIN_BLOCK_LEN).max(1).min(worker_budget);
    let (mean, var) = py.detach(|| pool.install(|| compute_statistics(x, n_chunks)));
    Ok((
        mean.into_pyarray(py).unbind(),
        var.into_pyarray(py).unbind(),
    ))
}

fn compute_statistics(x: ArrayView2<f64>, n_chunks: usize) -> (Array1<f64>, Array1<f64>) {
    let (n_rows, n_cols) = x.dim();
    let chunk_size = n_rows.div_ceil(n_chunks).max(1);

    // Step 1: Sum row blocks, retaining only O(chunks * features) statistics.
    let partials: Vec<Vec<f64>> = x
        .axis_chunks_iter(Axis(0), chunk_size)
        .into_par_iter()
        .map(|chunk| {
            let mut sum = vec![0.0; n_cols];
            for row in chunk.rows() {
                for j in 0..n_cols {
                    sum[j] += row[j];
                }
            }
            sum
        })
        .collect();
    let mut mean = Array1::<f64>::zeros(n_cols);
    for sum in partials {
        for j in 0..n_cols {
            mean[j] += sum[j];
        }
    }
    mean /= n_rows as f64;

    // Step 2: Correct the centered sum of squares for residual mean error.
    // This is safe for nearly constant columns with large offsets; Python uses
    // sklearn's constant-feature bound to choose the final scale.
    let partials: Vec<(Vec<f64>, Vec<f64>)> = x
        .axis_chunks_iter(Axis(0), chunk_size)
        .into_par_iter()
        .map(|chunk| {
            let mut residual = vec![0.0; n_cols];
            let mut squared = vec![0.0; n_cols];
            for row in chunk.rows() {
                for j in 0..n_cols {
                    let delta = row[j] - mean[j];
                    residual[j] += delta;
                    squared[j] += delta * delta;
                }
            }
            (residual, squared)
        })
        .collect();
    let mut residual = vec![0.0; n_cols];
    let mut var = Array1::<f64>::zeros(n_cols);
    for (sum, squared) in partials {
        for j in 0..n_cols {
            residual[j] += sum[j];
            var[j] += squared[j];
        }
    }
    for j in 0..n_cols {
        var[j] = ((var[j] - residual[j] * residual[j] / n_rows as f64) / n_rows as f64).max(0.0);
    }
    (mean, var)
}

#[pyfunction]
pub fn standard_scale_transform(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    mean: PyReadonlyArray1<f64>,
    scale: PyReadonlyArray1<f64>,
    worker_budget: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let x = x.as_array();
    let mean = mean.as_slice()?;
    let scale = scale.as_slice()?;
    let (n_rows, n_cols) = x.dim();
    if mean.len() != n_cols || scale.len() != n_cols {
        return Err(PyValueError::new_err(
            "mean/scale length must match X columns",
        ));
    }
    let pool = get_bounded_thread_pool(worker_budget).map_err(PyValueError::new_err)?;
    let n_chunks = (n_rows / MIN_BLOCK_LEN).max(1).min(worker_budget);
    let out = py.detach(|| {
        let mut out = Array2::<f64>::zeros((n_rows, n_cols));
        let chunk_size = n_rows.div_ceil(n_chunks).max(1);
        let compute = || {
            out.axis_chunks_iter_mut(Axis(0), chunk_size)
                .into_par_iter()
                .enumerate()
                .for_each(|(block, mut chunk)| {
                    for i in 0..chunk.nrows() {
                        let row = x.row(block * chunk_size + i);
                        for j in 0..n_cols {
                            chunk[[i, j]] = (row[j] - mean[j]) / scale[j];
                        }
                    }
                });
        };
        pool.install(compute);
        out
    });
    Ok(out.into_pyarray(py).unbind())
}
