use ndarray::Axis;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::threads::get_thread_pool;
use crate::util::{start_timing, print_timing};

fn compute_standard_scale_fit(
    x: ndarray::ArrayView2<f32>,
    n_chunks: usize,
) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
    let (n_rows, n_cols) = x.dim();
    let pool = get_thread_pool();
    let chunk_size = (n_rows / n_chunks).max(1);

    // Phase 1: each row-block computes partial sum and sum_sq per column.
    // Row-major layout means iterating rows within a block is cache-friendly.
    let mut compute = || {
        let partials: Vec<(Vec<f64>, Vec<f64>)> = x
            .axis_chunks_iter(Axis(0), chunk_size)
            .into_par_iter()
            .map(|chunk| {
                let mut sum = vec![0.0f64; n_cols];
                let mut sum_sq = vec![0.0f64; n_cols];
                for row in chunk.rows() {
                    for j in 0..n_cols {
                        let v = row[j] as f64;
                        sum[j] += v;
                        sum_sq[j] += v * v;
                    }
                }
                (sum, sum_sq)
            })
            .collect();

        // Phase 2: reduce partial results (single-threaded, cheap — just n_chunks * n_cols adds)
        let mut total_sum = vec![0.0f64; n_cols];
        let mut total_sum_sq = vec![0.0f64; n_cols];
        for (s, sq) in &partials {
            for j in 0..n_cols {
                total_sum[j] += s[j];
                total_sum_sq[j] += sq[j];
            }
        }

        // Phase 3: derive mean and scale
        let n = n_rows as f64;
        let mut mean = ndarray::Array1::<f32>::zeros(n_cols);
        let mut scale = ndarray::Array1::<f32>::zeros(n_cols);
        for j in 0..n_cols {
            let m = if n > 0.0 { total_sum[j] / n } else { 0.0 };
            let var = if n > 0.0 {
                (total_sum_sq[j] / n) - (m * m)
            } else {
                0.0
            };
            mean[j] = m as f32;
            // Match sklearn: avoid division by zero by falling back to 1.0
            scale[j] = if var > 0.0 { var.sqrt() as f32 } else { 1.0 };
        }
        (mean, scale)
    };

    match pool {
        Some(p) => p.install(&mut compute),
        None => compute(),
    }
}

#[pyfunction]
#[pyo3(signature = (x, n_chunks))]
pub fn standard_scale_fit(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    n_chunks: usize,
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    if n_chunks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_chunks must be >= 1",
        ));
    }
    let x_view = x.as_array();
    let (mean, scale) = py.allow_threads(|| compute_standard_scale_fit(x_view, n_chunks));
    let py_mean = mean.into_pyarray(py).to_owned();
    let py_scale = scale.into_pyarray(py).to_owned();
    Ok((Py::from(py_mean), Py::from(py_scale)))
}

/// This is a minimal kernel to be called from Python.
/// It assumes:
/// - `X` is shape (n_samples, n_features)
/// - `mean` and `scale` are length-n_features vectors
#[pyfunction]
#[pyo3(signature = (x, mean, scale, n_chunks))]
pub fn standard_scale_transform(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    mean: PyReadonlyArray1<f32>,
    scale: PyReadonlyArray1<f32>,
    n_chunks: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let x = x.as_array();
    let mean = mean.as_slice()?;
    let scale = scale.as_slice()?;

    let (n_rows, n_cols) = x.dim();
    if mean.len() != n_cols || scale.len() != n_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mean/scale length must match number of columns in X",
        ));
    }
    if n_chunks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_chunks must be >= 1",
        ));
    }

    let out = py.allow_threads(|| {
        // Allocate output; row-major so rows are contiguous
        let mut out = ndarray::Array2::<f32>::zeros((n_rows, n_cols));
        let pool = get_thread_pool();
        let chunk_size = (n_rows / n_chunks).max(1);
        let t0 = start_timing();
        let mut do_scale = || {
            out.axis_chunks_iter_mut(Axis(0), chunk_size)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, mut out_chunk)| {
                    let start = chunk_idx * chunk_size;
                    let chunk_rows = out_chunk.nrows();
                    for i in 0..chunk_rows {
                        let x_row = x.row(start + i);
                        for j in 0..n_cols {
                            out_chunk[[i, j]] = (x_row[j] - mean[j]) / scale[j];
                        }
                    }
                });
        };
        match pool {
            Some(p) => p.install(do_scale),
            None => do_scale(),
        }
        print_timing("standard_scale_transform", t0);
        out
    });

    let py_out = out.into_pyarray(py).to_owned();
    Ok(Py::from(py_out))
}

