use ndarray::Axis;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::threads::get_thread_pool;

fn compute_standard_scale_fit(
    x: PyReadonlyArray2<f32>,
) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
    let x = x.as_array();
    let (n_rows, n_cols) = x.dim();
    let mut mean = ndarray::Array1::<f32>::zeros(n_cols);
    let mut scale = ndarray::Array1::<f32>::zeros(n_cols);

    // Compute per-column mean and standard deviation in parallel over columns.
    let pool = get_thread_pool();
    let mut compute = || {
        let stats: Vec<(f32, f32)> = (0..n_cols)
            .into_par_iter()
            .map(|j| {
                let mut sum = 0.0f64;
                let mut sumsq = 0.0f64;
                for i in 0..n_rows {
                    let v = x[(i, j)] as f64;
                    sum += v;
                    sumsq += v * v;
                }
                let n = n_rows as f64;
                let mean = if n > 0.0 { sum / n } else { 0.0 };
                let var = if n > 0.0 {
                    (sumsq / n) - (mean * mean)
                } else {
                    0.0
                };
                // Match sklearn-style behavior: avoid division by zero by falling back to 1.0
                let std = if var > 0.0 { var.sqrt() } else { 1.0 };
                (mean as f32, std as f32)
            })
            .collect();

        for j in 0..n_cols {
            mean[j] = stats[j].0;
            scale[j] = stats[j].1;
        }
    };

    match pool {
        Some(p) => p.install(compute),
        None => compute(),
    }

    (mean, scale)
}

#[pyfunction]
#[pyo3(signature = (x))]
pub fn standard_scale_fit(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    // We keep the GIL here while computing; the heavy work is parallelized internally
    // with Rayon over columns, so this remains multi-threaded without needing
    // `allow_threads`, and avoids sending non-Send PyO3 types across threads. --> caused compilation error with release
    let (mean, scale) = compute_standard_scale_fit(x);
    let py_mean = mean.into_pyarray(py).to_owned();
    let py_scale = scale.into_pyarray(py).to_owned();
    Ok((Py::from(py_mean), Py::from(py_scale)))
}

/// This is a minimal kernel to be called from Python.
/// It assumes:
/// - `X` is shape (n_samples, n_features)
/// - `mean` and `scale` are length-n_features vectors
#[pyfunction]
#[pyo3(signature = (x, mean, scale))]
pub fn standard_scale_transform(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    mean: PyReadonlyArray1<f32>,
    scale: PyReadonlyArray1<f32>,
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

    let out = py.allow_threads(|| {
        // Allocate output; row-major so rows are contiguous
        let mut out = ndarray::Array2::<f32>::zeros((n_rows, n_cols));
        let pool = get_thread_pool();
        let mut do_scale = || {
            out.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut out_row)| {
                    let x_row = x.row(i);
                    for j in 0..n_cols {
                        let m = mean[j];
                        let s = scale[j];
                        out_row[j] = (x_row[j] - m) / s;
                    }
                });
        };
        match pool {
            Some(p) => p.install(do_scale),
            None => do_scale(),
        }
        out
    });

    let py_out = out.into_pyarray(py).to_owned();
    Ok(Py::from(py_out))
}

