use crate::threads::get_thread_pool;
use crate::util::{print_timing, start_timing};
use ndarray::Axis;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
pub fn compute_minmax_scale_transform(
    x: ndarray::ArrayView2<f32>,
    min: ndarray::ArrayView1<f32>,
    max: ndarray::ArrayView1<f32>,
    n_chunks: usize,
    clip: bool,
) -> ndarray::Array2<f32> {
    let (n_rows, n_cols) = x.dim();
    let chunk_size = (n_rows / n_chunks).max(1);
    let pool = get_thread_pool();
    let mut out = ndarray::Array2::uninit((n_rows, n_cols));

    let min_vec: Vec<f32> = min.iter().copied().collect();

    let inv_range: Vec<f32> = min_vec
        .iter()
        .zip(max.iter())
        .map(|(&lo, &hi)| {
            let range = hi - lo;
            if range == 0.0 {
                0.0
            } else {
                1.0 / range
            }
        })
        .collect();
    // resolve clipping once and unconditionally apply in hotloop
    let (lower, upper) = if clip {
        (0.0f32, 1.0f32)
    } else {
        (f32::NEG_INFINITY, f32::INFINITY)
    };
    let t0 = start_timing();
    let mut compute = || {
        out.axis_chunks_iter_mut(Axis(0), chunk_size)
            .into_par_iter()
            .zip(x.axis_chunks_iter(Axis(0), chunk_size))
            .for_each(|(mut out_chunk, in_chunk)| {
                for (mut out_row, in_row) in out_chunk.rows_mut().into_iter().zip(in_chunk.rows()) {
                    match (out_row.as_slice_mut(), in_row.as_slice()) {
                        (Some(out_row_value), Some(in_row_value)) => {
                            for (((o, &v), &m), &inv) in out_row_value
                                .iter_mut()
                                .zip(in_row_value.iter())
                                .zip(min_vec.iter())
                                .zip(inv_range.iter())
                            {
                                o.write(((v - m) * inv).clamp(lower, upper));
                            }
                        }
                        _ => {
                            for c in 0..n_cols {
                                out_row[c].write(
                                    ((in_row[c] - min_vec[c]) * inv_range[c]).clamp(lower, upper),
                                );
                            }
                        }
                    }
                }
            })
    };
    match pool {
        Some(p) => p.install(&mut compute),
        None => compute(),
    }
    print_timing("minmax scale transform", t0);

    unsafe { out.assume_init() }
}

pub fn compute_minmax_scale_fit(
    x: ndarray::ArrayView2<f32>,
    n_chunks: usize,
) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
    let (n_rows, n_cols) = x.dim();
    let chunk_size = (n_rows / n_chunks).max(1);
    let pool = get_thread_pool();

    if n_rows == 0 {
        return (
            ndarray::Array1::from(vec![f32::NAN; n_cols]),
            ndarray::Array1::from(vec![f32::NAN; n_cols]),
        );
    }

    let t0 = start_timing();
    let mut compute = || {
        x.axis_chunks_iter(Axis(0), chunk_size)
            .into_par_iter()
            .map(|chunk| {
                let mut min = vec![f32::INFINITY; n_cols];
                let mut max = vec![f32::NEG_INFINITY; n_cols];
                for row in chunk.rows() {
                    for col_idx in 0..n_cols {
                        let value = row[col_idx];
                        if min[col_idx] > value {
                            min[col_idx] = value
                        }
                        if max[col_idx] < value {
                            max[col_idx] = value
                        }
                    }
                }
                (min, max)
            })
            .reduce(
                || (vec![f32::INFINITY; n_cols], vec![f32::NEG_INFINITY; n_cols]),
                |(mut amin, mut amax), (bmin, bmax)| {
                    for col in 0..n_cols {
                        amin[col] = amin[col].min(bmin[col]);
                        amax[col] = amax[col].max(bmax[col]);
                    }

                    (amin, amax)
                },
            )
    };

    let (min, max) = match pool {
        Some(p) => p.install(&mut compute),
        None => compute(),
    };
    print_timing("minmax scale fit", t0);

    (ndarray::Array1::from(min), ndarray::Array1::from(max))
}

#[pyfunction]
#[pyo3(signature = (x, n_chunks))]
pub fn minmax_scale_fit(
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
    let (min, max) = py.allow_threads(|| compute_minmax_scale_fit(x_view, n_chunks));
    let py_min = min.into_pyarray(py).to_owned();
    let py_max = max.into_pyarray(py).to_owned();
    Ok((Py::from(py_min), Py::from(py_max)))
}

#[pyfunction]
#[pyo3(signature = (x, min, max, n_chunks, clip))]
pub fn minmax_scale_transform(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    min: PyReadonlyArray1<f32>,
    max: PyReadonlyArray1<f32>,
    n_chunks: usize,
    clip: bool,
) -> PyResult<Py<PyArray2<f32>>> {
    if n_chunks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_chunks must be >= 1",
        ));
    }

    let x_view = x.as_array();
    let min_view = min.as_array();
    let max_view = max.as_array();

    let result = py.allow_threads(|| {
        compute_minmax_scale_transform(x_view, min_view, max_view, n_chunks, clip)
    });
    let py_result = result.into_pyarray(py).to_owned();

    Ok(Py::from(py_result))
}
