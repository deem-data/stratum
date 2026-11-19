//! Dense SimpleImputer kernels.
//!
//! Fit parallelizes independent columns; median/mode retain at most one column
//! per worker. Transform writes imputed values and indicators into one allocation,
//! without materializing a missing mask, repeated fill values, or a concatenation.
use ndarray::{Array2, ArrayView2, Axis, NdFloat};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::threads::get_bounded_thread_pool;

fn statistics<T: NdFloat>(
    x: ArrayView2<T>, strategy: &str, keep_empty: bool, fill: f64,
) -> (Vec<f64>, Vec<bool>) {
    let columns: Vec<(f64, bool)> = (0..x.ncols()).into_par_iter().map(|j| {
        let column = x.column(j);
        let mut count = 0usize;
        let mut sum = T::zero();
        let mut values = Vec::with_capacity(
            if matches!(strategy, "mean" | "constant") { 0 } else { x.nrows() });
        for &value in column {
            if !value.is_nan() {
                count += 1;
                if strategy == "mean" {
                    sum = sum + value;
                } else if strategy != "constant" {
                    values.push(value);
                }
            }
        }
        let has_missing = count < x.nrows();
        let value = if count == 0 {
            if !keep_empty { f64::NAN } else if strategy == "constant" { fill } else { 0.0 }
        } else {
            match strategy {
                "constant" => fill,
                "mean" => sum.to_f64().unwrap() / count as f64,
                "median" => {
                    let mid = values.len() / 2;
                    let even = values.len() % 2 == 0;
                    let (lower, upper, _) = values.select_nth_unstable_by(
                        mid, |a, b| a.partial_cmp(b).unwrap());
                    let median = if even {
                        let low = lower.iter().copied().reduce(|a, b| a.max(b)).unwrap();
                        (low + *upper) / T::from(2).unwrap()
                    } else { *upper };
                    median.to_f64().unwrap()
                }
                "most_frequent" => {
                    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    let mut best = values[0];
                    let mut best_count = 0;
                    let mut start = 0;
                    while start < values.len() {
                        let mut end = start + 1;
                        while end < values.len() && values[end] == values[start] { end += 1; }
                        // Strict > preserves sklearn's smallest-value tie break.
                        if end - start > best_count {
                            best = values[start];
                            best_count = end - start;
                        }
                        start = end;
                    }
                    best.to_f64().unwrap()
                }
                _ => unreachable!(),
            }
        };
        (value, has_missing)
    }).collect();
    columns.into_iter().unzip()
}

#[pyfunction]
pub fn simple_imputer_fit(
    py: Python<'_>, x: &Bound<'_, PyAny>, strategy: &str,
    keep_empty: bool, fill_value: f64, worker_budget: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<bool>>)> {
    if !matches!(strategy, "mean" | "median" | "most_frequent" | "constant") {
        return Err(PyValueError::new_err("unsupported imputation strategy"));
    }
    let pool = get_bounded_thread_pool(worker_budget).map_err(PyValueError::new_err)?;
    macro_rules! fit {
        ($input:expr) => {{
            let input = $input;
            let x = input.as_array();
            if x.nrows() == 0 || x.ncols() == 0 {
                return Err(PyValueError::new_err("nonempty X required"));
            }
            let (stats, missing) = py.detach(|| pool.install(
                || statistics(x, strategy, keep_empty, fill_value)));
            Ok((stats.into_pyarray(py).unbind(), missing.into_pyarray(py).unbind()))
        }};
    }
    if let Ok(x) = x.extract::<PyReadonlyArray2<f32>>() { fit!(x) }
    else { fit!(x.extract::<PyReadonlyArray2<f64>>()?) }
}

fn apply<T: NdFloat>(
    x: ArrayView2<T>, stats: &[f64], columns: &[usize], indicators: &[usize],
) -> Array2<T> {
    let mut value_dest = vec![None; x.ncols()];
    let mut indicator_dest = vec![None; x.ncols()];
    for (out, &input) in columns.iter().enumerate() { value_dest[input] = Some(out); }
    for (out, &input) in indicators.iter().enumerate() {
        indicator_dest[input] = Some(columns.len() + out);
    }
    let mut out = Array2::zeros((x.nrows(), columns.len() + indicators.len()));
    // Each worker owns disjoint output rows; dtype and feature order are retained.
    out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
        for j in 0..x.ncols() {
            if value_dest[j].is_none() && indicator_dest[j].is_none() { continue; }
            let value = x[[i, j]];
            let missing = value.is_nan();
            if let Some(dest) = value_dest[j] {
                row[dest] = if missing { T::from(stats[j]).unwrap() } else { value };
            }
            if let Some(dest) = indicator_dest[j] {
                row[dest] = if missing { T::one() } else { T::zero() };
            }
        }
    });
    out
}

#[pyfunction]
pub fn simple_imputer_transform(
    py: Python<'_>, x: &Bound<'_, PyAny>, statistics: PyReadonlyArray1<f64>,
    columns: Vec<usize>, indicators: Vec<usize>, worker_budget: usize,
) -> PyResult<Py<PyAny>> {
    let stats = statistics.as_slice()?;
    let pool = get_bounded_thread_pool(worker_budget).map_err(PyValueError::new_err)?;
    macro_rules! transform {
        ($input:expr) => {{
            let input = $input;
            let x = input.as_array();
            if stats.len() != x.ncols() || columns.iter().chain(&indicators).any(|&j| j >= x.ncols()) {
                return Err(PyValueError::new_err("imputer feature count or indices mismatch"));
            }
            let out = py.detach(|| pool.install(|| apply(x, stats, &columns, &indicators)));
            Ok(out.into_pyarray(py).into_any().unbind())
        }};
    }
    if let Ok(x) = x.extract::<PyReadonlyArray2<f32>>() { transform!(x) }
    else { transform!(x.extract::<PyReadonlyArray2<f64>>()?) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn statistics_ties_empty_columns_and_even_median() {
        let x = array![[4., f64::NAN], [2., f64::NAN], [4., f64::NAN], [2., f64::NAN]];
        let pool = get_bounded_thread_pool(2).unwrap();
        pool.install(|| {
            let (stats, missing) = statistics(x.view(), "median", false, 0.);
            assert_eq!(stats[0], 3.);
            assert!(stats[1].is_nan());
            assert_eq!(missing, vec![false, true]);
            assert_eq!(statistics(x.view(), "most_frequent", true, 0.).0, vec![2., 0.]);
            assert_eq!(statistics(x.view(), "constant", true, -999.).0, vec![-999., -999.]);
        });
    }

    #[test]
    fn fused_output_keeps_indicator_for_a_dropped_column() {
        let x = array![[1.0f32, f32::NAN, f32::NAN], [f32::NAN, f32::NAN, 6.]];
        let expected = array![[1.0f32, 4., 0., 1.], [2., 6., 1., 1.]];
        for workers in [1, 2, 1] {
            let pool = get_bounded_thread_pool(workers).unwrap();
            let result = pool.install(|| apply(x.view(), &[2., f64::NAN, 4.], &[0, 2], &[0, 1]));
            assert_eq!(result, expected);
        }
    }
}
