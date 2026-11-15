use ndarray::{Array2, Axis, NdFloat};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::threads::get_thread_pool;
use crate::timing::{print_timing, start_timing};

// Pure Rust compute to run without GIL
fn compute_ohe_transform_csr<T: NdFloat>(
    codes_slices: &[&[i32]], 
    n_cats: &[usize], 
    drop_idx: &[Option<usize>]
) -> (Vec<T>, Vec<i32>, Vec<i64>, i64, i64) {
    let n_features = codes_slices.len();
    let n_rows = codes_slices[0].len();

    // Derive per-feature width after drop and global offsets
    let mut widths: Vec<usize> = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let n_cat = n_cats[j]; //#distinct items in feature j
        match drop_idx[j] {
            Some(_) => widths.push(n_cat.saturating_sub(1)),
            None => widths.push(n_cat)
        }
    }
    let mut offsets: Vec<usize> = vec![0; n_features];
    let mut n_cols:usize = 0;
    for j in 0..n_features {
        offsets[j] = n_cols;
        n_cols += widths[j];
    }

    // Pass 1: per-row nnz counts (row-parallel)
    let pool = get_thread_pool();
    let mut row_counts: Vec<usize> = vec![0; n_rows];
    let mut count_nnz = || {
        row_counts
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, rc)| {
                let mut c = 0usize; //per-row #nnz
                for j in 0..n_features {
                    let code = codes_slices[j][i];
                    if code >= 0 {
                        if let Some(d) = drop_idx[j] {
                            if (code as usize) == d {
                                continue; //dropped
                            }
                        }
                        c += 1;
                    }
                }
                *rc = c;
            });
    };
    match pool {
        Some(p) => p.install(count_nnz), //use custom threadpool
        None => count_nnz() //use global threadpool
    }

    // Build indptr by prefix sum, and capture total nnz
    let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
    indptr.push(0);
    for i in 0..n_rows {
        let next = indptr[i] + (row_counts[i] as i64);
        indptr.push(next);
    }
    let nnz = *indptr.last().unwrap() as usize;

    // Allocate data
    let mut indices: Vec<i32> = vec![0; nnz];
    let mut data: Vec<T> = vec![T::zero(); nnz];

    // Pass 2: Fill indices/data. TODO: Parallelize
    let mut cursor = 0usize;
    for i in 0..n_rows {
        let row_nnz = row_counts[i];
        let row_start = cursor;
        let row_end = row_start + row_nnz;
        let row_slice_idx = &mut indices[row_start..row_end];
        let row_slice_dat = &mut data[row_start..row_end];

        let mut w = 0usize;
        for j in 0..n_features {
            let k = codes_slices[j][i];
            if k < 0 { continue; } // unknown/NaN → zeros
            if let Some(d) = drop_idx[j] {
                if (k as usize) == d {
                    continue; // dropped category
                }
            }
            // Effective local column within feature block, accounting for drop
            let eff_local = if let Some(d) = drop_idx[j] {
                let ku = k as usize;
                if ku < d { ku } else { ku - 1 }
            } else {
                k as usize
            };
            let col = offsets[j] + eff_local;

            row_slice_idx[w] = col as i32;
            row_slice_dat[w] = T::one();
            w += 1;
        }
        debug_assert_eq!(w, row_nnz);
        cursor = row_end;
    }

    (data, indices, indptr, n_rows as i64, n_cols as i64)
}

// Build a CSR one-hot matrix from per-feature integer codes
#[pyfunction]
#[pyo3(signature = (codes, n_cats, drop_idx, dtype="float32"))]
pub fn ohe_transform_csr(
    py: Python<'_>, codes: Vec<PyReadonlyArray1<i32>>, n_cats: Vec<usize>, drop_idx: Vec<Option<usize>>,
    dtype: &str,
) -> PyResult<(
    Py<PyAny>,          //data, in the requested output dtype
    Py<PyArray1<i32>>,  //indices
    Py<PyArray1<i64>>,  //indptr
    i64,                //n_rows
    i64,                //n_cols (n_features)
)> {
    // Basic assertions
    // Row i of codes contains the recoded feature i
    let n_features = codes.len();
    // assert n_features > 0
    if n_features == 0 {return Err(pyo3::exceptions::PyValueError::new_err("code is empty"));}
    // Borrow as Rust slices for fast access.
    let code_slices: Vec<&[i32]> = codes.iter().map(|a| a.as_slice().unwrap()).collect();

    // Allocate directly in the requested dtype; OHE values are exactly zero/one
    // in both precisions, and no temporary float32 output is needed for float64.
    macro_rules! encode {
        ($ty:ty) => {{
            let (data, indices, indptr, n_rows, n_cols) = py.detach(|| {
                compute_ohe_transform_csr::<$ty>(&code_slices, &n_cats, &drop_idx)
            });
            Ok((
                data.into_pyarray(py).into_any().unbind(),
                indices.into_pyarray(py).unbind(),
                indptr.into_pyarray(py).unbind(),
                n_rows, n_cols,
            ))
        }};
    }
    match dtype {
        "float32" => encode!(f32),
        "float64" => encode!(f64),
        _ => Err(pyo3::exceptions::PyValueError::new_err("dtype must be float32 or float64")),
    }
}

// Pure Rust CSR to dense conversion without GIL
fn compute_csr_to_dense<T: NdFloat>(
    data: &[T],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize
) -> ndarray::Array2<T> {
    let pool = get_thread_pool();

    // Create dense output and fill rows in parallel
    let t0 = start_timing();
    let mut out = Array2::<T>::zeros((n_rows, n_cols));
    let mut densify = || {
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let start = indptr[i] as usize;
                let end = indptr[i + 1] as usize;
                for p in start..end {
                    let j = indices[p] as usize;
                    // All ones for OHE
                    row[j] = data[p];
                }
            });
    };
    match pool {
        Some(p) => p.install(densify), //use custom threadpool
        None => densify() //use global threadpool
    };
    print_timing("CSR to dense copy", t0);
    out
}

#[pyfunction]
#[pyo3(signature = (data, indices, indptr, n_rows, n_cols))]
pub fn csr_to_dense(py: Python<'_>, data: &Bound<'_, PyAny>, indices: PyReadonlyArray1<i32>,
    indptr: PyReadonlyArray1<i64>, n_rows: usize, n_cols: usize) -> PyResult<Py<PyAny>>
{
    let indices = indices.as_slice().unwrap();
    let indptr = indptr.as_slice().unwrap();
    if indptr.len() != n_rows + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err("indptr length mismatch"));
    }

    // Dispatch once by buffer dtype, then keep the dense allocation in that dtype.
    if let Ok(data) = data.extract::<PyReadonlyArray1<f32>>() {
        let data = data.as_slice()?;
        let out = py.detach(|| compute_csr_to_dense(data, indices, indptr, n_rows, n_cols));
        Ok(out.into_pyarray(py).into_any().unbind())
    } else {
        let data = data.extract::<PyReadonlyArray1<f64>>()?;
        let data = data.as_slice()?;
        let out = py.detach(|| compute_csr_to_dense(data, indices, indptr, n_rows, n_cols));
        Ok(out.into_pyarray(py).into_any().unbind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_drop_preserves_every_category_in_both_precisions() {
        let codes: &[&[i32]] = &[&[0, 1, -1], &[1, 0, -1]];
        let (data32, indices, indptr, rows, cols) =
            compute_ohe_transform_csr::<f32>(codes, &[2, 2], &[None, None]);
        let (data64, indices64, indptr64, _, _) =
            compute_ohe_transform_csr::<f64>(codes, &[2, 2], &[None, None]);
        assert_eq!((rows, cols), (3, 4));
        assert_eq!(indices, vec![0, 3, 1, 2]);
        assert_eq!(indices64, indices);
        assert_eq!(indptr, vec![0, 2, 4, 4]);
        assert_eq!(indptr64, indptr);
        assert_eq!(data32, vec![1.0f32; 4]);
        assert_eq!(data64, vec![1.0f64; 4]);
        let dense = compute_csr_to_dense(&data64, &indices, &indptr, 3, 4);
        assert_eq!(dense.row(0).to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(dense.row(2).to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn dropped_and_unknown_categories_can_produce_an_empty_csr() {
        let (data, indices, indptr, rows, cols) =
            compute_ohe_transform_csr::<f64>(&[&[0, -1]], &[2], &[Some(0)]);
        assert_eq!((rows, cols), (2, 1));
        assert!(data.is_empty());
        assert!(indices.is_empty());
        assert_eq!(indptr, vec![0, 0, 0]);
        assert_eq!(compute_csr_to_dense(&data, &indices, &indptr, 2, 1),
                   Array2::<f64>::zeros((2, 1)));
    }
}
