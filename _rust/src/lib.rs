use ndarray::{indices, Array, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyModule};
use pyo3::{PyErr, exceptions::PyValueError};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;
use rayon::{ThreadPool, ThreadPoolBuilder};
use crate::util::{start_timing, print_timing, get_num_threads};

mod tokenize;   //n-gram extraction for char/char_wb
mod hashing;    //stable fast hashing to [0, n_features)
mod tfidf;      //DF counting, IDF vector, TF*IDF, per-row L2 norm
mod csr;
mod fd;         //Frequent Directions
mod util;

// Simple mapping from domain error to PyErr
fn to_pyerr(err: tfidf::Error) -> PyErr {
    use tfidf::Error::*;
    let msg = match err {
        InvalidAnalyzer => "Invalid analyzer".to_string(),
        InvalidNgramRange => "Invalid ngram_range".to_string(),
        Internal => "Internal error".to_string()
    };
    PyErr::new::<PyValueError, _>(msg)
}

#[pyfunction]
#[pyo3(signature = (data, indices, indptr, n_rows, n_cols, k, oversample=16, seed=None))]
fn fd_embed_from_csr(py: Python<'_>, data: Bound<PyArray1<f32>>, indices: Bound<PyArray1<i32>>,
    indptr: Bound<PyArray1<i64>>, n_rows: usize, n_cols: usize, k: usize,
    oversample: usize, seed: Option<u64>) -> PyResult<Py<PyArray2<f32>>>
{
    // Step 1: Zero-copy view of NumPy arrays
    let data = unsafe { data.as_slice()? };
    let indices = unsafe { indices.as_slice()? };
    let indptr = unsafe { indptr.as_slice()? };

    // Step 2: Gather the parameters
    let out_w = k + oversample; //k+p
    let s = seed.unwrap_or(0xC0FFEE); //I love coffee :)

    // Step 3: Build Ω (d x out_w), but don't store full Ω. Generate on the fly per-column.
    // We pre-allocate Ω^T as Vec<Vec<f32>>; width is small (<= 128).
    // TODO: Avoid materializing omega. Stream random f32 numbers in during building Y
    let mut rng = StdRng::seed_from_u64(s);
    let mut omega_t: Vec<Vec<f32>> = Vec::with_capacity(out_w);
    for _ in 0..out_w {
        let mut col: Vec<f32> = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            let r: f32 = if rng.random::<bool>() { 1.0 } else { -1.0 };
            col.push(r); //col is a vector of 1s and -1s
        }
        omega_t.push(col);
    }

    // Create rayon thread pool
    let num_threads = get_num_threads();
    let pool: Option<ThreadPool> = if num_threads > 0 {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads).build()
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to build rayon pool: {e}")))?;
        Some(thread_pool)
    }
    else { //num_threads = 0. Use global threadpool
        None
    };
    let pool_ref = pool.as_ref();


    // Step 4: Compute Y = X · Ω  (n x out_w) in a single pass over CSR rows
    // TODO: Move this to CSR utility module
    //let t0 = Instant::now();
    let t0 = start_timing();
    let mut y = Array2::<f32>::zeros((n_rows, out_w)); //dense y
    let mut build_y = || {
        y.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(row, mut yrow)| {
                let start = indptr[row] as usize;
                let end   = indptr[row + 1] as usize;
                for t in 0..out_w {
                    let mut acc = 0.0f32;
                    for p in start..end {
                        let j = indices[p] as usize;
                        let v = data[p];
                        acc += v * omega_t[t][j];
                    }
                    yrow[t] = acc;
                }
            });
    };
    match pool_ref {
        Some(pool) => pool.install(build_y), //use custom threadpool
        None => build_y() //use global threadpool
    }
    print_timing("build y", t0);

    // Step 5: Run FD on Y (n x out_w) -> Z (n x k)
    // FD operates on small width (out_w), making it cheap
    let t0 = start_timing();
    let z = fd::fd_reduce(y.view(), k, pool_ref)?;
    print_timing("fd_reduce", t0);

    // Step 6: Return NumPy (zero-copy)
    let py_z = z.into_pyarray(py).to_owned();
    Ok(Py::from(py_z))
}

#[pyfunction]
#[pyo3(signature = (seq, analyzer, ngram_min, ngram_max, n_features))]
fn hashing_tfidf_csr(
    py: Python<'_>,
    seq: Bound<PyAny>,    //iterable of strings (empty for nulls)
    analyzer: &str, //"char"/"char_wb"
    ngram_min: usize, ngram_max: usize, n_features: usize
) -> PyResult<(
    Py<PyArray1<f32>>,  //data
    Py<PyArray1<i32>>,  //indices
    Py<PyArray1<i64>>,  //indptr
    usize,              //n_rows
    usize,              //n_cols (n_features)
    Py<PyArray1<f32>>   //idf (length of n_features)
)> {
    // Collect input into a vector. TODO: zero-copy
    let mut docs: Vec<String> = Vec::new();
    let iter = PyIterator::from_object(&seq)?;
    for item in iter {
        let obj = item?;
        // Treat none as empty string. Python pre-fill should already do this.
        let s: String = if obj.is_none() {String::new()} else {obj.extract()?};
        docs.push(s);
    }
    let n_rows = docs.len();

    // Work buffers to be produced by tfidf::build_csr
    // Compute-intensive work without the GIL. TODO: multi-threading.
    let (data, indices, indptr, idf) = py.allow_threads(|| {
        let builder = tfidf::Builder::new(analyzer, ngram_min, ngram_max, n_features)?;
        let out = builder.build_csr(&docs); //(data, indices, indptr, idf)
        out
    }).map_err(to_pyerr)?;

    // Convert to NumPy without copying where possible. from_vec is zero-copy but from_array is not.
    let py_data = PyArray1::<f32>::from_vec(py, data).to_owned();
    let py_indices = PyArray1::<i32>::from_vec(py, indices).to_owned();
    let py_indptr = PyArray1::<i64>::from_vec(py, indptr).to_owned();
    let py_idf = idf.into_pyarray(py).to_owned();

    Ok((Py::from(py_data), Py::from(py_indices), Py::from(py_indptr), n_rows, n_features, Py::from(py_idf)))

}

#[pymodule]
fn _rust_backend_native(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hashing_tfidf_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fd_embed_from_csr, m)?)?;
    Ok(())
}