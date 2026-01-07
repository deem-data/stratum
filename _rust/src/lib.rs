use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyList, PyModule};
use pyo3::{PyErr, exceptions::PyValueError};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use crate::threads::{get_thread_pool};
use crate::util::{start_timing, print_timing};

mod tokenize;   //n-gram extraction for char/char_wb
mod hashing;    //stable fast hashing to [0, n_features)
mod tfidf;      //DF counting, IDF vector, TF*IDF, per-row L2 norm
mod csr;
mod fd;         //Frequent Directions
mod truncated_svd;  //TruncatedSVD using randomized SVD
mod util;
mod threads;
mod one_hot_encoder;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};

// TODO (refactor): Move functions to corresponding modules
// TODO (perf): Test with blas/mkl. Accordingly move from faer

// ---- Global registry for models (TODO: Return pointer to Python) ----
static TFIDF_MODELS: Lazy<Mutex<Vec<tfidf::TfidfModel>>> = Lazy::new(|| Mutex::new(Vec::new()));

struct TruncatedSvdModel {
    n_cols: usize,
    k: usize,
    // components_t is (n_cols × k)
    components_t: Arc<ndarray::Array2<f32>>,
    singular_values: Vec<f32>,
}
static TSVD_NEXT_ID: AtomicU64 = AtomicU64::new(1);
static TSVD_MODELS: Lazy<Mutex<HashMap<u64, TruncatedSvdModel>>> = Lazy::new(|| Mutex::new(HashMap::new()));

struct FdEmbedModel {
    n_cols: usize,
    k: usize,
    oversample: usize,
    // Projection matrix P: (m × k) where m = k + oversample
    projection: Arc<ndarray::Array2<f32>>,
    // Random matrix Ω: (n_cols × m) stored column-major for efficient CSR matmul
    // omega[j * m + t] = Ω[j, t]
    omega: Arc<Vec<f32>>,
    m: usize,  // m = k + oversample, width of reduced space
}
static FD_EMBED_NEXT_ID: AtomicU64 = AtomicU64::new(1);
static FD_EMBED_MODELS: Lazy<Mutex<HashMap<u64, FdEmbedModel>>> = Lazy::new(|| Mutex::new(HashMap::new()));

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

// Helper: CSR × Omega matmul: X @ Ω -> Y
// Omega is stored column-major: omega[j * m + t] = Ω[j, t]
// Result Y is (n_rows × m)
fn csr_matmul_omega(
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
    omega: &[f32],
    m: usize,
    pool_ref: Option<&rayon::ThreadPool>,
) -> Array2<f32> {
    let mut y = Array2::<f32>::zeros((n_rows, m));
    let mut build_y = || {
        y.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(row, mut yrow)| {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                for t in 0..m {
                    let mut acc = 0.0f32;
                    for p in start..end {
                        let j = indices[p] as usize;
                        let v = data[p];
                        acc += v * omega[j * m + t];
                    }
                    yrow[t] = acc;
                }
            });
    };
    match pool_ref {
        Some(p) => p.install(build_y),
        None => build_y(),
    }
    y
}

fn compute_fd_embed(data: &[f32], indices: &[i32], indptr: &[i64],
    n_rows: usize, n_cols: usize, k: usize, oversample: usize, seed: Option<u64>) -> Result<Array2<f32>, PyErr>
{
    // Step 2: Gather the parameters
    let out_w = k + oversample; //k+p
    let s = seed.unwrap_or(0xC0FFEE); //I love coffee :)

    // Step 3: Build Ω (d x out_w), but don't store full Ω. Generate on the fly per-column.
    // We pre-allocate Ω^T as Vec<Vec<f32>>; width is small (<= 128).
    // Do all heavy work without the GIL (allow_threads closure)
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

    // Get rayon thread pool
    let pool = get_thread_pool();

    // Step 4: Compute Y = X · Ω  (n x out_w) in a single pass over CSR rows
    // TODO: Move this to CSR utility module
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
    match pool {
        Some(p) => p.install(build_y), //use custom threadpool
        None => build_y() //use global threadpool
    }
    print_timing("build y", t0);

    // Step 5: Run FD on Y (n x out_w) -> Z (n x k)
    // FD operates on small width (out_w), making it cheap
    let t0 = start_timing();
    let z = fd::fd_reduce(y.view(), k, pool)?;
    print_timing("fd_reduce", t0);
    Ok(z)
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

    let z = py.allow_threads(||
        compute_fd_embed(data, indices, indptr, n_rows, n_cols, k, oversample, seed))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("fd_embed failed: {e}")))?;

    // Step 6: Return NumPy (zero-copy)
    let py_z = z.into_pyarray(py).to_owned();
    Ok(Py::from(py_z))
}

fn compute_fd_fit(
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
    k: usize,
    oversample: usize,
    seed: Option<u64>,
) -> Result<(u64, Array2<f32>), PyErr> {
    let m = k + oversample;
    let s = seed.unwrap_or(0xC0FFEE);

    // Generate Ω matrix (n_cols × m) in column-major format
    let mut rng = StdRng::seed_from_u64(s);
    let mut omega = Vec::<f32>::with_capacity(n_cols * m);
    for _ in 0..(n_cols * m) {
        let r: f32 = if rng.random::<bool>() { 1.0 } else { -1.0 };
        omega.push(r);
    }

    // Get rayon thread pool
    let pool = get_thread_pool();

    // Compute Y = X @ Ω (n_rows × m)
    let t0 = start_timing();
    let y = csr_matmul_omega(data, indices, indptr, n_rows, n_cols, &omega, m, pool);
    print_timing("build y (fd_fit)", t0);

    // Run FD to get projection matrix P and reduced embeddings Z
    let t0 = start_timing();
    let (z, projection) = fd::fd_fit(y.view(), k, pool)?;
    print_timing("fd_fit", t0);

    // Store model
    let model_id = FD_EMBED_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let model = FdEmbedModel {
        n_cols,
        k,
        oversample,
        projection: Arc::new(projection),
        omega: Arc::new(omega),
        m,
    };

    FD_EMBED_MODELS
        .lock()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("FD_EMBED_MODELS mutex poisoned"))?
        .insert(model_id, model);

    Ok((model_id, z))
}

#[pyfunction]
#[pyo3(signature = (data, indices, indptr, n_rows, n_cols, k, oversample=16, seed=None))]
fn fd_fit_from_csr(
    py: Python<'_>,
    data: Bound<PyArray1<f32>>,
    indices: Bound<PyArray1<i32>>,
    indptr: Bound<PyArray1<i64>>,
    n_rows: usize,
    n_cols: usize,
    k: usize,
    oversample: usize,
    seed: Option<u64>,
) -> PyResult<(u64, Py<PyArray2<f32>>)> {
    // Zero-copy view of NumPy arrays
    let data = unsafe { data.as_slice()? };
    let indices = unsafe { indices.as_slice()? };
    let indptr = unsafe { indptr.as_slice()? };

    let (model_id, z) = py
        .allow_threads(|| compute_fd_fit(data, indices, indptr, n_rows, n_cols, k, oversample, seed))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("fd_fit failed: {e}")))?;

    // Return NumPy (zero-copy)
    let py_z = z.into_pyarray(py).to_owned();
    Ok((model_id, Py::from(py_z)))
}

fn compute_fd_transform(
    model_id: u64,
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Array2<f32>, PyErr> {
    // Fetch model
    let (projection, omega, model_n_cols, m) = {
        let guard = FD_EMBED_MODELS
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("FD_EMBED_MODELS mutex poisoned"))?;

        let model = guard
            .get(&model_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("Unknown model_id {model_id}")))?;

        (
            Arc::clone(&model.projection),
            Arc::clone(&model.omega),
            model.n_cols,
            model.m,
        )
    };

    // Validate cols match
    if n_cols != model_n_cols {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "n_cols mismatch: input n_cols={} but model expects {}",
            n_cols, model_n_cols
        )));
    }

    let pool = get_thread_pool();

    // Compute Y_new = X_new @ Ω (n_rows × m)
    let t0 = start_timing();
    let y_new = csr_matmul_omega(data, indices, indptr, n_rows, n_cols, &omega, m, pool);
    print_timing("build y_new (fd_transform)", t0);

    // Apply projection Z_new = Y_new @ P (n_rows × k)
    let t0 = start_timing();
    let k = projection.ncols();
    let mut z_new = Array2::<f32>::zeros((n_rows, k));
    let mut apply_projection = || {
        z_new
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(y_new.axis_iter(Axis(0)))
            .for_each(|(mut zrow, yrow)| {
                for r in 0..k {
                    let mut sum = 0.0f32;
                    for c in 0..m {
                        sum += yrow[c] * projection[(c, r)];
                    }
                    zrow[r] = sum;
                }
            });
    };
    match pool {
        Some(p) => p.install(apply_projection),
        None => apply_projection(),
    }
    print_timing("apply projection (fd_transform)", t0);

    Ok(z_new)
}

#[pyfunction]
#[pyo3(signature = (model_id, data, indices, indptr, n_rows, n_cols))]
fn fd_transform_from_csr(
    py: Python<'_>,
    model_id: u64,
    data: Bound<PyArray1<f32>>,
    indices: Bound<PyArray1<i32>>,
    indptr: Bound<PyArray1<i64>>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let data = unsafe { data.as_slice()? };
    let indices = unsafe { indices.as_slice()? };
    let indptr = unsafe { indptr.as_slice()? };

    let z = py
        .allow_threads(|| compute_fd_transform(model_id, data, indices, indptr, n_rows, n_cols))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("fd_transform failed: {e}")))?;

    let py_z = z.into_pyarray(py).to_owned();
    Ok(Py::from(py_z))
}

fn compute_truncated_svd_fit(data: &[f32], indices: &[i32], indptr: &[i64],
    n_rows: usize, n_cols: usize, k: usize, seed: Option<u64>) -> Result<(u64, Array2<f32>), PyErr>
{
    // Hardcoded sklearn defaults: n_iter = 5 or 7, oversample=10
    const N_ITER: usize = 5;
    const OVERSAMPLE: usize = 10;
    let pool = get_thread_pool();

    let (z, components_t, s) = truncated_svd::truncated_svd_csr(
        data, indices, indptr,
        n_rows, n_cols,
        k, N_ITER, OVERSAMPLE,
        seed,
        pool,
    )?;

    let model_id = TSVD_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let model = TruncatedSvdModel {
        n_cols,
        k: components_t.ncols(),
        components_t: Arc::new(components_t),
        singular_values: s,
    };

    TSVD_MODELS
        .lock()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("TSVD model cache mutex poisoned"))?
        .insert(model_id, model);

    Ok((model_id, z))
}

#[pyfunction]
#[pyo3(signature = (data, indices, indptr, n_rows, n_cols, k, seed=None))]
fn truncated_svd_fit_from_csr(py: Python<'_>, data: Bound<PyArray1<f32>>, indices: Bound<PyArray1<i32>>,
    indptr: Bound<PyArray1<i64>>, n_rows: usize, n_cols: usize, k: usize,
    seed: Option<u64>) -> PyResult<(u64, Py<PyArray2<f32>>)>
{
    // Step 1: Zero-copy view of NumPy arrays
    let data = unsafe { data.as_slice()? };
    let indices = unsafe { indices.as_slice()? };
    let indptr = unsafe { indptr.as_slice()? };

    let (model_id, z) = py.allow_threads(||
        compute_truncated_svd_fit(data, indices, indptr, n_rows, n_cols, k, seed))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("truncated_svd_fit failed: {e}")))?;

    // Step 2: Return NumPy (zero-copy)
    let py_z = z.into_pyarray(py).to_owned();
    Ok((model_id, Py::from(py_z)))
}

fn compute_truncated_svd_transform(
    model_id: u64,
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Array2<f32>, PyErr> {
    // Fetch model
    let (components_t, model_n_cols) = {
        let guard = TSVD_MODELS
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("TSVD model cache mutex poisoned"))?;

        let m = guard
            .get(&model_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("Unknown model_id {model_id}")))?;

        (Arc::clone(&m.components_t), m.n_cols)
    };

    // Validate cols match
    if n_cols != model_n_cols {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "n_cols mismatch: input n_cols={} but model expects {}",
            n_cols, model_n_cols
        )));
    }

    let pool = get_thread_pool();
    truncated_svd::truncated_svd_transform_csr(data, indices, indptr, n_rows, n_cols, &components_t, pool)
}

#[pyfunction]
#[pyo3(signature = (model_id, data, indices, indptr, n_rows, n_cols))]
fn truncated_svd_transform_from_csr(
    py: Python<'_>,
    model_id: u64,
    data: Bound<PyArray1<f32>>,
    indices: Bound<PyArray1<i32>>,
    indptr: Bound<PyArray1<i64>>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let data = unsafe { data.as_slice()? };
    let indices = unsafe { indices.as_slice()? };
    let indptr = unsafe { indptr.as_slice()? };

    let z = py
        .allow_threads(|| compute_truncated_svd_transform(model_id, data, indices, indptr, n_rows, n_cols))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("truncated_svd_transform failed: {e}")))?;

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

#[pyfunction]
#[pyo3(signature = (seq, analyzer, ngram_min, ngram_max, n_features, idf))]
fn hashing_tfidf_csr_with_idf(
    py: Python<'_>,
    seq: Bound<PyAny>,    //iterable of strings (empty for nulls)
    analyzer: &str, //"char"/"char_wb"
    ngram_min: usize, ngram_max: usize, n_features: usize,
    idf: PyReadonlyArray1<f32>  //pre-computed IDF vector
) -> PyResult<(
    Py<PyArray1<f32>>,  //data
    Py<PyArray1<i32>>,  //indices
    Py<PyArray1<i64>>,  //indptr
    usize,              //n_rows
    usize,              //n_cols (n_features)
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

    // Get IDF slice (zero-copy read)
    let idf_slice = idf.as_slice()?;
    if idf_slice.len() != n_features {
        return Err(PyErr::new::<PyValueError, _>(
            format!("IDF length {} does not match n_features {}", idf_slice.len(), n_features)
        ));
    }

    // Work buffers to be produced by tfidf::build_csr_with_idf
    // Compute-intensive work without the GIL.
    let (data, indices, indptr) = py.allow_threads(|| {
        let builder = tfidf::Builder::new(analyzer, ngram_min, ngram_max, n_features)?;
        let out = builder.build_csr_with_idf(&docs, idf_slice);
        out
    }).map_err(to_pyerr)?;

    // Convert to NumPy without copying where possible. from_vec is zero-copy but from_array is not.
    let py_data = PyArray1::<f32>::from_vec(py, data).to_owned();
    let py_indices = PyArray1::<i32>::from_vec(py, indices).to_owned();
    let py_indptr = PyArray1::<i64>::from_vec(py, indptr).to_owned();

    Ok((Py::from(py_data), Py::from(py_indices), Py::from(py_indptr), n_rows, n_features))

}

// ---- Fit TF-IDF vocabulary + return CSR ----
#[pyfunction]
#[pyo3(signature = (seq, analyzer, ngram_min, ngram_max))]
fn tfidf_fit_csr(
    py: Python<'_>,
    seq: Vec<String>, //Fixme: reference (&PyList) instead of copying
    analyzer: &str,
    ngram_min: usize,
    ngram_max: usize,
) -> PyResult<(
    u64,               // model_id
    Py<PyArray1<f32>>, // data
    Py<PyArray1<i32>>, // indices
    Py<PyArray1<i64>>, // indptr
    usize,             // n_rows
    usize,             // n_cols (vocab size)
)> {
    let docs = seq;
    let n_rows = docs.len();

    let (model, data, indices, indptr) = py
        .allow_threads(|| {
            let builder = tfidf::VocabBuilder::new(analyzer, ngram_min, ngram_max)?;
            builder.fit_csr(&docs)
        })
        .map_err(to_pyerr)?;

    // Store model for transform and get id
    // TODO: Delete the model (memory leaks) or use pyclass to return pointer to Python
    let model_id: u64 = {
        let mut guard = TFIDF_MODELS.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("TFIDF_MODELS mutex poisoned")
        })?;
        guard.push(model);
        (guard.len() - 1) as u64
    };

    let n_cols = {
        let guard = TFIDF_MODELS.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("TFIDF_MODELS mutex poisoned")
        })?;
        guard[model_id as usize].n_cols
    };

    // Convert to NumPy (Vec -> NumPy is zero-copy for from_vec)
    let py_data = PyArray1::<f32>::from_vec(py, data).to_owned();
    let py_indices = PyArray1::<i32>::from_vec(py, indices).to_owned();
    let py_indptr = PyArray1::<i64>::from_vec(py, indptr).to_owned();

    Ok((model_id, Py::from(py_data), Py::from(py_indices), Py::from(py_indptr), n_rows, n_cols))
}

// ---- Transform using stored vocab/idf + return CSR ----
#[pyfunction]
#[pyo3(signature = (model_id, seq))]
fn tfidf_transform_csr(
    py: Python<'_>,
    model_id: u64,
    seq: Vec<String>,
) -> PyResult<(
    Py<PyArray1<f32>>, // data
    Py<PyArray1<i32>>, // indices
    Py<PyArray1<i64>>, // indptr
    usize,             // n_rows
    usize,             // n_cols
)> {
    let docs = seq;
    let n_rows = docs.len();

    // Clone only small metadata if needed; simplest: borrow model under lock then compute
    // Note: For perf, avoid holding lock during heavy compute. We'll clone the model (vocab+idf)
    // TODO: Use Vec<Arc<TfidfModel>> model storage to avoid cloning.
    let model = {
        let guard = TFIDF_MODELS.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("TFIDF_MODELS mutex poisoned")
        })?;
        let idx = model_id as usize;
        if idx >= guard.len() {
            return Err(PyValueError::new_err(format!("Invalid model_id {model_id}")));
        }
        guard[idx].clone()
    };

    let n_cols = model.n_cols;

    let (data, indices, indptr) = py
        .allow_threads(|| model.transform_csr(&docs))
        .map_err(to_pyerr)?;

    let py_data = PyArray1::<f32>::from_vec(py, data).to_owned();
    let py_indices = PyArray1::<i32>::from_vec(py, indices).to_owned();
    let py_indptr = PyArray1::<i64>::from_vec(py, indptr).to_owned();

    Ok((Py::from(py_data), Py::from(py_indices), Py::from(py_indptr), n_rows, n_cols))
}

// ---- Expose module ----
#[pymodule]
fn _rust_backend_native(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hashing_tfidf_csr, m)?)?;
    m.add_function(wrap_pyfunction!(hashing_tfidf_csr_with_idf, m)?)?;
    m.add_function(wrap_pyfunction!(fd_fit_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fd_transform_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_svd_fit_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_svd_transform_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot_encoder::ohe_transform_csr, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot_encoder::csr_to_dense, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf_fit_csr, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf_transform_csr, m)?)?;
    Ok(())
}