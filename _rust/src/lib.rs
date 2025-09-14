use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyModule};
use pyo3::{PyErr, exceptions::PyValueError};

mod tokenize;   //n-gram extraction for char/char_wb
mod hashing;    //stable fast hashing to [0, n_features)
mod tfidf;      //DF counting, IDF vector, TF*IDF, per-row L2 norm
mod csr;        //CSR builder utilities

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
#[pyo3(signature = (seq, analyzer, ngram_min, ngram_max, n_features))]
fn hashing_tfidf_csr(
    py:Python<'_>,
    seq: Bound<PyAny>,    //iterable of strings (empty for nulls)
    analyzer: &str, //"char"/"char_wb"
    ngram_min: usize,
    ngram_max: usize,
    n_features: usize
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
    //let (data, indices, indptr, idf): (Vec<f32>, Vec<i32>, Vec<i64>, Array1<f32>);

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
    //let py_idf = PyArray1::<f32>::from_array(py, &idf).to_owned();
    let py_idf = idf.into_pyarray(py).to_owned();

    Ok((Py::from(py_data), Py::from(py_indices), Py::from(py_indptr), n_rows, n_features, Py::from(py_idf)))

}

#[pymodule]
//fn skrub_rust(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
fn _rust_backend_native(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hashing_tfidf_csr, m)?)?;
    Ok(())
}