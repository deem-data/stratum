use pyo3::prelude::*;
use pyo3::types::PyModule;

// Core Rust backend modules exposed through the Python extension.
mod countvectorizer;
mod csr;
mod fd;
mod hashing;
mod one_hot_encoder;
mod standard_scaler;
mod tfidf;
mod threads;
mod timing;
mod tokenize;
mod tree;
mod truncated_svd;

// Register Rust classes and pyfunctions on the native extension module.
#[pymodule]
fn _rust_backend_native(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<countvectorizer::CountVectorizerModelHandle>()?;
    m.add_class::<tfidf::TfidfModelHandle>()?;
    m.add_class::<fd::FdEmbedModel>()?;
    m.add_class::<truncated_svd::TruncatedSvdModel>()?;
    m.add_class::<tree::TreeModelHandle>()?;
    m.add_class::<tree::ForestModelHandle>()?;
    m.add_function(wrap_pyfunction!(tfidf::hashing_tfidf_csr, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf::hashing_tfidf_csr_with_idf, m)?)?;
    m.add_function(wrap_pyfunction!(fd::fd_fit_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fd::fd_transform_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_svd::truncated_svd_fit_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_svd::truncated_svd_transform_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot_encoder::ohe_transform_csr, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot_encoder::csr_to_dense, m)?)?;
    m.add_function(wrap_pyfunction!(standard_scaler::standard_scale_fit, m)?)?;
    m.add_function(wrap_pyfunction!(standard_scaler::standard_scale_transform, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf::tfidf_fit_csr, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf::tfidf_transform_csr, m)?)?;
    m.add_function(wrap_pyfunction!(tree::tree_fit_exact, m)?)?;
    m.add_function(wrap_pyfunction!(tree::tree_model_from_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(tree::tree_predict, m)?)?;
    m.add_function(wrap_pyfunction!(tree::tree_model_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(tree::forest_fit_exact, m)?)?;
    m.add_function(wrap_pyfunction!(tree::forest_fit_hist, m)?)?;
    m.add_function(wrap_pyfunction!(tree::forest_predict, m)?)?;
    m.add_function(wrap_pyfunction!(tree::forest_model_info, m)?)?;
    m.add_function(wrap_pyfunction!(countvectorizer::count_vectorize_fit, m)?)?;
    m.add_function(wrap_pyfunction!(countvectorizer::count_vectorize_transform, m)?)?;
    m.add_function(wrap_pyfunction!(countvectorizer::count_vectorize_fit_transform, m)?)?;
    Ok(())
}
