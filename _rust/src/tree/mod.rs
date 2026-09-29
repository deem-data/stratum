// Tree construction, split search, model storage, and RNG helpers.
mod builder;
mod exact;
mod feature_sampling;
mod forest;
mod histogram;
mod model;
mod quantization;
mod rng;

use std::sync::Arc;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use builder::{build_tree, BuildParams};
use exact::ExactSplitFinder;

pub(crate) use forest::{
    forest_fit_exact, forest_fit_hist, forest_model_info, forest_predict, ForestModelHandle,
};
pub(crate) use model::{tree_model_arrays, tree_model_from_arrays, tree_predict, TreeModelHandle};

// Fit the exact finite-value decision tree from dense float32 features.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub(crate) fn tree_fit_exact(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f32>,
    y: PyReadonlyArray1<'_, i64>,
    n_classes: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    min_impurity_decrease: f64,
    tree_seed: u32,
    max_leaf_nodes: Option<usize>,
) -> PyResult<Py<TreeModelHandle>> {
    let shape = x.shape();
    let n_rows = shape[0];
    let n_features = shape[1];
    let x = x.as_slice()?;
    let y_raw = y.as_slice()?;
    if y_raw.len() != n_rows {
        return Err(PyValueError::new_err("X and y have inconsistent lengths"));
    }
    let mut labels = Vec::with_capacity(n_rows);
    for &label in y_raw {
        if label < 0 || label as usize >= n_classes {
            return Err(PyValueError::new_err("encoded class is outside n_classes"));
        }
        labels.push(label as usize);
    }
    let weights = vec![1.0; n_rows];
    let mut finder =
        ExactSplitFinder::new(n_features, tree_seed, max_features, min_samples_leaf, 0.0);
    let model = py
        .detach(|| {
            build_tree(
                x,
                &labels,
                &weights,
                BuildParams {
                    n_features,
                    n_classes,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    min_impurity_decrease,
                    max_leaf_nodes,
                },
                &mut finder,
            )
        })
        .map_err(PyValueError::new_err)?;
    Py::new(
        py,
        TreeModelHandle {
            model: Arc::new(model),
        },
    )
}
