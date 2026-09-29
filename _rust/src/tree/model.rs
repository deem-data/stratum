use std::sync::Arc;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Leaf marker used by the serialized tree arrays.
pub(crate) const TREE_LEAF: i64 = -1;

// In-memory representation of a fitted tree, stored as parallel arrays.
#[derive(Debug)]
pub(crate) struct TreeModel {
    pub(crate) children_left: Vec<i64>,
    pub(crate) children_right: Vec<i64>,
    pub(crate) feature: Vec<i64>,
    pub(crate) threshold: Vec<f64>,
    pub(crate) missing_go_to_left: Vec<bool>,
    pub(crate) impurity: Vec<f64>,
    pub(crate) n_node_samples: Vec<i64>,
    pub(crate) weighted_n_node_samples: Vec<f64>,
    pub(crate) values: Vec<f64>,
    pub(crate) n_classes: usize,
    pub(crate) n_features: usize,
    pub(crate) max_depth: usize,
    pub(crate) n_leaves: usize,
    pub(crate) feature_importances: Vec<f64>,
}

impl TreeModel {
    // Validate array lengths, child pointers, and basic tree topology.
    pub(crate) fn validate(&self) -> Result<(), String> {
        let n_nodes = self.children_left.len();
        if n_nodes == 0 {
            return Err("tree must contain at least one node".into());
        }
        if self.n_classes == 0 {
            return Err("n_classes must be positive".into());
        }
        let lengths = [
            self.children_right.len(),
            self.feature.len(),
            self.threshold.len(),
            self.missing_go_to_left.len(),
            self.impurity.len(),
            self.n_node_samples.len(),
            self.weighted_n_node_samples.len(),
        ];
        if lengths.iter().any(|&length| length != n_nodes) {
            return Err("tree node arrays must have equal lengths".into());
        }
        if self.values.len() != n_nodes * self.n_classes {
            return Err("values shape does not match nodes and classes".into());
        }
        for node in 0..n_nodes {
            let left = self.children_left[node];
            let right = self.children_right[node];
            if (left == TREE_LEAF) != (right == TREE_LEAF) {
                return Err("a node must have either zero or two children".into());
            }
            if left != TREE_LEAF {
                if left < 0 || right < 0 || left as usize >= n_nodes || right as usize >= n_nodes {
                    return Err("child index is outside the tree".into());
                }
                let feature = self.feature[node];
                if feature < 0 || feature as usize >= self.n_features {
                    return Err("split feature is outside the fitted feature range".into());
                }
            }
        }
        self.topology_stats()?;
        Ok(())
    }

    // Walk the tree once to confirm it is connected, acyclic, and rooted.
    fn topology_stats(&self) -> Result<(usize, usize), String> {
        let mut state = vec![0u8; self.children_left.len()];
        let mut stack = vec![(0usize, 0usize, false)];
        let mut max_depth = 0usize;
        let mut leaves = 0usize;
        while let Some((node, depth, exiting)) = stack.pop() {
            if exiting {
                state[node] = 2;
                continue;
            }
            if state[node] == 1 {
                return Err("tree contains a cycle".into());
            }
            if state[node] == 2 {
                return Err("tree node has multiple parents".into());
            }
            state[node] = 1;
            max_depth = max_depth.max(depth);
            stack.push((node, depth, true));
            if self.children_left[node] == TREE_LEAF {
                leaves += 1;
            } else {
                stack.push((self.children_right[node] as usize, depth + 1, false));
                stack.push((self.children_left[node] as usize, depth + 1, false));
            }
        }
        if state.iter().any(|&node_state| node_state != 2) {
            return Err("tree contains unreachable nodes".into());
        }
        Ok((max_depth, leaves))
    }

    // Traverse a single feature row down to the leaf that predicts it.
    #[inline]
    pub(crate) fn leaf_for_row(&self, row: &[f32]) -> usize {
        let mut node = 0usize;
        while self.children_left[node] != TREE_LEAF {
            let value = row[self.feature[node] as usize];
            let go_left = if value.is_nan() {
                self.missing_go_to_left[node]
            } else {
                value as f64 <= self.threshold[node]
            };
            node = if go_left {
                self.children_left[node] as usize
            } else {
                self.children_right[node] as usize
            };
        }
        node
    }

    // Return class probabilities and leaf ids for every row in the input batch.
    pub(crate) fn predict(&self, x: &[f32], n_rows: usize) -> (Vec<f64>, Vec<i64>) {
        let mut probabilities = vec![0.0; n_rows * self.n_classes];
        let mut leaves = Vec::with_capacity(n_rows);
        for (row_idx, row) in x.chunks_exact(self.n_features).enumerate() {
            let leaf = self.leaf_for_row(row);
            leaves.push(leaf as i64);
            probabilities[row_idx * self.n_classes..(row_idx + 1) * self.n_classes]
                .copy_from_slice(&self.values[leaf * self.n_classes..(leaf + 1) * self.n_classes]);
        }
        (probabilities, leaves)
    }
}

// Python-owned wrapper that keeps a fitted tree alive across calls.
#[pyclass(name = "_TreeModelHandle", frozen)]
pub(crate) struct TreeModelHandle {
    pub(crate) model: Arc<TreeModel>,
}

// Build a tree handle from raw NumPy arrays produced on the Python side.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub(crate) fn tree_model_from_arrays(
    py: Python<'_>,
    children_left: PyReadonlyArray1<'_, i64>,
    children_right: PyReadonlyArray1<'_, i64>,
    feature: PyReadonlyArray1<'_, i64>,
    threshold: PyReadonlyArray1<'_, f64>,
    missing_go_to_left: PyReadonlyArray1<'_, bool>,
    values: PyReadonlyArray2<'_, f64>,
    n_features: usize,
) -> PyResult<Py<TreeModelHandle>> {
    let shape = values.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("values must be two-dimensional"));
    }
    let n_nodes = shape[0];
    let n_classes = shape[1];
    let mut model = TreeModel {
        children_left: children_left.as_slice()?.to_vec(),
        children_right: children_right.as_slice()?.to_vec(),
        feature: feature.as_slice()?.to_vec(),
        threshold: threshold.as_slice()?.to_vec(),
        missing_go_to_left: missing_go_to_left.as_slice()?.to_vec(),
        impurity: vec![0.0; n_nodes],
        n_node_samples: vec![0; n_nodes],
        weighted_n_node_samples: vec![0.0; n_nodes],
        values: values.as_slice()?.to_vec(),
        n_classes,
        n_features,
        max_depth: 0,
        n_leaves: 0,
        feature_importances: vec![0.0; n_features],
    };
    model.validate().map_err(PyValueError::new_err)?;
    (model.max_depth, model.n_leaves) = model.topology_stats().map_err(PyValueError::new_err)?;
    Py::new(
        py,
        TreeModelHandle {
            model: Arc::new(model),
        },
    )
}

// Validate the feature count and borrow the input array as a dense slice.
fn checked_input<'py>(
    model: &TreeModel,
    x: &'py PyReadonlyArray2<'py, f32>,
) -> PyResult<(&'py [f32], usize)> {
    let shape = x.shape();
    if shape[1] != model.n_features {
        return Err(PyValueError::new_err(format!(
            "X has {} features, but the tree expects {}",
            shape[1], model.n_features
        )));
    }
    Ok((x.as_slice()?, shape[0]))
}
// Predict probabilities and leaf ids through the native tree model.

#[pyfunction]
pub(crate) fn tree_predict(
    py: Python<'_>,
    model: PyRef<'_, TreeModelHandle>,
    x: PyReadonlyArray2<'_, f32>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<i64>>)> {
    let model = Arc::clone(&model.model);
    let (x, n_rows) = checked_input(&model, &x)?;
    let (probabilities, leaves) = py.detach(|| model.predict(x, n_rows));
    let probabilities = ndarray::Array2::from_shape_vec((n_rows, model.n_classes), probabilities)
        .expect("prediction output shape is exact");
    Ok((
        Py::from(probabilities.into_pyarray(py).to_owned()),
        Py::from(PyArray1::from_vec(py, leaves).to_owned()),
    ))
}

// Expose the model arrays for inspection, debugging, and parity tests.
#[pyfunction]
pub(crate) fn tree_model_arrays(
    py: Python<'_>,
    model: PyRef<'_, TreeModelHandle>,
) -> PyResult<Py<PyAny>> {
    let model = &model.model;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "children_left",
        PyArray1::from_vec(py, model.children_left.clone()),
    )?;
    dict.set_item(
        "children_right",
        PyArray1::from_vec(py, model.children_right.clone()),
    )?;
    dict.set_item("feature", PyArray1::from_vec(py, model.feature.clone()))?;
    dict.set_item("threshold", PyArray1::from_vec(py, model.threshold.clone()))?;
    dict.set_item(
        "missing_go_to_left",
        PyArray1::from_vec(py, model.missing_go_to_left.clone()),
    )?;
    dict.set_item("impurity", PyArray1::from_vec(py, model.impurity.clone()))?;
    dict.set_item(
        "n_node_samples",
        PyArray1::from_vec(py, model.n_node_samples.clone()),
    )?;
    dict.set_item(
        "weighted_n_node_samples",
        PyArray1::from_vec(py, model.weighted_n_node_samples.clone()),
    )?;
    let values = ndarray::Array2::from_shape_vec(
        (model.children_left.len(), model.n_classes),
        model.values.clone(),
    )
    .expect("validated model value shape");
    dict.set_item("value", values.into_pyarray(py))?;
    dict.set_item("max_depth", model.max_depth)?;
    dict.set_item("n_leaves", model.n_leaves)?;
    dict.set_item(
        "feature_importances",
        PyArray1::from_vec(py, model.feature_importances.clone()),
    )?;
    Ok(dict.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_multiclass_tree_and_nan_direction() {
        let model = TreeModel {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![0, -2, -2],
            threshold: vec![0.5, -2.0, -2.0],
            missing_go_to_left: vec![true, false, false],
            impurity: vec![0.0; 3],
            n_node_samples: vec![0; 3],
            weighted_n_node_samples: vec![0.0; 3],
            values: vec![0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.8, 0.1, 0.1],
            n_classes: 3,
            n_features: 1,
            max_depth: 1,
            n_leaves: 2,
            feature_importances: vec![1.0],
        };
        model.validate().unwrap();
        let (proba, leaves) = model.predict(&[0.0, 1.0, f32::NAN], 3);
        assert_eq!(leaves, vec![1, 2, 1]);
        assert_eq!(&proba[0..3], &[0.1, 0.7, 0.2]);
    }

    #[test]
    fn rejects_cyclic_model_arrays() {
        let model = TreeModel {
            children_left: vec![0, -1],
            children_right: vec![1, -1],
            feature: vec![0, -2],
            threshold: vec![0.5, -2.0],
            missing_go_to_left: vec![false; 2],
            impurity: vec![0.0; 2],
            n_node_samples: vec![0; 2],
            weighted_n_node_samples: vec![0.0; 2],
            values: vec![0.0; 4],
            n_classes: 2,
            n_features: 1,
            max_depth: 0,
            n_leaves: 0,
            feature_importances: vec![0.0],
        };
        assert_eq!(model.validate().unwrap_err(), "tree contains a cycle");
    }
}
