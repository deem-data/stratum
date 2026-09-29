use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use super::builder::{build_tree, BuildParams};
use super::exact::ExactSplitFinder;
use super::histogram::{HistogramScratch, HistogramSplitFinder};
use super::model::TreeModel;
use super::quantization::QuantizedData;
use super::rng::NumpyRng;
use crate::threads::get_bounded_thread_pool;

const PARALLEL_PREDICTION_ROWS: usize = 1_024;

// Immutable forest sharing the compact tree representation and traversal.
pub(crate) struct ForestModel {
    trees: Vec<TreeModel>,
    tree_seeds: Vec<u32>,
    n_classes: usize,
    n_features: usize,
    worker_budget: usize,
    training_info: ForestTrainingInfo,
}

struct ForestTrainingInfo {
    split_backend: &'static str,
    requested_bins: usize,
    quantization_count: usize,
    quantization_seconds: f64,
    tree_fit_seconds: f64,
    binned_matrix_bytes: usize,
    scratch_bytes: usize,
    actual_bins_min: usize,
    actual_bins_median: f64,
    actual_bins_max: usize,
}

#[pyclass(name = "_ForestModelHandle", frozen)]
pub(crate) struct ForestModelHandle {
    model: Arc<ForestModel>,
}

impl ForestModel {
    fn predict(&self, x: &[f32], n_rows: usize) -> Vec<f64> {
        let mut probabilities = vec![0.0; n_rows * self.n_classes];
        let scale = 1.0 / self.trees.len() as f64;
        let predict_row = |target: &mut [f64], row: &[f32]| {
            for tree in &self.trees {
                let leaf = tree.leaf_for_row(row);
                let source = &tree.values[leaf * self.n_classes..(leaf + 1) * self.n_classes];
                for class in 0..self.n_classes {
                    target[class] += source[class] * scale;
                }
            }
        };
        if self.worker_budget > 1 && n_rows >= PARALLEL_PREDICTION_ROWS {
            let pool = get_bounded_thread_pool(self.worker_budget)
                .expect("fitted worker budget was already validated");
            pool.install(|| {
                probabilities
                    .par_chunks_mut(self.n_classes)
                    .zip(x.par_chunks(self.n_features))
                    .for_each(|(target, row)| predict_row(target, row));
            });
        } else {
            probabilities
                .chunks_mut(self.n_classes)
                .zip(x.chunks(self.n_features))
                .for_each(|(target, row)| predict_row(target, row));
        }
        probabilities
    }
}

// Fit trees in bounded, long-lived jobs. Each worker processes one contiguous
// seed batch serially and owns backend-specific reusable scratch.
#[allow(clippy::too_many_arguments)]
fn fit_forest_batched<S, MakeState, FitTree>(
    pool: &rayon::ThreadPool,
    n_rows: usize,
    seeds: &[i64],
    bootstrap: bool,
    n_bootstrap: usize,
    make_state: MakeState,
    fit_tree: FitTree,
) -> Result<Vec<TreeModel>, String>
where
    S: Send,
    MakeState: Fn() -> S + Sync,
    FitTree: Fn(&mut S, u32, &[f64]) -> Result<TreeModel, String> + Sync,
{
    let workers = pool.current_num_threads().min(seeds.len());
    let batch_size = seeds.len().div_ceil(workers);
    let batches: Vec<&[i64]> = seeds.chunks(batch_size).collect();
    let built_batches: Result<Vec<Vec<TreeModel>>, String> = pool.install(|| {
        batches
            .par_iter()
            .map(|batch| {
                let mut state = make_state();
                let mut trees = Vec::with_capacity(batch.len());
                let mut weights = vec![1.0; n_rows];
                for &seed in batch.iter() {
                    let seed = seed as u32;
                    weights.fill(if bootstrap { 0.0 } else { 1.0 });
                    if bootstrap {
                        let mut bootstrap_rng = NumpyRng::new(seed);
                        for _ in 0..n_bootstrap {
                            weights[bootstrap_rng.interval((n_rows - 1) as u32) as usize] += 1.0;
                        }
                    }
                    // Bootstrap and split randomness use independent streams
                    // initialized from the same ordered per-tree seed.
                    let split_seed = NumpyRng::new(seed).interval(i32::MAX as u32 - 1);
                    trees.push(fit_tree(&mut state, split_seed, &weights)?);
                }
                Ok(trees)
            })
            .collect()
    });
    let mut trees = Vec::with_capacity(seeds.len());
    for batch in built_batches? {
        trees.extend(batch);
    }
    Ok(trees)
}

fn encoded_labels(y_raw: &[i64], n_rows: usize, n_classes: usize) -> PyResult<Vec<usize>> {
    if y_raw.len() != n_rows {
        return Err(PyValueError::new_err("X and y have inconsistent lengths"));
    }
    y_raw
        .iter()
        .map(|&label| {
            if label < 0 || label as usize >= n_classes {
                Err(PyValueError::new_err("encoded class is outside n_classes"))
            } else {
                Ok(label as usize)
            }
        })
        .collect()
}

fn validate_forest_inputs(
    n_rows: usize,
    n_features: usize,
    n_classes: usize,
    max_features: usize,
    seeds: &[i64],
    bootstrap: bool,
    n_bootstrap: usize,
) -> PyResult<()> {
    if n_rows == 0 || n_features == 0 || seeds.is_empty() {
        return Err(PyValueError::new_err(
            "training arrays and tree seeds must be non-empty",
        ));
    }
    if n_classes == 0 {
        return Err(PyValueError::new_err("n_classes must be positive"));
    }
    if max_features == 0 || max_features > n_features {
        return Err(PyValueError::new_err(
            "max_features must be in 1..=n_features",
        ));
    }
    if bootstrap && n_bootstrap == 0 {
        return Err(PyValueError::new_err(
            "n_bootstrap must be positive when bootstrap=True",
        ));
    }
    if seeds
        .iter()
        .any(|&seed| seed < 0 || seed >= i32::MAX as i64)
    {
        return Err(PyValueError::new_err("tree seeds must be in 0..i32::MAX"));
    }
    Ok(())
}

// Exact entry point retained unchanged apart from the shared forest scheduler.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub(crate) fn forest_fit_exact(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f32>,
    y: PyReadonlyArray1<'_, i64>,
    tree_seeds: PyReadonlyArray1<'_, i64>,
    n_classes: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    min_impurity_decrease: f64,
    max_leaf_nodes: Option<usize>,
    bootstrap: bool,
    n_bootstrap: usize,
    worker_budget: usize,
) -> PyResult<Py<ForestModelHandle>> {
    let shape = x.shape();
    let n_rows = shape[0];
    let n_features = shape[1];
    let x = x.as_slice()?;
    let y_raw = y.as_slice()?;
    let seeds = tree_seeds.as_slice()?;
    validate_forest_inputs(
        n_rows,
        n_features,
        n_classes,
        max_features,
        seeds,
        bootstrap,
        n_bootstrap,
    )?;
    let labels = encoded_labels(y_raw, n_rows, n_classes)?;
    let pool = get_bounded_thread_pool(worker_budget).map_err(PyValueError::new_err)?;
    let params = BuildParams {
        n_features,
        n_classes,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        max_leaf_nodes,
    };
    let result = py
        .detach(|| {
            let fit_start = Instant::now();
            let trees = fit_forest_batched(
                &pool,
                n_rows,
                seeds,
                bootstrap,
                n_bootstrap,
                || (),
                |_, split_seed, weights| {
                    let mut finder = ExactSplitFinder::new(
                        n_features,
                        split_seed,
                        max_features,
                        min_samples_leaf,
                        0.0,
                    );
                    build_tree(x, &labels, weights, params, &mut finder)
                },
            )?;
            Ok::<_, String>(ForestModel {
                trees,
                tree_seeds: seeds.iter().map(|&seed| seed as u32).collect(),
                n_classes,
                n_features,
                worker_budget,
                training_info: ForestTrainingInfo {
                    split_backend: "exact",
                    requested_bins: 0,
                    quantization_count: 0,
                    quantization_seconds: 0.0,
                    tree_fit_seconds: fit_start.elapsed().as_secs_f64(),
                    binned_matrix_bytes: 0,
                    scratch_bytes: 0,
                    actual_bins_min: 0,
                    actual_bins_median: 0.0,
                    actual_bins_max: 0,
                },
            })
        })
        .map_err(PyValueError::new_err)?;
    Py::new(
        py,
        ForestModelHandle {
            model: Arc::new(result),
        },
    )
}

// Build one shared quantized matrix, then fit histogram trees through the same
// bootstrap scheduler, builder, model, and prediction path as the exact forest.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub(crate) fn forest_fit_hist(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f32>,
    y: PyReadonlyArray1<'_, i64>,
    tree_seeds: PyReadonlyArray1<'_, i64>,
    n_classes: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    min_impurity_decrease: f64,
    max_leaf_nodes: Option<usize>,
    bootstrap: bool,
    n_bootstrap: usize,
    worker_budget: usize,
    n_bins: usize,
) -> PyResult<Py<ForestModelHandle>> {
    let shape = x.shape();
    let n_rows = shape[0];
    let n_features = shape[1];
    let x = x.as_slice()?;
    let y_raw = y.as_slice()?;
    let seeds = tree_seeds.as_slice()?;
    validate_forest_inputs(
        n_rows,
        n_features,
        n_classes,
        max_features,
        seeds,
        bootstrap,
        n_bootstrap,
    )?;
    let labels = encoded_labels(y_raw, n_rows, n_classes)?;
    let pool = get_bounded_thread_pool(worker_budget).map_err(PyValueError::new_err)?;
    let params = BuildParams {
        n_features,
        n_classes,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        max_leaf_nodes,
    };
    let result = py
        .detach(|| {
            let quantization_start = Instant::now();
            let quantized = QuantizedData::from_row_major(x, n_rows, n_features, n_bins)?;
            let quantization_seconds = quantization_start.elapsed().as_secs_f64();
            let (actual_bins_min, actual_bins_median, actual_bins_max) =
                quantized.actual_bin_summary();
            let fit_start = Instant::now();
            let trees = fit_forest_batched(
                &pool,
                n_rows,
                seeds,
                bootstrap,
                n_bootstrap,
                HistogramScratch::default,
                |scratch, split_seed, weights| {
                    let mut finder = HistogramSplitFinder::new(
                        &quantized,
                        split_seed,
                        max_features,
                        min_samples_leaf,
                        0.0,
                        scratch,
                    );
                    build_tree(x, &labels, weights, params, &mut finder)
                },
            )?;
            let workers = worker_budget.min(seeds.len());
            let batch_size = seeds.len().div_ceil(workers);
            let active_batches = seeds.len().div_ceil(batch_size);
            let scratch_per_worker = (actual_bins_max + 1) * (n_classes + 1) * size_of::<u64>()
                + 3 * n_classes * size_of::<u64>()
                + (actual_bins_max + 1) * size_of::<usize>();
            Ok::<_, String>(ForestModel {
                trees,
                tree_seeds: seeds.iter().map(|&seed| seed as u32).collect(),
                n_classes,
                n_features,
                worker_budget,
                training_info: ForestTrainingInfo {
                    split_backend: "histogram",
                    requested_bins: n_bins,
                    quantization_count: 1,
                    quantization_seconds,
                    tree_fit_seconds: fit_start.elapsed().as_secs_f64(),
                    binned_matrix_bytes: quantized.matrix_bytes(),
                    scratch_bytes: active_batches * scratch_per_worker,
                    actual_bins_min,
                    actual_bins_median,
                    actual_bins_max,
                },
            })
        })
        .map_err(PyValueError::new_err)?;
    Py::new(
        py,
        ForestModelHandle {
            model: Arc::new(result),
        },
    )
}

#[pyfunction]
pub(crate) fn forest_predict(
    py: Python<'_>,
    model: PyRef<'_, ForestModelHandle>,
    x: PyReadonlyArray2<'_, f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    let shape = x.shape();
    if shape[1] != model.model.n_features {
        return Err(PyValueError::new_err(format!(
            "X has {} features, but the forest expects {}",
            shape[1], model.model.n_features
        )));
    }
    let n_rows = shape[0];
    let x = x.as_slice()?;
    let model = Arc::clone(&model.model);
    let probabilities = py.detach(|| model.predict(x, n_rows));
    let output = Array2::from_shape_vec((n_rows, model.n_classes), probabilities)
        .expect("forest prediction output shape is exact");
    Ok(Py::from(output.into_pyarray(py).to_owned()))
}

#[pyfunction]
pub(crate) fn forest_model_info(
    py: Python<'_>,
    model: PyRef<'_, ForestModelHandle>,
) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    let n_nodes: usize = model
        .model
        .trees
        .iter()
        .map(|tree| tree.children_left.len())
        .sum();
    let n_leaves: usize = model.model.trees.iter().map(|tree| tree.n_leaves).sum();
    let max_depth = model
        .model
        .trees
        .iter()
        .map(|tree| tree.max_depth)
        .max()
        .unwrap_or(0);
    let model_bytes: usize = model
        .model
        .trees
        .iter()
        .map(|tree| {
            tree.children_left.len()
                * (3 * size_of::<i64>() + 4 * size_of::<f64>() + size_of::<bool>())
                + tree.values.len() * size_of::<f64>()
                + tree.feature_importances.len() * size_of::<f64>()
        })
        .sum();
    let mut importances = vec![0.0; model.model.n_features];
    for tree in &model.model.trees {
        for (target, source) in importances.iter_mut().zip(&tree.feature_importances) {
            *target += source / model.model.trees.len() as f64;
        }
    }
    dict.set_item("n_estimators", model.model.trees.len())?;
    dict.set_item("n_nodes", n_nodes)?;
    dict.set_item("n_leaves", n_leaves)?;
    dict.set_item("max_depth", max_depth)?;
    dict.set_item("model_bytes", model_bytes)?;
    dict.set_item("worker_budget", model.model.worker_budget)?;
    dict.set_item("split_backend", model.model.training_info.split_backend)?;
    dict.set_item("requested_bins", model.model.training_info.requested_bins)?;
    dict.set_item(
        "quantization_count",
        model.model.training_info.quantization_count,
    )?;
    dict.set_item(
        "quantization_seconds",
        model.model.training_info.quantization_seconds,
    )?;
    dict.set_item(
        "tree_fit_seconds",
        model.model.training_info.tree_fit_seconds,
    )?;
    dict.set_item(
        "binned_matrix_bytes",
        model.model.training_info.binned_matrix_bytes,
    )?;
    dict.set_item("scratch_bytes", model.model.training_info.scratch_bytes)?;
    dict.set_item("actual_bins_min", model.model.training_info.actual_bins_min)?;
    dict.set_item(
        "actual_bins_median",
        model.model.training_info.actual_bins_median,
    )?;
    dict.set_item("actual_bins_max", model.model.training_info.actual_bins_max)?;
    dict.set_item(
        "tree_seeds",
        PyArray1::from_vec(py, model.model.tree_seeds.clone()),
    )?;
    dict.set_item(
        "root_n_node_samples",
        PyArray1::from_vec(
            py,
            model
                .model
                .trees
                .iter()
                .map(|tree| tree.n_node_samples[0])
                .collect(),
        ),
    )?;
    dict.set_item(
        "root_weighted_n_node_samples",
        PyArray1::from_vec(
            py,
            model
                .model
                .trees
                .iter()
                .map(|tree| tree.weighted_n_node_samples[0])
                .collect(),
        ),
    )?;
    dict.set_item("feature_importances", PyArray1::from_vec(py, importances))?;
    Ok(dict.into_any().unbind())
}
