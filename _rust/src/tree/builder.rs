use super::exact::gini;
use super::model::{TreeModel, TREE_LEAF};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const EPSILON: f64 = f64::EPSILON;

// Per-node statistics gathered from the current training slice.
#[derive(Clone)]
pub(crate) struct NodeStats {
    pub(crate) class_weights: Vec<f64>,
    pub(crate) weight: f64,
    pub(crate) impurity: f64,
}

// A candidate split returned by the split search implementation.
#[derive(Clone, Debug)]
pub(crate) struct Split {
    pub(crate) feature: usize,
    pub(crate) threshold: f64,
    pub(crate) missing_go_to_left: bool,
    pub(crate) improvement: f64,
    pub(crate) left_impurity: f64,
    pub(crate) right_impurity: f64,
}

// Split-search output plus any constants that should be carried forward.
pub(crate) struct SplitSearch {
    pub(crate) split: Option<Split>,
    pub(crate) constants: Vec<usize>,
}

// Pluggable strategy used by `build_tree` to choose the best node split.
pub(crate) trait SplitFinder {
    fn best_split(
        &mut self,
        x: &[f32],
        n_features: usize,
        rows: &[usize],
        y: &[usize],
        weights: &[f64],
        n_classes: usize,
        parent: &NodeStats,
        known_constants: &[usize],
    ) -> Result<SplitSearch, String>;
}

// Hyperparameters and basic shape constraints for tree construction.
#[derive(Clone, Copy)]
pub(crate) struct BuildParams {
    pub(crate) n_features: usize,
    pub(crate) n_classes: usize,
    pub(crate) max_depth: usize,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) min_impurity_decrease: f64,
    pub(crate) max_leaf_nodes: Option<usize>,
}

// Temporary mutable storage used while assembling the final `TreeModel`.
struct MutableTree {
    left: Vec<i64>,
    right: Vec<i64>,
    feature: Vec<i64>,
    threshold: Vec<f64>,
    missing_left: Vec<bool>,
    impurity: Vec<f64>,
    n_samples: Vec<i64>,
    weighted_samples: Vec<f64>,
    values: Vec<f64>,
    importances: Vec<f64>,
    max_depth: usize,
    n_leaves: usize,
}

impl MutableTree {
    // Create an empty tree with per-feature importance storage.
    fn new(n_features: usize) -> Self {
        Self {
            left: vec![],
            right: vec![],
            feature: vec![],
            threshold: vec![],
            missing_left: vec![],
            impurity: vec![],
            n_samples: vec![],
            weighted_samples: vec![],
            values: vec![],
            importances: vec![0.0; n_features],
            max_depth: 0,
            n_leaves: 0,
        }
    }

    // Append a node and initialize it as a leaf placeholder.
    fn add_node(&mut self, stats: &NodeStats, n_samples: usize, n_classes: usize) -> usize {
        let node = self.left.len();
        self.left.push(TREE_LEAF);
        self.right.push(TREE_LEAF);
        self.feature.push(-2);
        self.threshold.push(-2.0);
        self.missing_left.push(false);
        self.impurity.push(stats.impurity);
        self.n_samples.push(n_samples as i64);
        self.weighted_samples.push(stats.weight);
        self.values
            .extend(stats.class_weights.iter().map(|v| v / stats.weight));
        debug_assert_eq!(self.values.len(), (node + 1) * n_classes);
        node
    }
}

// Build a decision tree by expanding active row slices from a work stack.
pub(crate) fn build_tree<F: SplitFinder>(
    x: &[f32],
    y: &[usize],
    weights: &[f64],
    params: BuildParams,
    finder: &mut F,
) -> Result<TreeModel, String> {
    // Validate input shapes and discard zero-weight rows up front.
    let n_rows = y.len();
    if x.len() != n_rows * params.n_features || weights.len() != n_rows {
        return Err("training arrays have inconsistent shapes".into());
    }
    let mut rows: Vec<usize> = (0..n_rows).filter(|&row| weights[row] != 0.0).collect();
    if rows.is_empty() {
        return Err("at least one positive-weight row is required".into());
    }
    let total_weight: f64 = rows.iter().map(|&row| weights[row]).sum();
    if let Some(max_leaf_nodes) = params.max_leaf_nodes {
        return build_tree_best_first(
            x,
            y,
            weights,
            params,
            finder,
            rows,
            total_weight,
            max_leaf_nodes,
        );
    }

    // The stack stores the active row intervals that still need processing.
    let mut tree = MutableTree::new(params.n_features);
    let active_rows = rows.len();
    let mut work = vec![BuildFrame {
        start: 0,
        end: active_rows,
        depth: 0,
        constants: Vec::new(),
        parent: None,
    }];

    // Expand nodes depth-first until every frame becomes a leaf or split.
    while let Some(frame) = work.pop() {
        // Compute impurity statistics for the current slice and create a node.
        let stats = node_stats(&rows[frame.start..frame.end], y, weights, params.n_classes);
        let node = tree.add_node(&stats, frame.end - frame.start, params.n_classes);

        // Link this node back to its parent, if any.
        if let Some((parent, is_left)) = frame.parent {
            if is_left {
                tree.left[parent] = node as i64;
            } else {
                tree.right[parent] = node as i64;
            }
        }

        // Track the deepest frame visited so far.
        tree.max_depth = tree.max_depth.max(frame.depth);

        // Stop immediately if the node is too shallow, too small, or pure.
        let should_stop = frame.depth >= params.max_depth
            || frame.end - frame.start < params.min_samples_split
            || frame.end - frame.start < 2 * params.min_samples_leaf
            || stats.impurity <= 0.0;
        if should_stop {
            tree.n_leaves += 1;
            continue;
        }

        // Ask the configured splitter for the best candidate on this slice.
        let search = finder.best_split(
            x,
            params.n_features,
            &rows[frame.start..frame.end],
            y,
            weights,
            params.n_classes,
            &stats,
            &frame.constants,
        )?;
        let Some(split) = search.split else {
            tree.n_leaves += 1;
            continue;
        };

        // Compare the local gain against the global impurity-decrease threshold.
        let weighted_improvement = stats.weight / total_weight * split.improvement;
        if weighted_improvement + EPSILON < params.min_impurity_decrease {
            tree.n_leaves += 1;
            continue;
        }

        // Partition the active rows in-place so both child slices stay contiguous.
        let mut boundary = frame.start;
        let mut right = frame.end;
        while boundary < right {
            let row = rows[boundary];
            let value = x[row * params.n_features + split.feature];
            let goes_left = if value.is_nan() {
                split.missing_go_to_left
            } else {
                value as f64 <= split.threshold
            };
            if goes_left {
                boundary += 1;
            } else {
                right -= 1;
                rows.swap(boundary, right);
            }
        }
        if boundary == frame.start || boundary == frame.end {
            return Err("split produced an empty child".into());
        }

        // Store split metadata and accumulate feature importance.
        tree.feature[node] = split.feature as i64;
        tree.threshold[node] = split.threshold;
        tree.missing_left[node] = split.missing_go_to_left;

        // Push right first so the left subtree is built next. This preserves
        // sklearn's depth-first RNG consumption and preorder node numbering.
        work.push(BuildFrame {
            start: boundary,
            end: frame.end,
            depth: frame.depth + 1,
            constants: search.constants.clone(),
            parent: Some((node, false)),
        });
        work.push(BuildFrame {
            start: frame.start,
            end: boundary,
            depth: frame.depth + 1,
            constants: search.constants,
            parent: Some((node, true)),
        });
        let _ = (split.left_impurity, split.right_impurity);
    }

    // Materialize the immutable model and validate its internal consistency.
    let mut model = TreeModel {
        children_left: tree.left,
        children_right: tree.right,
        feature: tree.feature,
        threshold: tree.threshold,
        missing_go_to_left: tree.missing_left,
        impurity: tree.impurity,
        n_node_samples: tree.n_samples,
        weighted_n_node_samples: tree.weighted_samples,
        values: tree.values,
        n_classes: params.n_classes,
        n_features: params.n_features,
        max_depth: tree.max_depth,
        n_leaves: tree.n_leaves,
        feature_importances: tree.importances,
    };
    compute_feature_importances(&mut model);
    model.validate()?;
    Ok(model)
}

// A prepared best-first frontier node. Ordering intentionally compares only
// improvement, matching sklearn's heap contract for equal-gain candidates.
struct FrontierNode {
    node: usize,
    start: usize,
    end: usize,
    depth: usize,
    stats: NodeStats,
    split: Split,
    child_constants: Vec<usize>,
}

impl PartialEq for FrontierNode {
    fn eq(&self, other: &Self) -> bool {
        self.split.improvement == other.split.improvement
    }
}

impl Eq for FrontierNode {}

impl PartialOrd for FrontierNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.split.improvement.partial_cmp(&other.split.improvement)
    }
}

impl Ord for FrontierNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// Evaluate a node at creation time so feature RNG consumption matches the
// best-first sklearn builder, including candidates that never get expanded.
#[allow(clippy::too_many_arguments)]
fn prepare_frontier_node<F: SplitFinder>(
    x: &[f32],
    y: &[usize],
    weights: &[f64],
    params: BuildParams,
    finder: &mut F,
    rows: &[usize],
    start: usize,
    end: usize,
    depth: usize,
    node: usize,
    stats: NodeStats,
    constants: &[usize],
    total_weight: f64,
) -> Result<Option<FrontierNode>, String> {
    if depth >= params.max_depth
        || end - start < params.min_samples_split
        || end - start < 2 * params.min_samples_leaf
        || stats.impurity <= 0.0
    {
        return Ok(None);
    }
    let search = finder.best_split(
        x,
        params.n_features,
        &rows[start..end],
        y,
        weights,
        params.n_classes,
        &stats,
        constants,
    )?;
    let Some(split) = search.split else {
        return Ok(None);
    };
    let weighted_improvement = stats.weight / total_weight * split.improvement;
    if weighted_improvement + EPSILON < params.min_impurity_decrease {
        return Ok(None);
    }
    Ok(Some(FrontierNode {
        node,
        start,
        end,
        depth,
        stats,
        split: Split {
            improvement: weighted_improvement,
            ..split
        },
        child_constants: search.constants,
    }))
}

// Build a max-leaf-limited tree by globally expanding the best frontier node.
#[allow(clippy::too_many_arguments)]
fn build_tree_best_first<F: SplitFinder>(
    x: &[f32],
    y: &[usize],
    weights: &[f64],
    params: BuildParams,
    finder: &mut F,
    mut rows: Vec<usize>,
    total_weight: f64,
    max_leaf_nodes: usize,
) -> Result<TreeModel, String> {
    let mut tree = MutableTree::new(params.n_features);
    let root_stats = node_stats(&rows, y, weights, params.n_classes);
    let root = tree.add_node(&root_stats, rows.len(), params.n_classes);
    let mut frontier = BinaryHeap::new();
    if let Some(candidate) = prepare_frontier_node(
        x,
        y,
        weights,
        params,
        finder,
        &rows,
        0,
        rows.len(),
        0,
        root,
        root_stats,
        &[],
        total_weight,
    )? {
        frontier.push(candidate);
    }
    let mut n_leaves = 1usize;

    while n_leaves < max_leaf_nodes {
        let Some(candidate) = frontier.pop() else {
            break;
        };
        let split = &candidate.split;
        let mut boundary = candidate.start;
        let mut right = candidate.end;
        while boundary < right {
            let row = rows[boundary];
            let value = x[row * params.n_features + split.feature];
            let goes_left = if value.is_nan() {
                split.missing_go_to_left
            } else {
                value as f64 <= split.threshold
            };
            if goes_left {
                boundary += 1;
            } else {
                right -= 1;
                rows.swap(boundary, right);
            }
        }
        if boundary == candidate.start || boundary == candidate.end {
            return Err("split produced an empty child".into());
        }

        tree.feature[candidate.node] = split.feature as i64;
        tree.threshold[candidate.node] = split.threshold;
        tree.missing_left[candidate.node] = split.missing_go_to_left;

        // Children are allocated and evaluated left-to-right before either is
        // queued, preserving sklearn's node numbering and splitter RNG order.
        for (start, end, is_left) in [
            (candidate.start, boundary, true),
            (boundary, candidate.end, false),
        ] {
            let stats = node_stats(&rows[start..end], y, weights, params.n_classes);
            let child = tree.add_node(&stats, end - start, params.n_classes);
            if is_left {
                tree.left[candidate.node] = child as i64;
            } else {
                tree.right[candidate.node] = child as i64;
            }
            let depth = candidate.depth + 1;
            tree.max_depth = tree.max_depth.max(depth);
            if let Some(next) = prepare_frontier_node(
                x,
                y,
                weights,
                params,
                finder,
                &rows,
                start,
                end,
                depth,
                child,
                stats,
                &candidate.child_constants,
                total_weight,
            )? {
                frontier.push(next);
            }
        }
        n_leaves += 1;
        let _ = (&candidate.stats, split.left_impurity, split.right_impurity);
    }

    tree.n_leaves = n_leaves;
    let mut model = TreeModel {
        children_left: tree.left,
        children_right: tree.right,
        feature: tree.feature,
        threshold: tree.threshold,
        missing_go_to_left: tree.missing_left,
        impurity: tree.impurity,
        n_node_samples: tree.n_samples,
        weighted_n_node_samples: tree.weighted_samples,
        values: tree.values,
        n_classes: params.n_classes,
        n_features: params.n_features,
        max_depth: tree.max_depth,
        n_leaves: tree.n_leaves,
        feature_importances: tree.importances,
    };
    compute_feature_importances(&mut model);
    model.validate()?;
    Ok(model)
}

// Derive importances from finalized child statistics, matching sklearn's
// fitted-tree calculation and avoiding split-search rounding artifacts.
fn compute_feature_importances(model: &mut TreeModel) {
    model.feature_importances.fill(0.0);
    for node in 0..model.children_left.len() {
        if model.children_left[node] == TREE_LEAF {
            continue;
        }
        let left = model.children_left[node] as usize;
        let right = model.children_right[node] as usize;
        model.feature_importances[model.feature[node] as usize] +=
            model.weighted_n_node_samples[node] * model.impurity[node]
                - model.weighted_n_node_samples[left] * model.impurity[left]
                - model.weighted_n_node_samples[right] * model.impurity[right];
    }
    let sum: f64 = model.feature_importances.iter().sum();
    if sum > 0.0 {
        for value in &mut model.feature_importances {
            *value /= sum;
        }
    }
}

// A stack frame describing one contiguous slice of active rows.
struct BuildFrame {
    start: usize,
    end: usize,
    depth: usize,
    constants: Vec<usize>,
    parent: Option<(usize, bool)>,
}

// Compute class counts, total weight, and impurity for a node slice.
fn node_stats(rows: &[usize], y: &[usize], weights: &[f64], n_classes: usize) -> NodeStats {
    let mut class_weights = vec![0.0; n_classes];
    let mut weight = 0.0;
    for &row in rows {
        class_weights[y[row]] += weights[row];
        weight += weights[row];
    }
    NodeStats {
        impurity: gini(&class_weights, weight),
        class_weights,
        weight,
    }
}

// A small unit test that verifies preorder layout and weighted accounting.
#[cfg(test)]
mod tests {
    use super::*;

    struct RootSplit;
    impl SplitFinder for RootSplit {
        fn best_split(
            &mut self,
            _x: &[f32],
            _nf: usize,
            rows: &[usize],
            _y: &[usize],
            _w: &[f64],
            _nc: usize,
            _parent: &NodeStats,
            constants: &[usize],
        ) -> Result<SplitSearch, String> {
            Ok(SplitSearch {
                split: (rows.len() > 2).then_some(Split {
                    feature: 0,
                    threshold: 1.5,
                    missing_go_to_left: false,
                    improvement: 0.5,
                    left_impurity: 0.0,
                    right_impurity: 0.0,
                }),
                constants: constants.to_vec(),
            })
        }
    }

    #[test]
    fn builds_preorder_topology_and_weighted_statistics() {
        let mut finder = RootSplit;
        let model = build_tree(
            &[0.0, 1.0, 2.0, 3.0],
            &[0, 0, 1, 1],
            &[2.0, 1.0, 1.0, 3.0],
            BuildParams {
                n_features: 1,
                n_classes: 2,
                max_depth: 1,
                min_samples_split: 2,
                min_samples_leaf: 1,
                min_impurity_decrease: 0.0,
                max_leaf_nodes: None,
            },
            &mut finder,
        )
        .unwrap();
        assert_eq!(model.children_left, vec![1, -1, -1]);
        assert_eq!(model.children_right, vec![2, -1, -1]);
        assert_eq!(model.n_node_samples, vec![4, 2, 2]);
        assert_eq!(model.weighted_n_node_samples, vec![7.0, 3.0, 4.0]);
    }
}
