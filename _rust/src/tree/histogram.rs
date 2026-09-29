use super::builder::{NodeStats, Split, SplitFinder, SplitSearch};
use super::feature_sampling::FeatureSampler;
use super::quantization::QuantizedData;
use crate::timing::{debug_enabled, TimingCounter};

// Reused by every tree assigned to one long-lived forest worker.
pub(crate) struct HistogramScratch {
    weighted: Vec<u64>,
    rows: Vec<u64>,
    left: Vec<u64>,
    right: Vec<u64>,
    missing: Vec<u64>,
    touched_bins: Vec<usize>,
    timings: HistogramTimings,
}

struct HistogramTimings {
    validate_weights: TimingCounter,
    clear: TimingCounter,
    accumulate: TimingCounter,
    prepare: TimingCounter,
    scan: TimingCounter,
}

impl HistogramTimings {
    fn new() -> Self {
        let enabled = debug_enabled();
        Self {
            validate_weights: TimingCounter::new(enabled),
            clear: TimingCounter::new(enabled),
            accumulate: TimingCounter::new(enabled),
            prepare: TimingCounter::new(enabled),
            scan: TimingCounter::new(enabled),
        }
    }

    fn print(&self) {
        self.validate_weights.print("rf_hist validate weights");
        self.clear.print("rf_hist clear touched bins");
        self.accumulate.print("rf_hist accumulate rows");
        self.prepare.print("rf_hist prepare counts");
        self.scan.print("rf_hist scan occupied bins");
    }
}

impl Default for HistogramScratch {
    fn default() -> Self {
        Self {
            weighted: Vec::new(),
            rows: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            missing: Vec::new(),
            touched_bins: Vec::new(),
            timings: HistogramTimings::new(),
        }
    }
}

impl Drop for HistogramScratch {
    fn drop(&mut self) {
        self.timings.print();
    }
}

pub(crate) struct HistogramSplitFinder<'a> {
    quantized: &'a QuantizedData,
    feature_sampler: FeatureSampler,
    min_samples_leaf: usize,
    min_weight_leaf: f64,
    scratch: &'a mut HistogramScratch,
    weights_validated: bool,
}

impl<'a> HistogramSplitFinder<'a> {
    pub(crate) fn new(
        quantized: &'a QuantizedData,
        seed: u32,
        max_features: usize,
        min_samples_leaf: usize,
        min_weight_leaf: f64,
        scratch: &'a mut HistogramScratch,
    ) -> Self {
        Self {
            quantized,
            feature_sampler: FeatureSampler::new(quantized.n_features, seed, max_features),
            min_samples_leaf,
            min_weight_leaf,
            scratch,
            weights_validated: false,
        }
    }
}

impl SplitFinder for HistogramSplitFinder<'_> {
    fn best_split(
        &mut self,
        _x: &[f32],
        n_features: usize,
        rows: &[usize],
        y: &[usize],
        weights: &[f64],
        n_classes: usize,
        parent: &NodeStats,
        known_constants: &[usize],
    ) -> Result<SplitSearch, String> {
        debug_assert_eq!(n_features, self.quantized.n_features);
        if !self.weights_validated {
            let timing = self.scratch.timings.validate_weights.start();
            let validation = weights
                .iter()
                .try_for_each(|&weight| integral_weight(weight).map(|_| ()));
            self.scratch.timings.validate_weights.record(timing);
            validation?;
            self.weights_validated = true;
        }
        let mut feature_search = self.feature_sampler.begin(known_constants);
        let mut best = None;
        let mut best_proxy = f64::NEG_INFINITY;

        // Build and scan one sampled-feature histogram at a time.
        while let Some(candidate) = self.feature_sampler.next(&mut feature_search) {
            let feature = candidate.feature;
            let actual_bins = self.quantized.actual_bins(feature);
            let histogram_len = (actual_bins + 1) * n_classes;

            // Clear only bins populated by the previous feature. This keeps
            // deep, small nodes proportional to their occupied bin count.
            let timing = self.scratch.timings.clear.start();
            for &bin in &self.scratch.touched_bins {
                self.scratch.rows[bin] = 0;
                self.scratch.weighted[bin * n_classes..(bin + 1) * n_classes].fill(0);
            }
            self.scratch.touched_bins.clear();
            self.scratch.weighted.resize(histogram_len, 0);
            self.scratch.rows.resize(actual_bins + 1, 0);
            self.scratch.timings.clear.record(timing);

            // Weights were validated once for this tree, so the hot row loop
            // can use the native integral bootstrap multiplicity directly.
            let timing = self.scratch.timings.accumulate.start();
            for &row in rows {
                let weight = weights[row] as u64;
                let bin = self.quantized.bin(row, feature);
                if self.scratch.rows[bin] == 0 {
                    self.scratch.touched_bins.push(bin);
                }
                self.scratch.weighted[bin * n_classes + y[row]] += weight;
                self.scratch.rows[bin] += 1;
            }
            self.scratch.timings.accumulate.record(timing);

            let timing = self.scratch.timings.prepare.start();
            self.scratch.touched_bins.sort_unstable();
            let missing_rows = self.scratch.rows[0] as usize;
            let finite_rows = rows.len() - missing_rows;
            let finite_start = self
                .scratch
                .touched_bins
                .iter()
                .position(|&bin| bin != 0)
                .unwrap_or(self.scratch.touched_bins.len());
            let occupied_finite_bins = self.scratch.touched_bins.len() - finite_start;
            self.scratch.timings.prepare.record(timing);
            if finite_rows == 0 || (missing_rows == 0 && occupied_finite_bins <= 1) {
                self.feature_sampler
                    .mark_constant(&mut feature_search, candidate);
                continue;
            }
            self.feature_sampler
                .mark_nonconstant(&mut feature_search, candidate);

            self.scratch.left.resize(n_classes, 0);
            self.scratch.left.fill(0);
            self.scratch.right.resize(n_classes, 0);
            self.scratch.right.fill(0);
            self.scratch.missing.resize(n_classes, 0);

            // Parent statistics are the authoritative class totals. Reusing
            // them avoids summing every histogram bin before each scan.
            let timing = self.scratch.timings.prepare.start();
            for class in 0..n_classes {
                self.scratch.missing[class] = self.scratch.weighted[class];
                debug_assert_eq!(parent.class_weights[class].fract(), 0.0);
                self.scratch.right[class] = parent.class_weights[class] as u64;
            }
            let missing_weight: u64 = self.scratch.missing.iter().sum();
            debug_assert_eq!(parent.weight.fract(), 0.0);
            let parent_weight = parent.weight as u64;
            let mut left_weight = 0u64;
            let mut left_rows = 0usize;
            self.scratch.timings.prepare.record(timing);

            let timing = self.scratch.timings.scan.start();
            let finite_bins = &self.scratch.touched_bins[finite_start..];

            // Preserve the leading empty-bin plateau: with missing values it
            // can represent the valid missing-left versus finite partition.
            if finite_bins[0] > 1 {
                let plateau_end = finite_bins[0] - 1;
                let middle_cut = (1 + plateau_end) / 2;
                consider_partition(
                    feature,
                    self.quantized.cuts(feature)[middle_cut - 1],
                    missing_rows,
                    finite_rows,
                    left_rows,
                    left_weight,
                    missing_weight,
                    parent_weight,
                    parent,
                    self.min_samples_leaf,
                    self.min_weight_leaf,
                    &mut self.scratch.left,
                    &mut self.scratch.right,
                    &self.scratch.missing,
                    &mut best_proxy,
                    &mut best,
                );
            }

            // Visit only occupied finite bins. The next occupied id defines
            // the same centered threshold as scanning each empty plateau.
            for (index, &bin) in finite_bins.iter().enumerate() {
                add_bin(
                    &self.scratch.weighted,
                    n_classes,
                    bin,
                    &mut self.scratch.left,
                    &mut self.scratch.right,
                    &mut left_weight,
                );
                left_rows += self.scratch.rows[bin] as usize;
                if bin < actual_bins {
                    let next_bin = finite_bins.get(index + 1).copied().unwrap_or(actual_bins);
                    let plateau_end = next_bin - 1;
                    let middle_cut = (bin + plateau_end) / 2;
                    consider_partition(
                        feature,
                        self.quantized.cuts(feature)[middle_cut - 1],
                        missing_rows,
                        finite_rows,
                        left_rows,
                        left_weight,
                        missing_weight,
                        parent_weight,
                        parent,
                        self.min_samples_leaf,
                        self.min_weight_leaf,
                        &mut self.scratch.left,
                        &mut self.scratch.right,
                        &self.scratch.missing,
                        &mut best_proxy,
                        &mut best,
                    );
                }
            }

            // The +infinity candidate remains last, preserving strict tie
            // behavior against an equivalent trailing finite-bin plateau.
            if missing_rows >= self.min_samples_leaf
                && left_rows >= self.min_samples_leaf
                && missing_weight as f64 >= self.min_weight_leaf
                && left_weight as f64 >= self.min_weight_leaf
            {
                let left_impurity = gini_u64(&self.scratch.left, left_weight);
                let right_impurity = gini_u64(&self.scratch.missing, missing_weight);
                let proxy = -(left_weight as f64) * left_impurity
                    - (missing_weight as f64) * right_impurity;
                if proxy > best_proxy {
                    best_proxy = proxy;
                    best = Some(make_split(
                        feature,
                        f64::INFINITY,
                        false,
                        parent,
                        left_weight,
                        missing_weight,
                        left_impurity,
                        right_impurity,
                    ));
                }
            }
            self.scratch.timings.scan.record(timing);
        }
        let constants = self.feature_sampler.finish(feature_search, known_constants);
        Ok(SplitSearch {
            split: best,
            constants,
        })
    }
}

// Evaluate one unique node partition. Missing-right remains first and strict
// proxy comparison preserves the existing deterministic tie behavior.
#[allow(clippy::too_many_arguments)]
fn consider_partition(
    feature: usize,
    threshold: f64,
    missing_rows: usize,
    finite_rows: usize,
    left_rows: usize,
    left_weight: u64,
    missing_weight: u64,
    parent_weight: u64,
    parent: &NodeStats,
    min_samples_leaf: usize,
    min_weight_leaf: f64,
    left: &mut [u64],
    right: &mut [u64],
    missing: &[u64],
    best_proxy: &mut f64,
    best: &mut Option<Split>,
) {
    let finite_right_rows = finite_rows - left_rows;
    let right_weight = parent_weight - left_weight;
    for missing_left in [false, true] {
        if missing_rows == 0 && missing_left {
            continue;
        }
        let candidate_left_rows = left_rows + usize::from(missing_left) * missing_rows;
        let candidate_right_rows = finite_right_rows + usize::from(!missing_left) * missing_rows;
        let candidate_left_weight = left_weight + if missing_left { missing_weight } else { 0 };
        let candidate_right_weight = right_weight - if missing_left { missing_weight } else { 0 };
        if candidate_left_rows < min_samples_leaf
            || candidate_right_rows < min_samples_leaf
            || (candidate_left_weight as f64) < min_weight_leaf
            || (candidate_right_weight as f64) < min_weight_leaf
        {
            continue;
        }
        if missing_left {
            for class in 0..left.len() {
                left[class] += missing[class];
                right[class] -= missing[class];
            }
        }
        let left_impurity = gini_u64(left, candidate_left_weight);
        let right_impurity = gini_u64(right, candidate_right_weight);
        let proxy = -(candidate_left_weight as f64) * left_impurity
            - (candidate_right_weight as f64) * right_impurity;
        if proxy > *best_proxy {
            *best_proxy = proxy;
            *best = Some(make_split(
                feature,
                threshold,
                if missing_rows == 0 {
                    left_rows > finite_right_rows
                } else {
                    missing_left
                },
                parent,
                candidate_left_weight,
                candidate_right_weight,
                left_impurity,
                right_impurity,
            ));
        }
        if missing_left {
            for class in 0..left.len() {
                left[class] -= missing[class];
                right[class] += missing[class];
            }
        }
    }
}

fn integral_weight(weight: f64) -> Result<u64, String> {
    if weight < 0.0 || weight > u64::MAX as f64 || weight.fract() != 0.0 {
        return Err("histogram split finder requires non-negative integral weights".into());
    }
    Ok(weight as u64)
}

fn add_bin(
    histogram: &[u64],
    n_classes: usize,
    bin: usize,
    left: &mut [u64],
    right: &mut [u64],
    left_weight: &mut u64,
) {
    for class in 0..n_classes {
        let count = histogram[bin * n_classes + class];
        left[class] += count;
        right[class] -= count;
        *left_weight += count;
    }
}

fn gini_u64(counts: &[u64], total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total = total as f64;
    1.0 - counts
        .iter()
        .map(|&value| (value as f64) * (value as f64))
        .sum::<f64>()
        / (total * total)
}

#[allow(clippy::too_many_arguments)]
fn make_split(
    feature: usize,
    threshold: f64,
    missing_go_to_left: bool,
    parent: &NodeStats,
    left_weight: u64,
    right_weight: u64,
    left_impurity: f64,
    right_impurity: f64,
) -> Split {
    Split {
        feature,
        threshold,
        missing_go_to_left,
        improvement: parent.impurity
            - left_weight as f64 / parent.weight * left_impurity
            - right_weight as f64 / parent.weight * right_impurity,
        left_impurity,
        right_impurity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::builder::{build_tree, BuildParams};
    use crate::tree::model::TreeModel;

    fn fit_one_feature(x: &[f32], y: &[usize], weights: &[f64], bins: usize) -> TreeModel {
        let quantized = QuantizedData::from_row_major(x, y.len(), 1, bins).unwrap();
        let mut scratch = HistogramScratch::default();
        let mut finder = HistogramSplitFinder::new(&quantized, 1, 1, 1, 0.0, &mut scratch);
        build_tree(
            x,
            y,
            weights,
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
        .unwrap()
    }

    #[test]
    fn centers_threshold_across_empty_node_bins() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let quantized = QuantizedData::from_row_major(&x, x.len(), 1, 6).unwrap();
        let rows = [0usize, 5usize];
        let y = [0usize, 0, 0, 0, 0, 1];
        let weights = [1.0; 6];
        let parent = NodeStats {
            class_weights: vec![1.0, 1.0],
            weight: 2.0,
            impurity: 0.5,
        };
        let mut scratch = HistogramScratch::default();
        let mut finder = HistogramSplitFinder::new(&quantized, 1, 1, 1, 0.0, &mut scratch);
        let split = finder
            .best_split(&x, 1, &rows, &y, &weights, 2, &parent, &[])
            .unwrap()
            .split
            .unwrap();
        assert_eq!(split.threshold, 2.5);
    }

    #[test]
    fn preserves_leading_plateau_missing_left_candidate() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN];
        let y = [0usize, 0, 0, 0, 0, 0, 1];
        let weights = [1.0; 7];
        let quantized = QuantizedData::from_row_major(&x, x.len(), 1, 6).unwrap();
        let parent = NodeStats {
            class_weights: vec![1.0, 1.0],
            weight: 2.0,
            impurity: 0.5,
        };
        let mut scratch = HistogramScratch::default();
        let mut finder = HistogramSplitFinder::new(&quantized, 1, 1, 1, 0.0, &mut scratch);
        let split = finder
            .best_split(&x, 1, &[5, 6], &y, &weights, 2, &parent, &[])
            .unwrap()
            .split
            .unwrap();
        assert_eq!(split.threshold, 2.5);
        assert!(split.missing_go_to_left);
    }

    #[test]
    fn preserves_trailing_plateau_missing_right_candidate() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN];
        let y = [0usize, 0, 0, 0, 0, 0, 1];
        let weights = [1.0; 7];
        let quantized = QuantizedData::from_row_major(&x, x.len(), 1, 6).unwrap();
        let parent = NodeStats {
            class_weights: vec![1.0, 1.0],
            weight: 2.0,
            impurity: 0.5,
        };
        let mut scratch = HistogramScratch::default();
        let mut finder = HistogramSplitFinder::new(&quantized, 1, 1, 1, 0.0, &mut scratch);
        let split = finder
            .best_split(&x, 1, &[0, 6], &y, &weights, 2, &parent, &[])
            .unwrap()
            .split
            .unwrap();
        assert_eq!(split.threshold, 2.5);
        assert!(!split.missing_go_to_left);
    }

    #[test]
    fn supports_bootstrap_counts_and_missingness_only_split() {
        let model = fit_one_feature(
            &[1.0, 1.0, f32::NAN, f32::NAN],
            &[0, 0, 1, 1],
            &[3.0, 1.0, 2.0, 4.0],
            128,
        );
        assert_eq!(model.threshold[0], f64::INFINITY);
        assert!(!model.missing_go_to_left[0]);
        assert_eq!(model.n_node_samples, vec![4, 2, 2]);
        assert_eq!(model.weighted_n_node_samples, vec![10.0, 4.0, 6.0]);
    }

    #[test]
    fn rejects_non_integral_weights_during_one_time_validation() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0usize, 0, 1, 1];
        let weights = [1.0, 1.5, 1.0, 1.0];
        let quantized = QuantizedData::from_row_major(&x, x.len(), 1, 4).unwrap();
        let parent = NodeStats {
            class_weights: vec![2.5, 2.0],
            weight: 4.5,
            impurity: 0.5,
        };
        let mut scratch = HistogramScratch::default();
        let mut finder = HistogramSplitFinder::new(&quantized, 1, 1, 1, 0.0, &mut scratch);
        let error = match finder.best_split(&x, 1, &[0, 1, 2, 3], &y, &weights, 2, &parent, &[]) {
            Err(error) => error,
            Ok(_) => panic!("non-integral histogram weights must be rejected"),
        };
        assert!(error.contains("integral weights"));
    }

    #[test]
    fn raw_threshold_partition_matches_selected_bins() {
        let x = [f32::MIN, -1.0, 0.0, f32::MAX, f32::NAN];
        let model = fit_one_feature(&x, &[0, 0, 1, 1, 1], &[1.0; 5], 2);
        let threshold = model.threshold[0];
        assert!(x[..4].iter().all(|&value| {
            let leaf = model.leaf_for_row(&[value]);
            (value as f64 <= threshold) == (leaf == model.children_left[0] as usize)
        }));
        assert!(threshold.is_finite());
    }
}
