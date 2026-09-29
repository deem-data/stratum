use super::builder::{NodeStats, Split, SplitFinder, SplitSearch};
use super::feature_sampling::FeatureSampler;

const FEATURE_THRESHOLD: f32 = 1.0e-7;

// Exact sklearn-style split search with deterministic feature sampling.
pub(crate) struct ExactSplitFinder {
    feature_sampler: FeatureSampler,
    min_samples_leaf: usize,
    min_weight_leaf: f64,
    sorted: Vec<(f32, usize)>,
    left_counts: Vec<f64>,
    right_counts: Vec<f64>,
    missing_counts: Vec<f64>,
}

impl ExactSplitFinder {
    // Store feature order, RNG state, and a reusable sort buffer.
    pub(crate) fn new(
        n_features: usize,
        seed: u32,
        max_features: usize,
        min_samples_leaf: usize,
        min_weight_leaf: f64,
    ) -> Self {
        Self {
            feature_sampler: FeatureSampler::new(n_features, seed, max_features),
            min_samples_leaf,
            min_weight_leaf,
            sorted: Vec::new(),
            left_counts: Vec::new(),
            right_counts: Vec::new(),
            missing_counts: Vec::new(),
        }
    }
}

// Search one node by sampling features, sorting feature values, and tracking the best gain.
impl SplitFinder for ExactSplitFinder {
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
    ) -> Result<SplitSearch, String> {
        let mut feature_search = self.feature_sampler.begin(known_constants);
        let mut best: Option<Split> = None;
        let mut best_proxy = f64::NEG_INFINITY;

        // Draw candidate features until the sklearn-style stopping rule is met.
        while let Some(candidate) = self.feature_sampler.next(&mut feature_search) {
            let feature = candidate.feature;
            self.sorted.clear();
            self.sorted
                .extend(rows.iter().map(|&row| (x[row * n_features + feature], row)));
            self.sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            let finite_end = self.sorted.partition_point(|(value, _)| !value.is_nan());
            if self.sorted[..finite_end]
                .iter()
                .any(|(value, _)| value.is_infinite())
            {
                return Err("exact split finder received an infinite feature value".into());
            }
            // Constant features are remembered and skipped in future search rounds.
            if finite_end == 0
                || (finite_end == self.sorted.len()
                    && self.sorted[finite_end - 1].0 <= self.sorted[0].0 + FEATURE_THRESHOLD)
            {
                self.feature_sampler
                    .mark_constant(&mut feature_search, candidate);
                continue;
            }

            self.feature_sampler
                .mark_nonconstant(&mut feature_search, candidate);
            self.left_counts.resize(n_classes, 0.0);
            self.left_counts.fill(0.0);
            self.right_counts.clone_from(&parent.class_weights);
            self.missing_counts.resize(n_classes, 0.0);
            self.missing_counts.fill(0.0);
            let n_missing = self.sorted.len() - finite_end;
            let mut missing_weight = 0.0;
            for &(_, row) in &self.sorted[finite_end..] {
                self.missing_counts[y[row]] += weights[row];
                missing_weight += weights[row];
            }
            let mut left_weight = 0.0;
            let mut right_weight = parent.weight;
            let mut p = 0usize;

            // Sweep candidate thresholds across groups of equal feature values.
            while p < finite_end {
                let p_prev = p;
                p += 1;
                while p < finite_end && self.sorted[p].0 <= self.sorted[p - 1].0 + FEATURE_THRESHOLD
                {
                    p += 1;
                }
                for &(_, row) in &self.sorted[p_prev..p] {
                    let weight = weights[row];
                    self.left_counts[y[row]] += weight;
                    self.right_counts[y[row]] -= weight;
                    left_weight += weight;
                    right_weight -= weight;
                }
                if p == finite_end {
                    continue;
                }

                // Missing-right is evaluated first so strict ties route right.
                for missing_left in [false, true] {
                    if n_missing == 0 && missing_left {
                        continue;
                    }
                    let n_left = p + usize::from(missing_left) * n_missing;
                    let n_right = self.sorted.len() - n_left;
                    let candidate_left_weight =
                        left_weight + if missing_left { missing_weight } else { 0.0 };
                    let candidate_right_weight =
                        right_weight - if missing_left { missing_weight } else { 0.0 };
                    if n_left < self.min_samples_leaf
                        || n_right < self.min_samples_leaf
                        || candidate_left_weight < self.min_weight_leaf
                        || candidate_right_weight < self.min_weight_leaf
                    {
                        continue;
                    }
                    if missing_left {
                        for class in 0..n_classes {
                            self.left_counts[class] += self.missing_counts[class];
                            self.right_counts[class] -= self.missing_counts[class];
                        }
                    }
                    let left_impurity = gini(&self.left_counts, candidate_left_weight);
                    let right_impurity = gini(&self.right_counts, candidate_right_weight);
                    let proxy = -candidate_left_weight * left_impurity
                        - candidate_right_weight * right_impurity;
                    if proxy > best_proxy {
                        best_proxy = proxy;
                        let lower = self.sorted[p - 1].0 as f64;
                        let upper = self.sorted[p].0 as f64;
                        let mut threshold = lower / 2.0 + upper / 2.0;
                        if threshold == upper || threshold.is_infinite() {
                            threshold = lower;
                        }
                        let improvement = parent.impurity
                            - candidate_left_weight / parent.weight * left_impurity
                            - candidate_right_weight / parent.weight * right_impurity;
                        best = Some(Split {
                            feature,
                            threshold,
                            missing_go_to_left: if n_missing == 0 {
                                p > self.sorted.len() - p
                            } else {
                                missing_left
                            },
                            improvement,
                            left_impurity,
                            right_impurity,
                        });
                    }
                    if missing_left {
                        for class in 0..n_classes {
                            self.left_counts[class] -= self.missing_counts[class];
                            self.right_counts[class] += self.missing_counts[class];
                        }
                    }
                }
            }

            // Also consider the finite-versus-missing split represented by +inf.
            if n_missing > 0
                && finite_end >= self.min_samples_leaf
                && n_missing >= self.min_samples_leaf
                && left_weight >= self.min_weight_leaf
                && missing_weight >= self.min_weight_leaf
            {
                let left_impurity = gini(&self.left_counts, left_weight);
                let right_impurity = gini(&self.missing_counts, missing_weight);
                let proxy = -left_weight * left_impurity - missing_weight * right_impurity;
                if proxy > best_proxy {
                    best_proxy = proxy;
                    let improvement = parent.impurity
                        - left_weight / parent.weight * left_impurity
                        - missing_weight / parent.weight * right_impurity;
                    best = Some(Split {
                        feature,
                        threshold: f64::INFINITY,
                        missing_go_to_left: false,
                        improvement,
                        left_impurity,
                        right_impurity,
                    });
                }
            }
        }
        let constants = self.feature_sampler.finish(feature_search, known_constants);
        Ok(SplitSearch {
            split: best,
            constants,
        })
    }
}

// Standard Gini impurity helper used by the exact split finder.
pub(crate) fn gini(counts: &[f64], total: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    1.0 - counts.iter().map(|value| value * value).sum::<f64>() / (total * total)
}
