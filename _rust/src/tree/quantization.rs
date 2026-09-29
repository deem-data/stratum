// Forest-wide row-major quantization used only while histogram trees train.
pub(crate) struct QuantizedData {
    bins: Vec<u8>,
    cuts: Vec<Vec<f64>>,
    actual_bins: Vec<usize>,
    pub(crate) n_features: usize,
}

impl QuantizedData {
    pub(crate) fn from_row_major(
        x: &[f32],
        n_rows: usize,
        n_features: usize,
        n_bins: usize,
    ) -> Result<Self, String> {
        if !(2..=255).contains(&n_bins) {
            return Err("n_bins must be in 2..=255".into());
        }
        if x.len() != n_rows * n_features {
            return Err("quantization input shape is inconsistent".into());
        }

        // Reuse one feature-sized sort buffer so preprocessing memory does not
        // scale with the number of columns.
        let mut sorted = Vec::with_capacity(n_rows);
        let mut cuts = Vec::with_capacity(n_features);
        let mut actual_bins = Vec::with_capacity(n_features);
        for feature in 0..n_features {
            sorted.clear();
            for row in 0..n_rows {
                let value = x[row * n_features + feature];
                if value.is_infinite() {
                    return Err("histogram quantization received an infinite feature value".into());
                }
                if !value.is_nan() {
                    sorted.push(value);
                }
            }
            sorted.sort_unstable_by(f32::total_cmp);
            let mut feature_cuts = Vec::new();
            if !sorted.is_empty() {
                let distinct = 1 + sorted.windows(2).filter(|pair| pair[0] != pair[1]).count();
                if distinct <= n_bins {
                    for pair in sorted.windows(2) {
                        if pair[0] != pair[1] {
                            feature_cuts.push(midpoint(pair[0], pair[1]));
                        }
                    }
                } else {
                    for j in 1..n_bins {
                        let mut rank =
                            ((j as u128 * sorted.len() as u128).div_ceil(n_bins as u128)) as usize;
                        rank = rank.min(sorted.len() - 1);
                        while rank < sorted.len() && sorted[rank - 1] == sorted[rank] {
                            rank += 1;
                        }
                        if rank == sorted.len() {
                            continue;
                        }
                        let cut = midpoint(sorted[rank - 1], sorted[rank]);
                        if feature_cuts.last().copied() != Some(cut) {
                            feature_cuts.push(cut);
                        }
                    }
                }
            }
            actual_bins.push(if sorted.is_empty() {
                0
            } else {
                feature_cuts.len() + 1
            });
            cuts.push(feature_cuts);
        }

        let mut bins = vec![0u8; x.len()];
        for (index, &value) in x.iter().enumerate() {
            if !value.is_nan() {
                let feature = index % n_features;
                bins[index] = (1 + cuts[feature].partition_point(|&cut| cut < value as f64)) as u8;
            }
        }
        Ok(Self {
            bins,
            cuts,
            actual_bins,
            n_features,
        })
    }

    #[inline]
    pub(crate) fn bin(&self, row: usize, feature: usize) -> usize {
        self.bins[row * self.n_features + feature] as usize
    }

    pub(crate) fn cuts(&self, feature: usize) -> &[f64] {
        &self.cuts[feature]
    }

    pub(crate) fn actual_bins(&self, feature: usize) -> usize {
        self.actual_bins[feature]
    }

    pub(crate) fn matrix_bytes(&self) -> usize {
        self.bins.len()
    }

    pub(crate) fn actual_bin_summary(&self) -> (usize, f64, usize) {
        if self.actual_bins.is_empty() {
            return (0, 0.0, 0);
        }
        let mut values = self.actual_bins.clone();
        values.sort_unstable();
        let middle = values.len() / 2;
        let median = if values.len() % 2 == 0 {
            (values[middle - 1] + values[middle]) as f64 / 2.0
        } else {
            values[middle] as f64
        };
        (values[0], median, values[values.len() - 1])
    }
}

// Use the exact split finder's overflow/equality fallback for raw thresholds.
fn midpoint(lower: f32, upper: f32) -> f64 {
    let lower = lower as f64;
    let upper = upper as f64;
    let midpoint = lower / 2.0 + upper / 2.0;
    if midpoint == upper || midpoint.is_infinite() {
        lower
    } else {
        midpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_low_cardinality_cuts_and_missing_bin() {
        let x = [0.0, f32::NAN, 1.0, 1.0, 1.0, 5.0];
        let quantized = QuantizedData::from_row_major(&x, 3, 2, 128).unwrap();
        assert_eq!(quantized.cuts(0), &[0.5]);
        assert_eq!(quantized.cuts(1), &[3.0]);
        assert_eq!(quantized.actual_bins, vec![2, 2]);
        assert_eq!(quantized.bins, vec![1, 0, 2, 1, 2, 2]);
    }

    #[test]
    fn quantiles_are_deterministic_and_collapse_duplicate_ranks() {
        let x = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 100.0, 100.0];
        for requested in [2, 64, 128, 255] {
            let first = QuantizedData::from_row_major(&x, x.len(), 1, requested).unwrap();
            let second = QuantizedData::from_row_major(&x, x.len(), 1, requested).unwrap();
            assert_eq!(first.cuts, second.cuts);
            assert_eq!(first.bins, second.bins);
            assert!(first.actual_bins[0] <= requested);
        }
    }

    #[test]
    fn handles_constant_and_all_missing_features_and_rejects_infinity() {
        let x = [1.0, f32::NAN, 1.0, f32::NAN];
        let quantized = QuantizedData::from_row_major(&x, 2, 2, 128).unwrap();
        assert_eq!(quantized.actual_bins, vec![1, 0]);
        assert_eq!(quantized.bins, vec![1, 0, 1, 0]);
        assert!(QuantizedData::from_row_major(&[f32::INFINITY], 1, 1, 128).is_err());
    }
}
