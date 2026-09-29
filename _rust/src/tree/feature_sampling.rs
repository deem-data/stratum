use super::rng::SklearnRng;

// Shared sklearn-style per-node feature visitation.
//
// Split finders report whether each drawn feature is constant. Keeping these
// swaps here ensures exact and histogram training consume RNG identically.
pub(crate) struct FeatureSampler {
    features: Vec<usize>,
    rng: SklearnRng,
    max_features: usize,
}

pub(crate) struct FeatureSearch {
    n_known: usize,
    f_i: usize,
    n_visited: usize,
    n_found: usize,
    n_drawn_constants: usize,
    n_total_constants: usize,
}

pub(crate) struct FeatureCandidate {
    pub(crate) feature: usize,
    index: usize,
}

impl FeatureSampler {
    pub(crate) fn new(n_features: usize, seed: u32, max_features: usize) -> Self {
        Self {
            features: (0..n_features).collect(),
            rng: SklearnRng::new(seed),
            max_features,
        }
    }

    pub(crate) fn begin(&mut self, known_constants: &[usize]) -> FeatureSearch {
        let n_known = known_constants.len();
        self.features[..n_known].copy_from_slice(known_constants);
        FeatureSearch {
            n_known,
            f_i: self.features.len(),
            n_visited: 0,
            n_found: 0,
            n_drawn_constants: 0,
            n_total_constants: n_known,
        }
    }

    pub(crate) fn next(&mut self, search: &mut FeatureSearch) -> Option<FeatureCandidate> {
        while search.f_i > search.n_total_constants
            && (search.n_visited < self.max_features
                || search.n_visited <= search.n_found + search.n_drawn_constants)
        {
            search.n_visited += 1;
            let mut index = self
                .rng
                .bounded(search.n_drawn_constants, search.f_i - search.n_found);
            if index < search.n_known {
                self.features.swap(search.n_drawn_constants, index);
                search.n_drawn_constants += 1;
                continue;
            }
            index += search.n_found;
            return Some(FeatureCandidate {
                feature: self.features[index],
                index,
            });
        }
        None
    }

    pub(crate) fn mark_constant(
        &mut self,
        search: &mut FeatureSearch,
        candidate: FeatureCandidate,
    ) {
        self.features
            .swap(candidate.index, search.n_total_constants);
        search.n_found += 1;
        search.n_total_constants += 1;
    }

    pub(crate) fn mark_nonconstant(
        &mut self,
        search: &mut FeatureSearch,
        candidate: FeatureCandidate,
    ) {
        search.f_i -= 1;
        self.features.swap(search.f_i, candidate.index);
    }

    pub(crate) fn finish(
        &mut self,
        search: FeatureSearch,
        known_constants: &[usize],
    ) -> Vec<usize> {
        let mut constants = known_constants.to_vec();
        constants.extend_from_slice(&self.features[search.n_known..search.n_total_constants]);
        self.features[..search.n_known].copy_from_slice(known_constants);
        constants
    }
}
