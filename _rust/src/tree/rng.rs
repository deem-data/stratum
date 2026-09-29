// Small deterministic xorshift RNG that mirrors sklearn's feature sampling.
#[derive(Clone, Copy)]
pub(crate) struct SklearnRng(u32);

impl SklearnRng {
    // Seed the RNG with the raw sklearn-provided tree seed.
    pub(crate) fn new(seed: u32) -> Self {
        Self(seed)
    }

    // Return a bounded index using the same xorshift-style update every call.
    pub(crate) fn bounded(&mut self, low: usize, high: usize) -> usize {
        debug_assert!(low < high);
        let mut value = if self.0 == 0 { 1 } else { self.0 };
        value ^= value.wrapping_shl(13);
        value ^= value.wrapping_shr(17);
        value ^= value.wrapping_shl(5);
        self.0 = value;
        low + ((value % 0x8000_0000) as usize % (high - low))
    }
}

// Compact MT19937 implementation for NumPy RandomState-compatible forest
// seeds and bootstrap draws. Keeping it native avoids materializing bootstrap
// index matrices at the Python boundary.
pub(crate) struct NumpyRng {
    state: [u32; 624],
    index: usize,
}

impl NumpyRng {
    pub(crate) fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000)
                    | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    // NumPy's legacy rk_interval draws uniformly from [0, max] by masking and
    // rejecting, rather than using modulo reduction.
    pub(crate) fn interval(&mut self, max: u32) -> u32 {
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.next_u32() & mask;
            if value <= max {
                return value;
            }
        }
    }
}

// Regression test for the deterministic feature-selection sequence.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_sklearn_xorshift_sequence() {
        let mut rng = SklearnRng::new(209_652_396);
        assert_eq!(rng.bounded(0, 5), 2);
        assert_eq!(rng.bounded(0, 4), 2);
    }


    #[test]
    fn matches_numpy_bootstrap_sequence() {
        let mut rng = NumpyRng::new(209_652_396);
        let values: Vec<u32> = (0..10).map(|_| rng.interval(9)).collect();
        assert_eq!(values, vec![3, 2, 0, 0, 2, 6, 9, 7, 3, 9]);
    }
}
