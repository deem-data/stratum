use ndarray::{Array2, ArrayBase, ArrayView2, Axis, DataMut, Ix2};
use rayon::prelude::*;
use crate::threads::get_thread_pool;

// ---- Vectorizable (reassociated) row reductions ----
//
// A strict left-to-right `iter().map(..).sum()` cannot be vectorized: SIMD would
// add the elements in a different order, and floating-point addition is not
// associative. We split each row sum into `NORM_LANES` independent partial sums
// so LLVM can lower the inner loop to NEON `fadd.4s`; only the final fold over
// the lanes reassociates, so the result differs from the scalar sum by at most a
// few ULP (well inside the benchmark's rtol=1e-3/atol=1e-4 check). Rows that are
// not contiguous fall back to the strict scalar path.
const NORM_LANES: usize = 8;

#[inline]
fn l2_norm_sq(s: &[f32]) -> f32 {
    let mut acc = [0.0f32; NORM_LANES];
    let chunks = s.chunks_exact(NORM_LANES);
    let rem = chunks.remainder();
    for c in chunks {
        for l in 0..NORM_LANES { acc[l] += c[l] * c[l]; }
    }
    let mut sum = 0.0f32;
    for &x in rem { sum += x * x; }
    for l in 0..NORM_LANES { sum += acc[l]; }
    sum
}

#[inline]
fn l1_norm(s: &[f32]) -> f32 {
    let mut acc = [0.0f32; NORM_LANES];
    let chunks = s.chunks_exact(NORM_LANES);
    let rem = chunks.remainder();
    for c in chunks {
        for l in 0..NORM_LANES { acc[l] += c[l].abs(); }
    }
    let mut sum = 0.0f32;
    for &x in rem { sum += x.abs(); }
    for l in 0..NORM_LANES { sum += acc[l]; }
    sum
}

// ---- Stateless row-wise normalizations ----
//
// Each norm comes in two flavours:
//   * `normalize_*`         — copy-returning: read `src` and write the
//                             normalized result into a freshly allocated output
//                             in a single pass (uninitialized output, written
//                             exactly once, so net RAM traffic is 1 read + 1
//                             write per element).
//   * `normalize_*_inplace` — generic over `DataMut`, mutating the buffer in
//                             place; used by the zero-allocation numpy bindings.

pub fn normalize_l2_inplace<S: DataMut<Elem = f32> + Sync + Send>(data: &mut ArrayBase<S, Ix2>) {
    let pool = get_thread_pool();                            // shared stratum pool, or None
    let mut work = || {                                      // deferred so we can run it inside the pool
        data.axis_iter_mut(Axis(0))                          // iterate mutably over rows (one 1-D row view each)
            .into_par_iter()                                 // turn that row-iterator into a Rayon parallel iterator
            .for_each(|mut row| {                            // each Rayon task owns one row
                // Σ xⱼ²  for this row (vectorized reduction; scalar fallback if non-contiguous)
                let norm_sq: f32 = row.as_slice().map(l2_norm_sq)
                    .unwrap_or_else(|| row.iter().map(|x| x * x).sum());
                if norm_sq > 0.0 {                           // skip all-zero rows (no div-by-0)
                    let inv = 1.0 / norm_sq.sqrt();           // 1 / ‖row‖₂
                    row.mapv_inplace(|x| x * inv);            // scale row in place
                }
            });
    };
    match pool {
        Some(p) => p.install(work),                          // run on the stratum pool (honours SKRUB_RUST_THREADS)
        None => work(),                                      // no pool configured → Rayon's default global pool
    }
}

// Copy-returning L2 normalize: reads `src` and writes the normalized result
// straight into a freshly allocated output in a single pass. Avoids the
// redundant `to_owned()` copy the binding used to make (which read the input and
// wrote a raw duplicate, only to overwrite every element again). The output is
// allocated uninitialized — every element is written exactly once below — so
// there's no zero-fill pass either: net RAM traffic is 1 read + 1 write per
// element (2N) instead of the old 4N.
pub fn normalize_l2(src: ArrayView2<f32>) -> Array2<f32> {
    let n_rows = src.nrows();
    let n_cols = src.ncols();
    let mut out = Array2::<f32>::uninit((n_rows, n_cols));    // uninitialised → no zero-fill pass
    let pool = get_thread_pool();
    let mut work = || {
        out.axis_iter_mut(Axis(0))                           // output rows, mutable (one per task)
            .into_par_iter()                                 // parallelise across rows
            .zip(src.axis_iter(Axis(0)))                     // pair each output row with its input row
            .for_each(|(mut out_row, in_row)| {              // each task: one (out_row, in_row) pair
                // Σ xⱼ²  (1st scan of the row; stays L1-hot). Vectorized reduction.
                let norm_sq: f32 = in_row.as_slice().map(l2_norm_sq)
                    .unwrap_or_else(|| in_row.iter().map(|x| x * x).sum());
                if norm_sq > 0.0 {
                    let inv = 1.0 / norm_sq.sqrt();           // 1 / ‖row‖₂
                    for (o, &x) in out_row.iter_mut().zip(in_row.iter()) {  // 2nd scan: write scaled values
                        o.write(x * inv);                     // write into uninitialised slot (write, not assign)
                    }
                } else {
                    // All-zero (or underflowing) row: pass through unchanged,
                    // matching the in-place kernel's semantics.
                    for (o, &x) in out_row.iter_mut().zip(in_row.iter()) {
                        o.write(x);                           // still must initialise every output element
                    }
                }
            });
    };
    match pool {
        Some(p) => p.install(work),
        None => work(),
    }
    // SAFETY: every element of `out` was written exactly once above (each row is
    // fully traversed in one of the two branches).
    unsafe { out.assume_init() }                             // reinterpret MaybeUninit buffer as initialised
}

pub fn normalize_l1_inplace<S: DataMut<Elem = f32> + Sync + Send>(data: &mut ArrayBase<S, Ix2>) {
    let pool = get_thread_pool();
    let mut work = || {
        data.axis_iter_mut(Axis(0))                          // iterate mutably over rows
            .into_par_iter()                                 // one Rayon task per row
            .for_each(|mut row| {
                // ‖row‖₁ = Σ |xⱼ|  (vectorized reduction; scalar fallback if non-contiguous)
                let norm: f32 = row.as_slice().map(l1_norm)
                    .unwrap_or_else(|| row.iter().map(|x| x.abs()).sum());
                if norm > 0.0 {                              // skip all-zero rows
                    let inv = 1.0 / norm;                    // 1 / ‖row‖₁
                    row.mapv_inplace(|x| x * inv);           // scale row in place
                }
            });
    };
    match pool {
        Some(p) => p.install(work),
        None => work(),
    }
}

pub fn normalize_max_inplace<S: DataMut<Elem = f32> + Sync + Send>(data: &mut ArrayBase<S, Ix2>) {
    let pool = get_thread_pool();
    let mut work = || {
        data.axis_iter_mut(Axis(0))                          // iterate mutably over rows
            .into_par_iter()                                 // one Rayon task per row
            .for_each(|mut row| {
                let max_abs = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);  // max |xⱼ| over the row
                if max_abs > 0.0 {                           // skip all-zero rows
                    let inv = 1.0 / max_abs;                 // 1 / max|row|
                    row.mapv_inplace(|x| x * inv);           // scale row in place
                }
            });
    };
    match pool {
        Some(p) => p.install(work),
        None => work(),
    }
}

// Copy-returning L1 normalize (counterpart to in-place `normalize_l1_inplace`).
// Single pass, uninitialized output written exactly once. See `normalize_l2`.
pub fn normalize_l1(src: ArrayView2<f32>) -> Array2<f32> {
    let n_rows = src.nrows();
    let n_cols = src.ncols();
    let mut out = Array2::<f32>::uninit((n_rows, n_cols));
    let pool = get_thread_pool();
    let mut work = || {
        out.axis_iter_mut(Axis(0))                           // output rows (one per task)
            .into_par_iter()                                 // parallelise across rows
            .zip(src.axis_iter(Axis(0)))                     // pair each output row with its input row
            .for_each(|(mut out_row, in_row)| {
                // ‖row‖₁ = Σ |xⱼ|  (vectorized reduction; scalar fallback if non-contiguous)
                let norm: f32 = in_row.as_slice().map(l1_norm)
                    .unwrap_or_else(|| in_row.iter().map(|x| x.abs()).sum());
                if norm > 0.0 {
                    let inv = 1.0 / norm;                    // 1 / ‖row‖₁
                    for (o, &x) in out_row.iter_mut().zip(in_row.iter()) {
                        o.write(x * inv);                    // write scaled value into uninitialised slot
                    }
                } else {
                    for (o, &x) in out_row.iter_mut().zip(in_row.iter()) {
                        o.write(x);                          // zero row passes through; still initialise output
                    }
                }
            });
    };
    match pool {
        Some(p) => p.install(work),
        None => work(),
    }
    // SAFETY: every element of `out` is written exactly once above.
    unsafe { out.assume_init() }
}

// Copy-returning max normalize (counterpart to in-place `normalize_max_inplace`).
// Single pass, uninitialized output written exactly once. See `normalize_l2`.
pub fn normalize_max(src: ArrayView2<f32>) -> Array2<f32> {
    let n_rows = src.nrows();
    let n_cols = src.ncols();
    let mut out = Array2::<f32>::uninit((n_rows, n_cols));
    let pool = get_thread_pool();
    let mut work = || {
        out.axis_iter_mut(Axis(0))                           // output rows (one per task)
            .into_par_iter()                                 // parallelise across rows
            .zip(src.axis_iter(Axis(0)))                     // pair each output row with its input row
            .for_each(|(mut out_row, in_row)| {
                let max_abs = in_row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);  // max |xⱼ| over the row
                if max_abs > 0.0 {
                    let inv = 1.0 / max_abs;                 // 1 / max|row|
                    for (o, &x) in out_row.iter_mut().zip(in_row.iter()) {
                        o.write(x * inv);                    // write scaled value into uninitialised slot
                    }
                } else {
                    for (o, &x) in out_row.iter_mut().zip(in_row.iter()) {
                        o.write(x);                          // zero row passes through; still initialise output
                    }
                }
            });
    };
    match pool {
        Some(p) => p.install(work),
        None => work(),
    }
    // SAFETY: every element of `out` is written exactly once above.
    unsafe { out.assume_init() }
}

// ---- Unit tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPS: f32 = 1e-5;

    #[test]
    fn l2_unit_norm() {
        let mut data = array![[3.0f32, 4.0], [1.0, 0.0], [0.0, 0.0]];
        normalize_l2_inplace(&mut data);
        // Row 0: norm = 5, so [0.6, 0.8]
        assert!((data[[0, 0]] - 0.6).abs() < EPS);
        assert!((data[[0, 1]] - 0.8).abs() < EPS);
        // Row 1: norm = 1, unchanged
        assert!((data[[1, 0]] - 1.0).abs() < EPS);
        assert!((data[[1, 1]] - 0.0).abs() < EPS);
        // Row 2: all zeros, stays zero
        assert!((data[[2, 0]] - 0.0).abs() < EPS);
    }

    #[test]
    fn l2_copy_matches_inplace() {
        let data = array![[3.0f32, 4.0], [1.0, 0.0], [0.0, 0.0]];
        let out = normalize_l2(data.view());
        let mut inplace = data.clone();
        normalize_l2_inplace(&mut inplace);
        for (a, b) in out.iter().zip(inplace.iter()) {
            assert!((a - b).abs() < EPS);
        }
        // Spot-check absolute values too.
        assert!((out[[0, 0]] - 0.6).abs() < EPS);
        assert!((out[[0, 1]] - 0.8).abs() < EPS);
        assert!((out[[2, 0]] - 0.0).abs() < EPS);
    }

    #[test]
    fn l1_unit_norm() {
        let mut data = array![[3.0f32, 1.0], [-2.0, 2.0], [0.0, 0.0]];
        normalize_l1_inplace(&mut data);
        // Row 0: L1 = 4, so [0.75, 0.25]
        assert!((data[[0, 0]] - 0.75).abs() < EPS);
        assert!((data[[0, 1]] - 0.25).abs() < EPS);
        // Row 1: L1 = 4, so [-0.5, 0.5]
        assert!((data[[1, 0]] + 0.5).abs() < EPS);
        assert!((data[[1, 1]] - 0.5).abs() < EPS);
        // All-zero row unchanged
        assert!((data[[2, 0]]).abs() < EPS);
    }

    #[test]
    fn l1_copy_matches_inplace() {
        let data = array![[3.0f32, 1.0], [-2.0, 2.0], [0.0, 0.0]];
        let out = normalize_l1(data.view());
        let mut inplace = data.clone();
        normalize_l1_inplace(&mut inplace);
        for (a, b) in out.iter().zip(inplace.iter()) {
            assert!((a - b).abs() < EPS);
        }
    }

    #[test]
    fn max_copy_matches_inplace() {
        let data = array![[2.0f32, -6.0, 3.0], [0.0, 0.0, 0.0]];
        let out = normalize_max(data.view());
        let mut inplace = data.clone();
        normalize_max_inplace(&mut inplace);
        for (a, b) in out.iter().zip(inplace.iter()) {
            assert!((a - b).abs() < EPS);
        }
    }

    #[test]
    fn max_unit_norm() {
        let mut data = array![[2.0f32, -6.0, 3.0], [0.0, 0.0, 0.0]];
        normalize_max_inplace(&mut data);
        // Row 0: max_abs = 6, so [1/3, -1.0, 0.5]
        assert!((data[[0, 0]] - (2.0 / 6.0)).abs() < EPS);
        assert!((data[[0, 1]] + 1.0).abs() < EPS);
        assert!((data[[0, 2]] - 0.5).abs() < EPS);
        // All-zero row unchanged
        assert!(data[[1, 0]].abs() < EPS);
    }

}
