use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

use crate::threads::get_thread_pool;


pub fn nanpercentile(
    x: ArrayView2<f64>,
    references: ArrayView1<f64>,
) -> Array2<f64> {
    let n_features = x.ncols();
    let n_refs = references.len();
    let x_rows = x.nrows();

    // Built transposed: one feature per *row*, so each task writes a contiguous
    // run of n_refs doubles that no other task shares a cache line with. The
    // (n_refs, n_features) layout the caller wants would put adjacent features
    // 8 bytes apart and have every worker fighting over the same lines.
    // `reversed_axes` at the end is a stride swap, not a copy.
    let mut quantiles = Array2::<f64>::zeros((n_features, n_refs));

    // One feature per task. Gathering several columns per task in one row-order
    // sweep was measurably worse: it coarsens the parallel work unit, which costs
    // more in load balancing than the strided gather saves in cache traffic --
    // neighbouring columns are being read concurrently by other workers anyway,
    // so those lines are already resident.
    let mut work = || {
        quantiles
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each_init(
                // One buffer per worker, reused across every feature it takes,
                // instead of a fresh allocation per column.
                || Vec::<f64>::with_capacity(x_rows),
                |buffer, (feature, mut out)| {
                    buffer.clear();

                    let column = x.index_axis(Axis(1), feature);
                    match column.as_slice() {
                        // Contiguous exactly when x is column-major, the layout
                        // sklearn's validation hands over for a DataFrame.
                        Some(slice) => buffer.extend_from_slice(slice),
                        // Otherwise a single strided pointer walk, which still
                        // beats indexing x by (row, feature) element by element.
                        None => buffer.extend(column.iter().copied()),
                    }

                    // NaNs ride along through the gather so it stays a memcpy /
                    // branch-free push, then total_cmp banishes them to the ends:
                    // IEEE totalOrder sorts -NaN below every value and +NaN above
                    // every value, and x86 produces both.
                    buffer.sort_unstable_by(f64::total_cmp);

                    let mut lo = 0;
                    while lo < buffer.len() && buffer[lo].is_nan() {
                        lo += 1;
                    }
                    let mut hi = buffer.len();
                    while hi > lo && buffer[hi - 1].is_nan() {
                        hi -= 1;
                    }
                    let values = &buffer[lo..hi];

                    if values.is_empty() {
                        out.fill(f64::NAN);
                        return;
                    }

                    for (slot, &q) in out.iter_mut().zip(references.iter()) {
                        *slot = percentile_sorted(values, q);
                    }
                },
            );
    };

    match get_thread_pool() {
        Some(pool) => pool.install(work), //use custom threadpool
        None => work(), //use global threadpool
    }

    quantiles.reversed_axes()
}


#[inline(always)]
fn percentile_sorted(values: &[f64], q: f64) -> f64 {
    let n = values.len();

    if n == 1 {
        return values[0];
    }

    let pos = q * (n as f64 - 1.0);

    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    if lo == hi {
        values[lo]
    } else {
        let weight = pos - lo as f64;
        values[lo] * (1.0 - weight) + values[hi] * weight
    }
}
