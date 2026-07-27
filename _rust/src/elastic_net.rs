//! ElasticNet regression via cyclic-Jacobi coordinate descent.
//!
//! Minimises: (1/2n)||y − Xw − b||² + α·ρ·||w||₁ + (α/2)·(1−ρ)·||w||²
//!   where α = alpha, ρ = l1_ratio.
//!
//! Algorithm: fused single-pass coordinate descent
//! -------------------------------------------------
//! The original two-pass Jacobi loop read X twice per iteration:
//!   pass 1 — accumulate X^T·r  (fold/reduce over rows)
//!   pass 2 — update residuals  (par_for_each over rows)
//! For matrices larger than L3 cache (≥ 20 MB) both passes hit DRAM,
//! making the algorithm memory-bandwidth bound.
//!
//! The fused pass reads X once per iteration. Each Rayon task owns a block
//! of B rows (target ~256 KB → fits in L2 cache) and performs both operations
//! before moving on:
//!
//!   for each row-block B:
//!     phase 1 — r[B] -= X[B] · Δw_prev        (X[B] cold → loads into L2)
//!     phase 2 — dots += X[B]^T · r[B]          (X[B] L2-hot; r[B] L1-hot)
//!   coordinate update  (O(n_cols), sequential)
//!
//! Δw_prev is the coefficient delta from the PREVIOUS iteration; applying it
//! in phase 1 brings r in sync with the current coef before accumulating dots.
//! The first iteration is correct because Δw_prev starts at zero.
//!
//! Intercept lag correction
//! -------------------------
//! After the fused pass r reflects coef_prev. The exact intercept update is:
//!   b_delta = mean(r) − dot(col_sum, Δw_cur) / n
//! where col_sum[j] = Σ_i X[i,j] is precomputed once before the main loop.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use crate::threads::get_thread_pool;

// ─── Public types ─────────────────────────────────────────────────────────────

pub struct ElasticNetModel {
    pub n_cols:    usize,
    pub coef:      Vec<f32>,
    pub intercept: f32,
    pub n_iter:    usize,
}

// ─── Core algorithm ───────────────────────────────────────────────────────────

#[inline]
fn soft_threshold(x: f32, t: f32) -> f32 {
    if x > t { x - t } else if x < -t { x + t } else { 0.0 }
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn axpy_scalar(acc: &mut [f32], scale: f32, x: &[f32]) {
    for (a, &xi) in acc.iter_mut().zip(x.iter()) { *a += scale * xi; }
}

/// Fit an ElasticNet model using Jacobi parallel coordinate descent.
///
/// # Parameters
/// - `x`             — (n_rows × n_cols) design matrix, C-contiguous f32
/// - `y`             — (n_rows,) target vector, f32
/// - `alpha`         — total regularisation strength (≥ 0)
/// - `l1_ratio`      — mixing: 0 = Ridge, 1 = Lasso
/// - `max_iter`      — maximum full passes over all coordinates
/// - `tol`           — stop when max relative |Δw_j| < tol
/// - `fit_intercept` — whether to fit an unpenalised bias term
pub fn elastic_net_fit(
    x:             ArrayView2<f32>,
    y:             ArrayView1<f32>,
    alpha:         f32,
    l1_ratio:      f32,
    max_iter:      usize,
    tol:           f32,
    fit_intercept: bool,
) -> Result<ElasticNetModel, String> {
    let (n_rows, n_cols) = x.dim();
    // Reachable from Python, so return an error instead of panicking across FFI.
    if y.len() != n_rows {
        return Err(format!(
            "y length must match number of rows in X: got {} but X has {n_rows} rows",
            y.len(),
        ));
    }
    let n      = n_rows as f32;
    let l1_pen = alpha * l1_ratio;
    let l2_pen = alpha * (1.0 - l1_ratio);

    let x_raw = x.as_slice().ok_or("X must be C-contiguous")?;
    let pool  = get_thread_pool();

    // ── Precompute (1/n)·||X[:,j]||² ──────────────────────────────────────────
    let col_norm_sq: Vec<f32> = {
        let compute = || x_raw
            .par_chunks(n_cols)
            .fold(
                || vec![0.0f32; n_cols],
                |mut acc, row| {
                    for (a, &v) in acc.iter_mut().zip(row.iter()) { *a += v * v; }
                    acc
                },
            )
            .reduce(
                || vec![0.0f32; n_cols],
                |mut a, b| { for j in 0..n_cols { a[j] += b[j]; } a },
            )
            .into_iter()
            .map(|s| s / n)
            .collect();
        match pool { Some(p) => p.install(compute), None => compute() }
    };

    // ── Precompute Σ_i X[i,j] — exact intercept lag correction ───────────────
    let col_sum: Vec<f32> = if fit_intercept {
        let compute = || x_raw
            .par_chunks(n_cols)
            .fold(
                || vec![0.0f32; n_cols],
                |mut acc, row| { axpy_scalar(&mut acc, 1.0, row); acc },
            )
            .reduce(
                || vec![0.0f32; n_cols],
                |mut a, b| { for j in 0..n_cols { a[j] += b[j]; } a },
            );
        match pool { Some(p) => p.install(compute), None => compute() }
    } else {
        Vec::new()
    };

    // ── Initialise ────────────────────────────────────────────────────────────
    let mut coef = vec![0.0f32; n_cols];
    let mut intercept = if fit_intercept { y.iter().sum::<f32>() / n } else { 0.0 };
    let mut r: Vec<f32> = y.iter().map(|&yi| yi - intercept).collect();

    // block_rows: target ~256 KB of X per block to fit in L2 cache.
    let block_rows = (256_usize * 1024 / (n_cols * 4)).clamp(64, 4096);

    let mut prev_deltas = vec![0.0f32; n_cols];
    let mut n_iter = max_iter;

    for iter in 0..max_iter {
        // ── Fused pass: ONE read of X per iteration ───────────────────────────
        // Each Rayon task handles block_rows rows:
        //   phase 1 — r[block] -= X[block] · prev_deltas  (X cold → L2)
        //   phase 2 — dots     += X[block]^T · r[block]   (X L2-hot)
        let dots: Vec<f32> = {
            let mut fused = || x_raw
                .par_chunks(block_rows * n_cols)
                .zip(r.par_chunks_mut(block_rows))
                .fold(
                    || vec![0.0f32; n_cols],
                    |mut local_dots, (x_block, r_block)| {
                        for (row, ri) in x_block.chunks(n_cols).zip(r_block.iter_mut()) {
                            *ri -= dot_scalar(row, &prev_deltas);
                        }
                        for (row, &ri) in x_block.chunks(n_cols).zip(r_block.iter()) {
                            axpy_scalar(&mut local_dots, ri, row);
                        }
                        local_dots
                    },
                )
                .reduce(
                    || vec![0.0f32; n_cols],
                    |mut a, b| { for j in 0..n_cols { a[j] += b[j]; } a },
                );
            match pool { Some(p) => p.install(fused), None => fused() }
        };

        // ── Coordinate update (sequential, O(n_cols)) ─────────────────────────
        let mut max_delta = 0.0f32;
        prev_deltas.fill(0.0);
        for j in 0..n_cols {
            if col_norm_sq[j] == 0.0 { continue; }
            let rho_j = dots[j] / n + coef[j] * col_norm_sq[j];
            let new_w = soft_threshold(rho_j, l1_pen) / (col_norm_sq[j] + l2_pen);
            let delta = new_w - coef[j];
            prev_deltas[j] = delta;
            coef[j]        = new_w;
            let rel = delta.abs() / new_w.abs().max(1.0);
            if rel > max_delta { max_delta = rel; }
        }

        // ── Intercept update (sequential, O(n_rows + n_cols)) ─────────────────
        // Exact correction for the one-iteration lag in r:
        //   mean(r_true) = mean(r) − dot(col_sum, Δw_cur) / n
        if fit_intercept {
            let lag: f32 = col_sum.iter().zip(prev_deltas.iter())
                .map(|(&s, &d)| s * d).sum::<f32>();
            let b_delta = r.iter().sum::<f32>() / n - lag / n;
            if b_delta.abs() > f32::EPSILON {
                for ri in r.iter_mut() { *ri -= b_delta; }
                intercept += b_delta;
            }
        }

        if max_delta < tol { n_iter = iter + 1; break; }
    }

    Ok(ElasticNetModel { n_cols, coef, intercept, n_iter })
}

/// Predict with a fitted model: ŷ = X·w + b.
///
/// Takes a borrowed view (fix #1): predict is read-only, so there is no need to
/// copy the whole design matrix across the FFI boundary. Rows are processed in
/// L2-sized blocks (fix #3) so each Rayon task does substantial work — one dot
/// product per row was far too fine-grained and let scheduling overhead dominate.
pub fn elastic_net_predict(
    x:     ArrayView2<f32>,
    model: &ElasticNetModel,
) -> Result<Vec<f32>, String> {
    let n_cols = model.n_cols;
    if x.ncols() != n_cols {
        return Err(format!(
            "n_cols mismatch: got {} but model expects {}",
            x.ncols(), n_cols
        ));
    }
    let n_rows    = x.nrows();
    let coef      = &model.coef;
    let intercept = model.intercept;
    let raw  = x.as_slice().ok_or("elastic_net_predict: C-contiguous input required")?;
    let pool = get_thread_pool();

    // Coarse task granularity: ~256 KB of X per block, matching the fit loop.
    let block_rows = (256_usize * 1024 / (n_cols * 4)).clamp(64, 4096);

    let mut preds = vec![0.0f32; n_rows];
    let mut compute = || {
        raw.par_chunks(block_rows * n_cols)
            .zip(preds.par_chunks_mut(block_rows))
            .for_each(|(x_block, p_block)| {
                for (row, p) in x_block.chunks(n_cols).zip(p_block.iter_mut()) {
                    *p = dot_scalar(row, coef) + intercept;
                }
            });
    };
    match pool { Some(p) => p.install(compute), None => compute() };
    Ok(preds)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPS: f32 = 1e-2;

    fn arr2(rows: Vec<Vec<f32>>) -> Array2<f32> {
        let n_rows = rows.len();
        let n_cols = rows[0].len();
        Array2::from_shape_vec(
            (n_rows, n_cols),
            rows.into_iter().flatten().collect(),
        ).unwrap()
    }

    #[test]
    fn fits_linear_signal() {
        let x = arr2(vec![
            vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 4.0],
            vec![4.0, 3.0], vec![5.0, 5.0],
        ]);
        let y: Array1<f32> = array![8.0, 7.0, 18.0, 17.0, 25.0];
        let model = elastic_net_fit(x.view(), y.view(), 1e-4, 0.5, 5000, 1e-6, false).unwrap();
        let preds = elastic_net_predict(x.view(), &model).unwrap();
        for (p, &t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 0.1, "pred={p:.3} target={t:.3}");
        }
    }

    #[test]
    fn lasso_produces_sparsity() {
        let x = arr2(vec![
            vec![1.0, 0.0, 5.0], vec![2.0, 0.0, 6.0],
            vec![3.0, 0.0, 7.0], vec![4.0, 0.0, 8.0],
        ]);
        let y: Array1<f32> = array![5.0, 10.0, 15.0, 20.0];
        let model = elastic_net_fit(x.view(), y.view(), 1.0, 1.0, 5000, 1e-6, false).unwrap();
        assert!(model.coef[1].abs() < EPS, "zero column should stay zero");
    }

    #[test]
    fn ridge_does_not_zero_coefficients() {
        let x = arr2(vec![
            vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0], vec![4.0, 5.0],
        ]);
        let y: Array1<f32> = array![3.0, 5.0, 7.0, 9.0];
        let model = elastic_net_fit(x.view(), y.view(), 0.1, 0.0, 5000, 1e-6, false).unwrap();
        assert!(model.coef[0].abs() > 1e-4 || model.coef[1].abs() > 1e-4);
        let preds = elastic_net_predict(x.view(), &model).unwrap();
        let mse: f32 = preds.iter().zip(y.iter()).map(|(p, &t)| (p - t).powi(2)).sum::<f32>() / 4.0;
        assert!(mse < 1.0, "MSE={mse:.4}");
    }

    #[test]
    fn fit_intercept_shifts_prediction() {
        let x = arr2(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let y: Array1<f32> = array![11.0, 12.0, 13.0]; // y = x + 10
        let model = elastic_net_fit(x.view(), y.view(), 1e-4, 0.5, 5000, 1e-6, true).unwrap();
        assert!((model.intercept - 10.0).abs() < 0.5, "intercept={:.3}", model.intercept);
    }

    #[test]
    fn predict_ncols_mismatch_errors() {
        let x = arr2(vec![vec![1.0, 2.0]]);
        let y: Array1<f32> = array![3.0];
        let model = elastic_net_fit(x.view(), y.view(), 0.1, 0.5, 100, 1e-4, false).unwrap();
        let bad = arr2(vec![vec![1.0]]);
        assert!(elastic_net_predict(bad.view(), &model).is_err());
    }

    #[test]
    fn fit_rejects_mismatched_y_length() {
        let x = arr2(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y: Array1<f32> = array![1.0];
        assert!(elastic_net_fit(x.view(), y.view(), 0.1, 0.5, 10, 1e-4, false).is_err());
    }

    #[test]
    fn fit_rejects_non_contiguous_x() {
        let x = arr2(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let y: Array1<f32> = array![1.0, 2.0];
        // Transposed: right shape, not C-contiguous.
        assert!(elastic_net_fit(x.t(), y.view(), 0.1, 0.5, 10, 1e-4, false).is_err());
    }

    #[test]
    fn zero_input_zero_predictions() {
        let x = Array2::<f32>::zeros((4, 3));
        let y: Array1<f32> = array![1.0, 2.0, 3.0, 4.0];
        let model = elastic_net_fit(x.view(), y.view(), 0.1, 0.5, 100, 1e-4, true).unwrap();
        assert!(model.coef.iter().all(|&c| c.abs() < EPS));
    }
}
