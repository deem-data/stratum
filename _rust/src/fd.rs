use ndarray::{Array2, ArrayView2, Axis, s};
use ndarray_linalg::{Lapack, SVDInto, SVD};
use rayon::{ThreadPool, ThreadPoolBuilder};
use rayon::prelude::*;
use pyo3::{exceptions, exceptions::PyValueError, PyErr};

// Simple Frequent Directions (FD) implementation for a tall matrix Y (n x m), where
// m is small (≈ k+p). Maintain a sketch B (l x m) with l = 2k (or slightly larger), then shrinks.
pub fn fd_reduce(y: ArrayView2<f32>, k: usize, pool_ref: Option<&ThreadPool>) -> Result<Array2<f32>, PyErr> {
    let n = y.nrows();
    let m = y.ncols();
    if k == 0 || k > m {
        return Err(PyErr::new::<PyValueError, _>("Invalid k"));
    }

    let ell = (2 * k).min(4 * k).max(k + 1); //TODO: tune

    // Create rayon thread pool
    /*let pool: Option<ThreadPool> = if n_threads > 0 {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(n_threads).build()
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to build rayon pool: {e}")))?;
        Some(thread_pool)
    }
    else { //n_threads = 0. Use global threadpool
        None
    };
    let pool_ref = pool.as_ref();*/


    // The sketch
    let mut b = Array2::<f32>::zeros((ell, m));
    let mut filled = 0usize;

    // Stream rows of Y to B till filled
    for i in 0..n {
        if filled < ell {
            b.slice_mut(s![filled, ..]).assign(&y.slice(s![i, ..]));
            filled += 1;
            continue;
        }
        // B is full. Shrink.
        shrink(&mut b, k)?;
        // After shrink, k rows remain. Append ith row at position k
        b.slice_mut(s![k, ..]).assign(&y.slice(s![i, ..]));
        filled = k + 1;
    }

    // Final shrink
    if filled > k {
        shrink(&mut b, k)?;
    }

    // At this point, top-k rows of B span principal directions in the reduced space (m-dim).
    // Project Y -> Z (n x k): Z = Y · (V_k), but here rows of B already align; use the least squares:
    // We compute R = qr(B_top) and use its columns as basis; for simplicity, do SVD of B_top.
    let b_top = b.slice(s![0..k, ..]).to_owned();
    let (u_opt, s_vec, vt_opt) = b_top
        .svd_into(true, true)
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("SVD failed: {e}")))?;
    let _u = u_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD: U missing"))?;
    let vt = vt_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD: VT missing"))?;
    let p = vt.t().to_owned();

    // Z = Y.P -> (n x m) . (m x k) = (n x k)
    // n is large and m, k are small. Parallelization is better than BLAS/MKL (prefers big blocks)
    let mut z = Array2::<f32>::zeros((n, k));
    let mut do_work = || {
        z.axis_iter_mut(Axis(0)) //mutable view of each row
            .into_par_iter() //convert to parallel iterator
            .zip(y.axis_iter(Axis(0))) //combine z.row_mut and y.row
            .for_each(|(mut zrow, yrow)| {
                for r in 0..k {
                    let mut sum = 0.0f32;
                    for c in 0..m {
                        sum += yrow[c] * p[(c, r)];
                    }
                    zrow[r] = sum;
                }
            });
    };
    match pool_ref {
        Some(pool) => pool.install(do_work), //use custom threadpool
        None => do_work() //use global threadpool
    }

    Ok(z)
}

// Shrink step. Do SVD of B (ell x m).
fn shrink(b: &mut Array2<f32>, k: usize) -> Result<(), PyErr> {
    let  (u_opt, s_vec, vt_opt) = b.view()
        .svd(true, true)
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("SVD (shrink) failed: {e}")))?;
    let u = u_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD (shrink): U missing"))?;
    let vt = vt_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD (shrink): VT missing"))?;
    let delta = s_vec.get(k.saturating_sub(1)).copied().unwrap_or(0.0f32).powi(2); //s_k^2

    // s_i' = sqrt(max(s_i^2 - delta, 0))
    let mut s_shrunk = s_vec.clone();
    for x in s_shrunk.iter_mut() {
        let v = (*x) * (*x) - delta;
        *x = if v > 0.0 {
            v.sqrt()
        }
        else { 0.0 }
    }
    // B' = U * diag(s') * V^T
    // Recompose B' = U * diag(s') * V^T
    // Instead of forming diag(s'), scale columns of U by s'_i and then multiply by VT.
    let (ell, m) = (b.nrows(), b.ncols());
    let r = u.ncols().min(s_shrunk.len()).min(vt.nrows());
    let mut u_scaled = u.slice(s![.., 0..r]).to_owned();
    for j in 0..r {
        let sj = s_shrunk[j];
        if sj == 0.0 { continue; }
        // scale column j
        for i in 0..ell {
            u_scaled[(i, j)] *= sj;
        }
    }
    let vt_r = vt.slice(s![0..r, ..]).to_owned(); // (r × m)

    // B' = (ell × r) @ (r × m)  → (ell × m), BLAS-backed
    let mut b_new = Array2::<f32>::zeros((ell, m));
    // ndarray-linalg's matmul is backed by BLAS/MKL.
    ndarray::linalg::general_mat_mul(1.0, &u_scaled, &vt_r, 0.0, &mut b_new);
    *b = b_new;

    Ok(())
}