use ndarray::{Array2, ArrayView2, Axis, s};
use ndarray_linalg::{SVDInto, SVD};
use rayon::{ThreadPool};
use rayon::prelude::*;
use pyo3::{exceptions::PyValueError, PyErr};

// Simple Frequent Directions (FD) implementation for a tall matrix Y (n x m), where
// m is small (≈ k+p). Maintain a sketch B (l x m) with l = 2k (or slightly larger), then shrinks.
pub fn fd_reduce(y: ArrayView2<f32>, k: usize, pool_ref: Option<&ThreadPool>) -> Result<Array2<f32>, PyErr> {
    let n = y.nrows();
    let m = y.ncols();
    if k == 0 || k > m {
        return Err(PyErr::new::<PyValueError, _>("Invalid k"));
    }

    // Use ell = 8*k to reduce shrink frequency. With k=30, this means:
    // - ell = 240, shrink every ~210 rows
    // Each shrink does a full SVD on the sketch matrix.
    let ell = (8 * k).max(k + 1);

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

/// FD fit: computes both the projection Z and the projection matrix P.
/// Returns (Z, P) where:
/// - Z: (n × k) the reduced embeddings
/// - P: (m × k) the projection matrix to transform new data
pub fn fd_fit(y: ArrayView2<f32>, k: usize, pool_ref: Option<&ThreadPool>) -> Result<(Array2<f32>, Array2<f32>), PyErr> {
    let n = y.nrows();
    let m = y.ncols();
    if k == 0 || k > m {
        return Err(PyErr::new::<PyValueError, _>("Invalid k"));
    }

    // Use ell = 8*k to reduce shrink frequency. With k=30, this means:
    // - ell = 240, shrink every ~210 rows
    // Each shrink does a full SVD on the sketch matrix.
    let ell = (8 * k).max(k + 1);

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

    // At this point, top-k rows of B span principal directions in the reduced space (m-dim).
    // Project Y -> Z (n x k): Z = Y · (V_k), but here rows of B already align; use the least squares:
    let b_filled = b.slice(s![0..filled, ..]);
    let (_, _, vt_opt) = b_filled
        .svd(false, true) // We only need VT, so we skip computing U
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("Final SVD failed: {e}")))?;

    let vt = vt_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("Final SVD: VT missing"))?;

    // P (m x k) is the transpose of the first k rows of VT
    let p = vt.slice(s![0..k, ..]).t().to_owned();

    // Z = Y.P -> (n x m) . (m x k) = (n x k)
    let mut z = Array2::<f32>::zeros((n, k));
    ndarray::linalg::general_mat_mul(1.0, &y, &p, 0.0, &mut z);

    Ok((z, p))
}

// Shrink step. Do SVD of B (ell x m).
fn shrink(b: &mut Array2<f32>, k: usize) -> Result<(), PyErr> {
    // 1. Compute SVD. Note: b.view().svd() returns owned U and VT matrices.
    let (u_opt, s_vec, vt_opt) = b.view()
        .svd(true, true)
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("SVD (shrink) failed: {e}")))?;

    let mut u = u_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD (shrink): U missing"))?;
    let vt = vt_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD (shrink): VT missing"))?;

    // 2. Compute delta (s_k^2)
    // s_vec is already sorted in descending order by ndarray-linalg.
    let s_k = s_vec.get(k.saturating_sub(1)).copied().unwrap_or(0.0);
    let delta = s_k * s_k;

    // 3. Scale U in-place
    // Instead of allocating a new 'u_scaled' matrix, we modify the columns of 'u' directly.
    let r = s_vec.len();
    for j in 0..r {
        let s_sq = s_vec[j] * s_vec[j];
        let s_shrunk = if s_sq > delta { (s_sq - delta).sqrt() } else { 0.0 };

        // Scale column j of U by s_shrunk.
        // If s_shrunk is 0, this effectively zeros out directions beyond the top singular values.
        let mut col = u.column_mut(j);
        col *= s_shrunk;
    }

    // 4. Direct Recomposition into 'b'
    // We use views of the modified U and the original VT to avoid any '.to_owned()' calls.
    let u_view = u.slice(s![.., 0..r]);
    let vt_view = vt.slice(s![0..r, ..]);

    // general_mat_mul(alpha, A, B, beta, C) computes C = alpha*A*B + beta*C.
    // By setting beta to 0.0, we overwrite the contents of 'b' without an intermediate allocation.
    ndarray::linalg::general_mat_mul(1.0, &u_view, &vt_view, 0.0, b);

    Ok(())
}