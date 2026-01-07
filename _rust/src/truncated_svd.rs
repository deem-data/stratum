// truncated_svd_faer.rs
//
// Randomized TruncatedSVD for CSR matrices, tuned for StringEncoder.
// Goal: match sklearn TruncatedSVD behavior "well enough" for downstream models,
// while being fast and easy to understand.
//
// Key design choices (important for performance):
// - Build CSC once (transpose of CSR). Then X^T @ Q is fast (column-parallel).
// - Keep l = k + oversample small (k~30, oversample=10 => l~40).
// - Use faer for dense QR + eigen on small l×l Gram matrix.
// - Avoid materializing huge dense matrices like (l × n_cols) explicitly.
//
// TODO (perf):
//  Reuse the CSC across calls for the same sparsity pattern.
//  Custom threadpool (pool_ref) integration for faer (faer uses global rayon).
//  Use f64 for the small l×l Gram for better stability if needed.
//  Compare ndarry::linalg (blas-backed) with faer

use ndarray::{Array2, Axis};
use pyo3::{exceptions::PyValueError, PyErr};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use rayon::ThreadPool;

use faer::{Mat, Side};

fn csr_validate(
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
) -> Result<(), PyErr> {
    if indptr.len() != n_rows + 1 {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Invalid indptr length: {} (expected {})",
            indptr.len(),
            n_rows + 1
        )));
    }
    if data.len() != indices.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Data and indices length mismatch: {} vs {}",
            data.len(),
            indices.len()
        )));
    }
    let nnz = data.len();
    let last = indptr[n_rows] as isize;
    if last < 0 || (last as usize) != nnz {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Invalid CSR: indptr[{}] = {} but data.len() = {}",
            n_rows, indptr[n_rows], nnz
        )));
    }
    // Basic bounds check on indices (cheapish; linear in nnz)
    for &j in indices {
        let ju = j as isize;
        if ju < 0 || (ju as usize) >= n_cols {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Invalid CSR: column index out of bounds: {j} (n_cols={n_cols})"
            )));
        }
    }
    Ok(())
}

/// Build CSC (Compressed Sparse Column) representation from CSR.
///
/// Returns (data_t, row_indices_t, col_ptr_t) where:
/// - col_ptr_t has length n_cols + 1
/// - data_t and row_indices_t have length nnz
/// - entries in each column are not guaranteed sorted by row (not required here)
fn csr_to_csc(
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
) -> (Vec<f32>, Vec<i32>, Vec<i64>) {
    let nnz = data.len();
    let mut col_counts = vec![0i64; n_cols];
    for &j in indices {
        col_counts[j as usize] += 1;
    }

    let mut col_ptr = vec![0i64; n_cols + 1];
    for j in 0..n_cols {
        col_ptr[j + 1] = col_ptr[j] + col_counts[j];
    }

    // Next write position per column.
    let mut next = col_ptr.clone();

    let mut data_t = vec![0.0f32; nnz];
    let mut rows_t = vec![0i32; nnz];

    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for p in start..end {
            let col = indices[p] as usize;
            let dst = next[col] as usize;
            next[col] += 1;
            data_t[dst] = data[p];
            rows_t[dst] = row as i32;
        }
    }

    (data_t, rows_t, col_ptr)
}

/// CSR (n_rows × n_cols) times dense Omega (n_cols × l) -> Y (n_rows × l).
/// Omega is stored column-major: omega[col * l + t] = Omega[col, t].
fn csr_matmul_dense_colmajor(
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
    omega: &[f32],
    l: usize,
    pool_ref: Option<&ThreadPool>,
) -> Array2<f32> {
    let mut y = Array2::<f32>::zeros((n_rows, l));

    let mut work = || {
        y.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(row, mut yrow)| {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                for t in 0..l {
                    let mut acc = 0.0f32;
                    for p in start..end {
                        let j = indices[p] as usize; // validated
                        debug_assert!(j < n_cols);
                        let v = data[p];
                        acc += v * omega[j * l + t];
                    }
                    yrow[t] = acc;
                }
            });
    };

    match pool_ref {
        Some(pool) => pool.install(work),
        None => work(),
    }

    y
}

/// CSC transpose matmul: B = X^T @ Q, where X is CSR but we use CSC for speed.
///
/// X^T is (n_cols × n_rows), Q is (n_rows × l) => B is (n_cols × l).
fn csc_transpose_matmul_dense(
    data_t: &[f32],
    rows_t: &[i32],
    col_ptr: &[i64],
    n_cols: usize,
    q: &Array2<f32>,
    l: usize,
    pool_ref: Option<&ThreadPool>,
) -> Array2<f32> {
    let mut b = Array2::<f32>::zeros((n_cols, l));
    let q_view = q.view(); // safe immutable view

    let mut work = || {
        b.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(col, mut brow)| {
                let start = col_ptr[col] as usize;
                let end = col_ptr[col + 1] as usize;
                for p in start..end {
                    let row = rows_t[p] as usize;
                    let v = data_t[p];
                    // brow[t] += v * q[row,t]
                    for t in 0..l {
                        brow[t] += v * q_view[(row, t)];
                    }
                }
            });
    };

    match pool_ref {
        Some(pool) => pool.install(work),
        None => work(),
    }

    b
}

/// Convert ndarray row-major Array2 into a faer Mat (column-major) by copying.
fn array2_to_faer(a: &Array2<f32>) -> Mat<f32> {
    let (m, n) = a.dim();
    let mut out = Mat::<f32>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            out[(i, j)] = a[(i, j)];
        }
    }
    out
}

/// Convert faer Mat into ndarray Array2 row-major by copying.
fn faer_to_array2(m: &Mat<f32>) -> Array2<f32> {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut out = Array2::<f32>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            out[(i, j)] = m[(i, j)];
        }
    }
    out
}

/// Thin QR: returns Q with shape (m × n) for an m×n input (m>=n typically).
/// Uses faer::Mat::qr under the hood.
fn thin_q_from(a: &Array2<f32>) -> Result<Array2<f32>, PyErr> {
    let a_f = array2_to_faer(a);
    let qr = a_f.qr();
    // Materialize the Thin Q matrix.
    let q_mat = qr.compute_thin_Q();
    Ok(faer_to_array2(&q_mat))
}

/// Eigen decomposition of a small self-adjoint matrix G (l×l).
/// Returns eigenvalues (ascending) and eigenvectors U (l×l).
fn self_adjoint_eigen(g: &Array2<f32>) -> Result<(Vec<f32>, Array2<f32>), PyErr> {
    // Symmetrize G before calling eigen
    let mut g_sym = g.to_owned();
    for i in 0..g_sym.nrows() {
        for j in 0..i {
            let v = 0.5 * (g_sym[(i,j)] + g_sym[(j,i)]);
            g_sym[(i,j)] = v;
            g_sym[(j,i)] = v;
        }
    }
    let g_f = array2_to_faer(&g_sym);
    let eig = g_f
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Eigendecomposition failed: {:?}", e))
        })?;
    let s = eig.S();
    let u = eig.U();
    // Copy out eigenvalues from diagonal
    let s_col = s.column_vector();
    let n_eig = s_col.nrows();
    let mut evals = vec![0.0f32; n_eig];
    for i in 0..n_eig {
        evals[i] = s_col[i];
    }
    let u_nd = faer_to_array2(&(u.to_owned()));
    Ok((evals, u_nd))
}

pub fn truncated_svd_csr(
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
    k: usize,
    n_iter: usize,      // keep 7 in caller
    oversample: usize,  // keep 10 in caller
    seed: Option<u64>,
    pool_ref: Option<&ThreadPool>,
) -> Result<(Array2<f32>, Array2<f32>, Vec<f32>), PyErr> {
    if k == 0 {
        return Err(PyErr::new::<PyValueError, _>("k must be >= 1"));
    }
    if k > n_cols {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "k={} > n_cols={} (cannot compute more components than columns)",
            k, n_cols
        )));
    }

    csr_validate(data, indices, indptr, n_rows, n_cols)?;

    if data.is_empty() || indptr[n_rows] == 0 {
        // X is entirely zero => embedding is zero.
        let out_k = k.min(n_rows).min(n_cols); // sklearn-like effective k
        let z = Array2::<f32>::zeros((n_rows, out_k));
        let components_t = Array2::<f32>::zeros((n_cols, out_k));
        let s_trunc = vec![0.0f32; out_k];
        return Ok((z, components_t, s_trunc))
    }

    let l = k + oversample;
    let l = l.min(n_cols).min(n_rows.max(1)); // keep l reasonable
    if l == 0 {
        return Err(PyErr::new::<PyValueError, _>("l is 0 after bounds check"));
    }

    // Build CSC once for X^T @ Q and for final Gram.
    let t0 = crate::util::start_timing();
    let (data_t, rows_t, col_ptr) = csr_to_csc(data, indices, indptr, n_rows, n_cols);
    crate::util::print_timing("build CSC transpose", t0);

    // Step 1: Gaussian omega (n_cols × l), column-major.
    let t0 = crate::util::start_timing();
    let s = seed.unwrap_or(0xC0FFEE);
    let mut rng = StdRng::seed_from_u64(s);
    let mut omega = Vec::<f32>::with_capacity(n_cols * l);
    let mut use_sin = false;
    let mut z1 = 0.0f32;
    for _ in 0..(n_cols * l) {
        let val = if use_sin {
            use_sin = false;
            z1
        } else {
            use_sin = true;
            // Box-Muller
            let u1: f32 = rng.random::<f32>().max(1e-10);
            let u2: f32 = rng.random::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            z1 = r * (2.0 * std::f32::consts::PI * u2).sin();
            r * (2.0 * std::f32::consts::PI * u2).cos()
        };
        omega.push(val);
    }
    crate::util::print_timing("generate random matrix", t0);

    // Step 2: Y = X @ Ω (n_rows × l)
    let t0 = crate::util::start_timing();
    let y = csr_matmul_dense_colmajor(data, indices, indptr, n_rows, n_cols, &omega, l, pool_ref);
    crate::util::print_timing("compute Y = X @ Ω", t0);

    // Step 3: QR(Y) -> Q (n_rows × l)
    let t0 = crate::util::start_timing();
    let mut q_current = thin_q_from(&y)?;
    crate::util::print_timing("QR decomposition (faer)", t0);

    // Step 4: Power iterations
    // Efficiently compute B = X^T @ Q using CSC (column-parallel).
    for iter in 0..n_iter {
        let t0 = crate::util::start_timing();
        let b = csc_transpose_matmul_dense(&data_t, &rows_t, &col_ptr, n_cols, &q_current, l, pool_ref);
        crate::util::print_timing(&format!("power iter {}: B = X^T @ Q", iter + 1), t0);

        let t0 = crate::util::start_timing();
        let q_b = thin_q_from(&b)?; // (n_cols × l)
        crate::util::print_timing(&format!("power iter {}: QR(B) (faer)", iter + 1), t0);

        let t0 = crate::util::start_timing();
        // Y_new = X @ Q_b (n_rows × l)
        // Q_b is stored row-major in ndarray, but our kernel expects "dense right matrix" in col-major,
        // so we transpose into a column-major buffer once.
        let mut qb_colmaj = vec![0.0f32; n_cols * l];
        for j in 0..n_cols {
            for t in 0..l {
                qb_colmaj[j * l + t] = q_b[(j, t)];
            }
        }
        let y_new = csr_matmul_dense_colmajor(data, indices, indptr, n_rows, n_cols, &qb_colmaj, l, pool_ref);
        crate::util::print_timing(&format!("power iter {}: Y = X @ Q_b", iter + 1), t0);

        let t0 = crate::util::start_timing();
        q_current = thin_q_from(&y_new)?; // (n_rows × l)
        crate::util::print_timing(&format!("power iter {}: QR(Y) (faer)", iter + 1), t0);
    }

    // Step 5: Compute B_t = X^T @ Q (n_cols × l) using CSC, then Gram G = B_t^T B_t (l×l).
    let t0 = crate::util::start_timing();
    let b_t = csc_transpose_matmul_dense(&data_t, &rows_t, &col_ptr, n_cols, &q_current, l, pool_ref);
    crate::util::print_timing("B_t = X^T @ Q", t0);

    let t0 = crate::util::start_timing();
    let mut g = Array2::<f32>::zeros((l, l));
    // G[a,b] = sum_j B_t[j,a] * B_t[j,b]
    // Parallelize over rows j of B_t. Each worker accumulates local G and reduce.
    let build_g = || {
        (0..n_cols).into_par_iter().map(|j| {
            let mut g_loc = Array2::<f32>::zeros((l, l));
            for a in 0..l {
                let ba = b_t[(j, a)];
                if ba != 0.0 {
                    for b in 0..l {
                        g_loc[(a, b)] += ba * b_t[(j, b)];
                    }
                }
            }
            g_loc
        }).reduce(
            || Array2::<f32>::zeros((l, l)),
            |mut acc, g_loc| { acc = &acc + &g_loc; acc }
        )
    };
    g = match pool_ref { Some(pool) => pool.install(build_g), None => build_g() };
    crate::util::print_timing("Gram G = B_t^T B_t", t0);

    // Eigen on small G.
    let t0 = crate::util::start_timing();
    let (eigvals_asc, u_g) = self_adjoint_eigen(&g)?; // ascending
    crate::util::print_timing("eigendecomp(G) (faer)", t0);

    // Take top-k (largest eigenvalues) => last k in ascending order.
    let want_k = k.min(l).min(n_rows);
    if want_k == 0 {
        return Err(PyErr::new::<PyValueError, _>("k is 0 after bounds check"));
    }

    let t0 = crate::util::start_timing();
    let mut u_trunc = Array2::<f32>::zeros((l, want_k));
    let mut s_trunc = vec![0.0f32; want_k];

    // eigvals_asc is ascending; take from the end
    for r in 0..want_k {
        let idx = l - 1 - r; // largest to smaller
        let lam = eigvals_asc[idx].max(0.0);
        s_trunc[r] = lam.sqrt();
        for a in 0..l {
            // u_g columns correspond to eigvals in ascending order too
            u_trunc[(a, r)] = u_g[(a, idx)];
        }
    }

    // Final: Z = Q_current @ U_trunc @ diag(S_trunc)  => (n_rows × want_k)
    let mut z = Array2::<f32>::zeros((n_rows, want_k));
    ndarray::linalg::general_mat_mul(1.0, &q_current, &u_trunc, 0.0, &mut z);

    for j in 0..want_k {
        let s = s_trunc[j];
        if s != 0.0 {
            for i in 0..n_rows {
                z[(i, j)] *= s;
            }
        }
    }
    crate::util::print_timing("final Z = Q U sqrt(Λ)", t0);

    let t0 = crate::util::start_timing();
    let mut components_t = Array2::<f32>::zeros((n_cols, want_k));
    ndarray::linalg::general_mat_mul(1.0, &b_t, &u_trunc, 0.0, &mut components_t);

    // Scale each column by 1/s (avoid div-by-zero)
    for j in 0..want_k {
        let s = s_trunc[j];
        if s != 0.0 {
            let inv = 1.0 / s;
            for i in 0..n_cols {
                components_t[(i, j)] *= inv;
            }
        } else {
            // If s==0, keep that column as zeros (safe)
        }
    }
    crate::util::print_timing("compute components_t = V", t0);

    Ok((z, components_t, s_trunc))
}

pub fn truncated_svd_transform_csr(
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
    components_t: &Array2<f32>, // (n_cols, k)
    pool_ref: Option<&ThreadPool>,
) -> Result<Array2<f32>, PyErr> {
    // Basic CSR validation (you likely already have csr_validate; use that if available)
    if indptr.len() != n_rows + 1 {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Invalid indptr length: {} (expected {})",
            indptr.len(),
            n_rows + 1
        )));
    }
    if data.len() != indices.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "data/indices length mismatch: {} vs {}",
            data.len(),
            indices.len()
        )));
    }
    if (indptr[n_rows] as usize) != data.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Invalid CSR: indptr[n_rows]={} but data.len()={}",
            indptr[n_rows],
            data.len()
        )));
    }

    // Validate components_t shape
    let ct_rows = components_t.nrows();
    let k = components_t.ncols();
    if ct_rows != n_cols {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "components_t shape mismatch: components_t is {}x{}, expected {}xk where n_cols={}",
            ct_rows, k, n_cols, n_cols
        )));
    }

    // If X is all-zero, return all-zero
    if data.is_empty() || indptr[n_rows] == 0 {
        return Ok(Array2::<f32>::zeros((n_rows, k)));
    }

    let mut z = Array2::<f32>::zeros((n_rows, k));

    // Row-parallel: each row writes to its own zrow
    let mut work = || {
        z.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(row, mut zrow)| {
                // row bounds already guaranteed by axis_iter enumerate
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;

                // Accumulate over nnz in this row:
                // z[row, :] += v * components_t[col, :]
                for p in start..end {
                    let col = indices[p] as usize;
                    if col >= n_cols {
                        continue; // defensive: skip invalid
                    }
                    let v = data[p];

                    // components_t row view: (k,)
                    // simple inner loop (fast, predictable)
                    for j in 0..k {
                        zrow[j] += v * components_t[(col, j)];
                    }
                }
            });
    };

    match pool_ref {
        Some(pool) => pool.install(work),
        None => work(),
    }

    Ok(z)
}
