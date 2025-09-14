pub struct CsrParts {
    pub data: Vec<f32>,
    pub indices: Vec<i32>,
    pub indptr: Vec<i64>,
}

impl CsrParts {
    pub fn with_capacity(n_rows: usize, est_nnz: usize) -> Self {
        let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
        indptr.push(0); //CSR invariant. Starts at 0.
        Self {
            data: Vec::with_capacity(est_nnz),
            indices: Vec::with_capacity(est_nnz),
            indptr,
        }
    }
}