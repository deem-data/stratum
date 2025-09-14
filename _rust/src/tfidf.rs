use anyhow::Result;
use crate::{tokenize, hashing};
use crate::csr::CsrParts;
use numpy::ndarray::Array1;
use std::collections::HashMap;

#[derive(Debug)]
pub enum Error {
    InvalidAnalyzer,
    InvalidNgramRange,
    Internal,
}

pub(crate) struct Builder {
    analyzer: tokenize::Analyzer,
    nmin: usize,
    nmax: usize,
    nfeat: usize,
}

impl Builder {
    pub fn new(analyzer: &str, nmin: usize, nmax: usize, nfeat: usize) -> Result<Self, Error> {
        let a = match analyzer {
            "char" => tokenize::Analyzer::Char,
            "char_wb" => tokenize::Analyzer::Char_wb,
            _ => return Err(Error::InvalidAnalyzer)
        };
        if nmin == 0 || nmin > nmax {
            return Err(Error::InvalidNgramRange);
        }
        Ok(Self {
            analyzer: a,
            nmin,
            nmax,
            nfeat
        })
    }

    pub fn build_csr(&self, docs: &[String]) -> Result<(Vec<f32>, Vec<i32>, Vec<i64>, Array1<f32>), Error> {
        let n_docs = docs.len();
        let mut df = vec![0u32; self.nfeat];

        // Step1: Create per-doc hashmaps and document frequency (DF)
        let mut per_doc: Vec<HashMap<usize, u32>> = Vec::with_capacity(n_docs); //list of token frequencies
        for s in docs {
            let mut tokens_char: Vec<&str> = Vec::new();
            let mut tokens_wb: Vec<String> = Vec::new(); //FIXME: Avoid allocating Strings
            match self.analyzer {
                tokenize::Analyzer::Char => {
                    tokenize::char_ngrams(s, self.nmin, self.nmax, &mut tokens_char);
                }
                tokenize::Analyzer::Char_wb => {
                    tokenize::char_wb_ngrams(s, self.nmin, self.nmax, &mut tokens_wb);
                }
            }

            let mut seen: HashMap<usize, u32> = HashMap::new(); // bucket -> count
            match self.analyzer {
                tokenize::Analyzer::Char => {
                    for t in tokens_char.iter() {
                        let b = hashing::bucket_id(t, self.nfeat);
                        *seen.entry(b).or_insert(0) += 1;
                    }
                }
                tokenize::Analyzer::Char_wb => {
                    for t in tokens_wb.iter() {
                        let b = hashing::bucket_id(t, self.nfeat);
                        *seen.entry(b).or_insert(0) += 1;
                    }
                }
            }

            // update DF once per present feature
            for (&feat, _) in seen.iter() {
                df[feat] += 1;
            }
            per_doc.push(seen);
        }

        // Step2: Compute smoothed IDF: log((1 + n_docs) / (1 + df)) + 1
        let n_docs_f = n_docs as f32;
        let mut idf = Array1::<f32>::zeros(self.nfeat);
        for j in 0..self.nfeat {
            let dfj = df[j] as f32;
            idf[j] = ((1.0 + n_docs_f) / (1.0 + dfj)).ln() + 1.0;
        }

        // Step3: Build CSR with TF*IDF values and per-row L2 normalization
        let mut csr = CsrParts::with_capacity(n_docs, 0);
        let mut nnz_cum = 0i64;

        for doc_map in per_doc.into_iter() {
            // Convert to sorted vector for stable indices (optional)
            let mut items: Vec<(usize, u32)> = doc_map.into_iter().collect();
            items.sort_by_key(|&(k, _)| k);

            // L2 norm accumulator
            let row_start = csr.data.len();
            let mut norm_sq: f32 = 0.0;
            for (feat, cnt) in items.iter() {
                let tf = *cnt as f32; // TODO: match sklearn's sublinear_tf/norm options
                let val = tf * idf[*feat];
                csr.indices.push(*feat as i32);
                csr.data.push(val);
                norm_sq += val * val;
            }

            let row_end = csr.data.len();
            if norm_sq > 0.0 {
                let scale = 1.0 / norm_sq.sqrt();
                for v in &mut csr.data[row_start..row_end] {
                    *v *= scale;
                }
            }

            // Update cumulative nnz and indptr
            let row_nnz = (row_end - row_start) as i64;
            nnz_cum += row_nnz;
            csr.indptr.push(nnz_cum);
        }

        // Return to get the Python integration wired
        Ok((csr.data, csr.indices, csr.indptr, idf))
    }
}