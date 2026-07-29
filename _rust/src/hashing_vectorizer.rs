//! sklearn `HashingVectorizer` (word analyzer) — the stateless hashing trick.
//!
//! Documents are processed in parallel chunks. Each chunk reuses a compact
//! `(bucket, sign)` scratch vector across rows, sorts and reduces collisions,
//! and emits one flat CSR block. The blocks are then concatenated into the
//! final CSR arrays without per-row maps or copies.

use ahash::AHashMap as HashMap;
use rayon::prelude::*;

use crate::hashing::{signed_bucket, signed_bucket_ascii_lowercase};
use crate::threads::get_thread_pool;
#[cfg(test)]
use crate::tokenize::word_tokens;
use crate::tokenize::{for_each_word_ngram, for_each_word_token_ascii, word_tokens_ascii};

/// Target number of chunks per Rayon worker.
const CHUNKS_PER_THREAD: usize = 4;
/// Avoid tiny chunks on small corpora where Rayon and merge overhead dominate.
const CHUNK_DOCS_MIN: usize = 1000;
/// Compact sort/reduce wins for ordinary text rows. Beyond this point, switch
/// to a sparse map so very long documents or broad n-gram ranges cannot retain
/// memory proportional to every generated term.
const COMPACT_FEATURE_LIMIT: usize = 256;

/// Row-wise normalization matching sklearn `HashingVectorizer.norm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    None,
    L1,
    L2,
}

/// Options mirroring the sklearn parameters we support on the fast path.
#[derive(Debug)]
pub struct HashingOptions {
    pub n_features: usize,
    pub nmin: usize,
    pub nmax: usize,
    pub binary: bool,
    pub alternate_sign: bool,
    pub lowercase: bool,
    pub norm: Norm,
}

/// Flat CSR block for one document chunk. `indptr` is local and starts at zero.
struct CsrChunk {
    data: Vec<f64>,
    indices: Vec<i32>,
    indptr: Vec<i64>,
}

/// Reusable scratch for compact bucket accumulation and canonical ordering.
struct ChunkScratch {
    compact_features: Vec<(i32, i8)>,
    sparse_counts: HashMap<i32, i64>,
    sparse_features: Vec<(i32, i64)>,
    sparse_mode: bool,
}

impl ChunkScratch {
    fn new() -> Self {
        Self {
            compact_features: Vec::new(),
            sparse_counts: HashMap::new(),
            sparse_features: Vec::new(),
            sparse_mode: false,
        }
    }

    fn clear(&mut self) {
        self.compact_features.clear();
        self.sparse_counts.clear();
        self.sparse_features.clear();
        self.sparse_mode = false;
    }

    #[inline]
    fn add(&mut self, feature: (i32, i8)) {
        if !self.sparse_mode && self.compact_features.len() < COMPACT_FEATURE_LIMIT {
            self.compact_features.push(feature);
            return;
        }

        if !self.sparse_mode {
            for (column, sign) in self.compact_features.drain(..) {
                *self.sparse_counts.entry(column).or_insert(0) += i64::from(sign);
            }
            self.sparse_mode = true;
        }
        *self.sparse_counts.entry(feature.0).or_insert(0) += i64::from(feature.1);
    }
}

/// Transform `documents` into CSR parts `(data, indices, indptr)` of shape
/// `(documents.len(), n_features)`.
///
/// `AsRef<str>` lets the PyO3 boundary pass `PyBackedStr` values, so the kernel
/// reads Python's UTF-8 buffers directly instead of copying every document into
/// an owned Rust `String`.
pub fn transform<S: AsRef<str> + Sync>(
    documents: &[S],
    options: &HashingOptions,
) -> (Vec<f64>, Vec<i32>, Vec<i64>) {
    debug_assert!(
        documents
            .iter()
            .all(|document| document.as_ref().is_ascii()),
        "the HashingVectorizer word kernel requires ASCII documents",
    );
    let t0 = crate::util::start_timing();
    let chunks = map_document_chunks(documents, |chunk| transform_chunk(chunk, options));
    crate::util::print_timing("hv map_chunks", t0);

    let t1 = crate::util::start_timing();
    let out = assemble_chunks(chunks, documents.len());
    crate::util::print_timing("hv assemble_csr", t1);
    out
}

/// Target a few tasks per worker for load balance, but avoid over-parallelizing
/// small corpora into chunks below [`CHUNK_DOCS_MIN`].
fn target_chunk_count(n_docs: usize) -> usize {
    if n_docs == 0 {
        return 0;
    }
    let by_threads = worker_threads().max(1).saturating_mul(CHUNKS_PER_THREAD);
    let by_min_docs = (n_docs / CHUNK_DOCS_MIN).max(1);
    by_threads.min(by_min_docs).min(n_docs).max(1)
}

/// Documents per chunk from [`target_chunk_count`], never larger than
/// `n_docs`.
fn chunk_size(n_docs: usize) -> usize {
    if n_docs == 0 {
        return 1;
    }
    n_docs.div_ceil(target_chunk_count(n_docs)).max(1)
}

fn worker_threads() -> usize {
    match get_thread_pool() {
        Some(pool) => pool.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}

fn map_document_chunks<S, T, F>(documents: &[S], map: F) -> Vec<T>
where
    S: Sync,
    T: Send,
    F: Fn(&[S]) -> T + Sync + Send,
{
    if documents.is_empty() {
        return Vec::new();
    }
    let size = chunk_size(documents.len());
    let run = || documents.par_chunks(size).map(&map).collect();
    match get_thread_pool() {
        Some(pool) => pool.install(run),
        None => run(),
    }
}

fn transform_chunk<S: AsRef<str>>(documents: &[S], options: &HashingOptions) -> CsrChunk {
    let mut scratch = ChunkScratch::new();
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(documents.len() + 1);
    indptr.push(0);

    for document in documents {
        transform_document_into(
            document.as_ref(),
            options,
            &mut scratch,
            &mut data,
            &mut indices,
        );
        indptr.push(indices.len() as i64);
    }

    CsrChunk {
        data,
        indices,
        indptr,
    }
}

fn transform_document_into(
    document: &str,
    options: &HashingOptions,
    scratch: &mut ChunkScratch,
    data: &mut Vec<f64>,
    indices: &mut Vec<i32>,
) {
    scratch.clear();
    accumulate_document(document, options, scratch);

    let row_start = data.len();
    if scratch.sparse_mode {
        scratch
            .sparse_features
            .extend(scratch.sparse_counts.drain());
        scratch
            .sparse_features
            .sort_unstable_by_key(|&(column, _)| column);
        for &(column, value) in &scratch.sparse_features {
            indices.push(column);
            data.push(if options.binary { 1.0 } else { value as f64 });
        }
    } else {
        scratch
            .compact_features
            .sort_unstable_by_key(|&(column, _)| column);
        let mut offset = 0;
        while offset < scratch.compact_features.len() {
            let column = scratch.compact_features[offset].0;
            let mut value = 0i64;
            while offset < scratch.compact_features.len()
                && scratch.compact_features[offset].0 == column
            {
                value += i64::from(scratch.compact_features[offset].1);
                offset += 1;
            }
            indices.push(column);
            data.push(if options.binary { 1.0 } else { value as f64 });
        }
    }

    normalize(&mut data[row_start..], options.norm);
}

fn accumulate_document(document: &str, options: &HashingOptions, scratch: &mut ChunkScratch) {
    debug_assert!(document.is_ascii());
    accumulate_ascii_text(document, options, &mut |feature| scratch.add(feature));
}

#[cfg(test)]
fn accumulate_text(text: &str, options: &HashingOptions, features: &mut Vec<(i32, i8)>) {
    let mut tokens = Vec::with_capacity(16);
    word_tokens(text, &mut tokens);
    for_each_word_ngram(&tokens, options.nmin, options.nmax, |ngram| {
        features.push(signed_bucket(
            ngram.as_bytes(),
            options.n_features,
            options.alternate_sign,
        ));
    });
}

fn accumulate_ascii_text<F>(text: &str, options: &HashingOptions, add_feature: &mut F)
where
    F: FnMut((i32, i8)),
{
    debug_assert!(text.is_ascii());

    let mut hash_term = |term: &str| {
        let feature = if options.lowercase {
            signed_bucket_ascii_lowercase(
                term.as_bytes(),
                options.n_features,
                options.alternate_sign,
            )
        } else {
            signed_bucket(term.as_bytes(), options.n_features, options.alternate_sign)
        };
        add_feature(feature);
    };

    if options.nmin == 1 && options.nmax == 1 {
        for_each_word_token_ascii(text, &mut hash_term);
        return;
    }

    let mut tokens = Vec::with_capacity(16);
    word_tokens_ascii(text, &mut tokens);
    for_each_word_ngram(&tokens, options.nmin, options.nmax, hash_term);
}

fn normalize(data: &mut [f64], norm: Norm) {
    let denominator = match norm {
        Norm::None => return,
        Norm::L1 => data.iter().map(|value| value.abs()).sum(),
        Norm::L2 => data.iter().map(|value| value * value).sum::<f64>().sqrt(),
    };
    if denominator > 0.0 {
        for value in data {
            *value /= denominator;
        }
    }
}

fn assemble_chunks(chunks: Vec<CsrChunk>, document_count: usize) -> (Vec<f64>, Vec<i32>, Vec<i64>) {
    let nonzero_count = chunks.iter().map(|chunk| chunk.indices.len()).sum();
    let mut data = Vec::with_capacity(nonzero_count);
    let mut indices = Vec::with_capacity(nonzero_count);
    let mut indptr = Vec::with_capacity(document_count + 1);
    indptr.push(0);

    let mut nonzero_offset = 0i64;
    for chunk in chunks {
        for &local_end in &chunk.indptr[1..] {
            indptr.push(nonzero_offset + local_end);
        }
        nonzero_offset += chunk.indices.len() as i64;
        data.extend(chunk.data);
        indices.extend(chunk.indices);
    }

    debug_assert_eq!(indptr.len(), document_count + 1);
    (data, indices, indptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts_default(n_features: usize) -> HashingOptions {
        HashingOptions {
            n_features,
            nmin: 1,
            nmax: 1,
            binary: false,
            alternate_sign: true,
            lowercase: true,
            norm: Norm::L2,
        }
    }

    /// Reference implementation of the original per-document code path.
    fn transform_reference(
        documents: &[String],
        options: &HashingOptions,
    ) -> (Vec<f64>, Vec<i32>, Vec<i64>) {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for document in documents {
            let lowered;
            let text = if options.lowercase {
                lowered = document.to_lowercase();
                lowered.as_str()
            } else {
                document.as_str()
            };

            let mut features = Vec::new();
            accumulate_text(text, options, &mut features);
            features.sort_unstable_by_key(|&(column, _)| column);

            let row_start = data.len();
            let mut offset = 0;
            while offset < features.len() {
                let column = features[offset].0;
                let mut value = 0i64;
                while offset < features.len() && features[offset].0 == column {
                    value += i64::from(features[offset].1);
                    offset += 1;
                }
                indices.push(column);
                data.push(if options.binary { 1.0 } else { value as f64 });
            }
            normalize(&mut data[row_start..], options.norm);
            indptr.push(indices.len() as i64);
        }
        (data, indices, indptr)
    }

    fn assert_matches_reference(documents: &[String], options: &HashingOptions) {
        let got = transform(documents, options);
        let expected = transform_reference(documents, options);
        assert_eq!(got.0, expected.0);
        assert_eq!(got.1, expected.1);
        assert_eq!(got.2, expected.2);
    }

    #[test]
    fn empty_docs_produce_empty_rows() {
        let documents = vec![String::new(), "a".to_owned()];
        let (data, indices, indptr) = transform(&documents, &opts_default(1 << 10));
        assert_eq!(indptr, vec![0, 0, 0]);
        assert!(data.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn rows_are_l2_normalized() {
        let documents = vec!["the quick brown fox".to_owned()];
        let (data, _indices, indptr) = transform(&documents, &opts_default(1 << 18));
        assert_eq!(indptr.len(), 2);
        let norm = data.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "row norm = {norm}");
    }

    #[test]
    fn rows_are_l1_normalized() {
        let documents = vec!["the quick brown fox".to_owned()];
        let options = HashingOptions {
            norm: Norm::L1,
            ..opts_default(1 << 18)
        };
        let (data, _indices, indptr) = transform(&documents, &options);
        assert_eq!(indptr.len(), 2);
        let norm = data.iter().map(|value| value.abs()).sum::<f64>();
        assert!((norm - 1.0).abs() < 1e-9, "row l1 norm = {norm}");
    }

    #[test]
    fn lowercase_folds_case() {
        let options = opts_default(1 << 18);
        let lower = transform(&["hello world"], &options);
        let upper = transform(&["HELLO WORLD"], &options);
        assert_eq!(lower.1, upper.1);
    }

    #[test]
    fn optimized_lowercasing_matches_reference() {
        let documents = [
            "the quick brown fox",
            "The Quick BROWN Fox",
            "HELLO, WORLD! hello.",
            "MIXED123 under_score X_Y a b cc",
            "",
            "a",
        ]
        .map(str::to_owned);

        for &(nmin, nmax) in &[(1, 1), (1, 2), (2, 3)] {
            for &binary in &[false, true] {
                for &alternate_sign in &[false, true] {
                    for &norm in &[Norm::None, Norm::L1, Norm::L2] {
                        let options = HashingOptions {
                            n_features: 1 << 16,
                            nmin,
                            nmax,
                            binary,
                            alternate_sign,
                            lowercase: true,
                            norm,
                        };
                        assert_matches_reference(&documents, &options);
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_boundaries_preserve_rows() {
        let documents: Vec<String> = (0..(CHUNK_DOCS_MIN * 3 + 17))
            .map(|row| format!("document number {row} repeated repeated"))
            .collect();
        let options = HashingOptions {
            nmin: 1,
            nmax: 2,
            ..opts_default(1 << 18)
        };
        assert_matches_reference(&documents, &options);
    }

    #[test]
    fn long_ngram_rows_use_sparse_accumulation() {
        let document = (0..300)
            .map(|index| format!("token{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let options = HashingOptions {
            n_features: 1 << 12,
            nmin: 1,
            nmax: 3,
            ..opts_default(1 << 12)
        };
        assert_matches_reference(&[document], &options);
    }

    #[test]
    fn chunk_count_respects_threads_and_min_docs() {
        assert_eq!(target_chunk_count(0), 0);
        assert_eq!(chunk_size(0), 1);
        assert_eq!(chunk_size(1), 1);
        assert_eq!(target_chunk_count(1), 1);
        assert_eq!(target_chunk_count(CHUNK_DOCS_MIN - 1), 1);
        assert_eq!(chunk_size(CHUNK_DOCS_MIN - 1), CHUNK_DOCS_MIN - 1);

        let by_threads = worker_threads().max(1) * CHUNKS_PER_THREAD;
        assert_eq!(
            target_chunk_count(10_000),
            by_threads.min(10_000 / CHUNK_DOCS_MIN),
        );
        assert_eq!(
            target_chunk_count(1_000_000),
            by_threads.min(1_000_000 / CHUNK_DOCS_MIN),
        );
        assert!(chunk_size(100_000) >= CHUNK_DOCS_MIN);
        assert!(target_chunk_count(100) <= 100);
        assert!(chunk_size(100) * target_chunk_count(100) >= 100);
    }
}
