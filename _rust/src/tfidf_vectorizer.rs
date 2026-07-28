//! sklearn word-analyzer `TfidfVectorizer`.
//!
//! Learned fits are two passes over document chunks: collect DF/TF and
//! discovery order, then retokenize into CSR. Fixed-vocabulary fits tokenize
//! once and derive DF from the emitted indices when IDF is on.
//!
//! Chunk count is `min(n_docs, threads * CHUNKS_PER_THREAD,
//! floor(n_docs / CHUNK_DOCS_MIN))`, with at least one chunk. Chunk size is
//! `ceil(n_docs / n_chunks)`.

use std::sync::Arc;

use ahash::AHashMap as HashMap;
use rayon::prelude::*;

use crate::threads::get_thread_pool;
use crate::tokenize::{for_each_word_ngram, for_each_word_token_ascii, word_tokens_ascii};

const CHUNKS_PER_THREAD: usize = 4;
/// Avoid tiny chunks on small corpora where Rayon/merge overhead dominates.
const CHUNK_DOCS_MIN: usize = 1000;
const MAX_MERGE_SHARDS: usize = 64;
const DENSE_COUNT_FEATURE_LIMIT: usize = 1 << 18;

#[derive(Debug)]
pub enum Error {
    EmptyDocuments,
    EmptyVocabulary,
    NoTermsAfterPruning,
    TooManyFeatures,
    AmbiguousMaxFeatures,
    InvalidMaxFeatures,
    DuplicateVocabularyTerm(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    None,
    L1,
    L2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDtype {
    Float32,
    Float64,
}

impl OutputDtype {
    /// Largest consecutive integer exactly representable by the output dtype.
    fn exact_integer_limit(self) -> u64 {
        match self {
            Self::Float32 => 1 << 24,
            Self::Float64 => 1 << 53,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub nmin: usize,
    pub nmax: usize,
    pub lowercase: bool,
    pub binary: bool,
    pub norm: Norm,
    pub use_idf: bool,
    pub smooth_idf: bool,
    pub sublinear_tf: bool,
    pub output_dtype: OutputDtype,
}

#[derive(Debug)]
pub struct Model {
    vocabulary: HashMap<String, i32>,
    terms: Vec<String>,
    idf: Vec<f64>,
    options: Options,
}

pub struct FitOutput {
    pub model: Arc<Model>,
    pub data: Vec<f64>,
    pub indices: Vec<i32>,
    pub indptr: Vec<i64>,
    /// Feature indices in vocabulary insertion order (sklearn's observable order).
    pub vocabulary_order: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DiscoveryOrder {
    chunk: usize,
    position: usize,
}

impl Default for DiscoveryOrder {
    fn default() -> Self {
        Self {
            chunk: usize::MAX,
            position: usize::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TermStats {
    document_frequency: u64,
    term_frequency: u64,
    first_occurrence: DiscoveryOrder,
    /// Chunk-local document index used to update DF without a per-row term map.
    last_document: usize,
}

impl Default for TermStats {
    fn default() -> Self {
        Self {
            document_frequency: 0,
            term_frequency: 0,
            first_occurrence: DiscoveryOrder::default(),
            last_document: usize::MAX,
        }
    }
}

/// Flat CSR block for one document chunk. `indptr` is local (starts at 0).
struct CsrChunk {
    data: Vec<f64>,
    indices: Vec<i32>,
    indptr: Vec<i64>,
}

/// Reusable per-chunk scratch for counting and row assembly.
struct ChunkScratch {
    feature_counts: FeatureCounts,
    features: Vec<(i32, u64)>,
}

impl ChunkScratch {
    fn new(n_features: usize) -> Self {
        Self {
            feature_counts: if n_features <= DENSE_COUNT_FEATURE_LIMIT {
                FeatureCounts::Dense(vec![0; n_features])
            } else {
                FeatureCounts::Sparse(HashMap::new())
            },
            features: Vec::new(),
        }
    }
}

enum FeatureCounts {
    Dense(Vec<u64>),
    Sparse(HashMap<i32, u64>),
}

impl Model {
    pub fn terms(&self) -> &[String] {
        &self.terms
    }

    pub fn idf(&self) -> &[f64] {
        &self.idf
    }

    pub fn n_features(&self) -> usize {
        self.terms.len()
    }

    pub fn transform<S: AsRef<str> + Sync>(
        &self,
        documents: &[S],
    ) -> (Vec<f64>, Vec<i32>, Vec<i64>) {
        debug_assert!(
            documents
                .iter()
                .all(|document| document.as_ref().is_ascii()),
            "the TF-IDF word kernel requires ASCII documents",
        );
        let t0 = crate::util::start_timing();
        let chunks = map_document_chunks(documents, |chunk| emit_weighted_chunk(chunk, self, None));
        crate::util::print_timing("tv map_chunks", t0);

        let t1 = crate::util::start_timing();
        let out = assemble_chunks(chunks, documents.len());
        crate::util::print_timing("tv assemble_csr", t1);
        out
    }
}

/// Fit and return the model plus the training CSR matrix.
///
/// `fixed_terms`, if set, must be ordered by feature index.
/// `min_document_count` / `max_document_count` are absolute sklearn thresholds
/// (already expanded from fractions). Documents are `AsRef<str>` so callers
/// can pass `String` or `PyBackedStr` without copying.
pub fn fit<S: AsRef<str> + Sync>(
    documents: &[S],
    options: Options,
    fixed_terms: Option<Vec<String>>,
    min_document_count: f64,
    max_document_count: f64,
    max_features: Option<usize>,
) -> Result<FitOutput, Error> {
    if max_features == Some(0) {
        return Err(Error::InvalidMaxFeatures);
    }
    if documents.is_empty() {
        return Err(Error::EmptyDocuments);
    }
    debug_assert!(
        documents
            .iter()
            .all(|document| document.as_ref().is_ascii()),
        "the TF-IDF word kernel requires ASCII documents",
    );

    match fixed_terms {
        Some(terms) => fit_fixed_vocabulary(documents, options, terms),
        None => fit_learned_vocabulary(
            documents,
            options,
            min_document_count,
            max_document_count,
            max_features,
        ),
    }
}

fn fit_learned_vocabulary<S: AsRef<str> + Sync>(
    documents: &[S],
    options: Options,
    min_document_count: f64,
    max_document_count: f64,
    max_features: Option<usize>,
) -> Result<FitOutput, Error> {
    // Pass A: DF/TF + discovery order per chunk.
    let t0 = crate::util::start_timing();
    let mut chunk_stats =
        map_document_chunks(documents, |chunk| accumulate_chunk_stats(chunk, &options));
    for (chunk, stats) in chunk_stats.iter_mut().enumerate() {
        for term_stats in stats.values_mut() {
            term_stats.first_occurrence.chunk = chunk;
        }
    }
    let stats = merge_term_stats(chunk_stats);
    crate::util::print_timing("tv pass_a_stats", t0);

    let t1 = crate::util::start_timing();
    let terms = select_terms(
        stats.iter(),
        min_document_count,
        max_document_count,
        max_features,
        options.output_dtype.exact_integer_limit(),
    )?;
    ensure_feature_count_fits_i32(terms.len())?;

    let vocabulary = vocabulary_from_terms(&terms);
    let idf = terms
        .iter()
        .map(|term| {
            let df = stats
                .get(term.as_str())
                .map_or(0, |term_stats| term_stats.document_frequency);
            compute_idf(documents.len() as u64, df, &options)
        })
        .collect();

    let model = Arc::new(Model {
        vocabulary,
        terms,
        idf,
        options,
    });

    // sklearn fit_transform emits rows in discovery order, then remaps
    // feature ids lexicographically without re-sorting. Match that layout;
    // transform uses final feature-index order.
    let fit_feature_order: Vec<DiscoveryOrder> = model
        .terms
        .iter()
        .map(|term| stats[term.as_str()].first_occurrence)
        .collect();
    let mut vocabulary_order: Vec<i32> = (0..model.n_features() as i32).collect();
    vocabulary_order.sort_unstable_by_key(|&feature| fit_feature_order[feature as usize]);
    crate::util::print_timing("tv select_vocab", t1);

    // Pass B: retokenize into weighted CSR chunks.
    let t2 = crate::util::start_timing();
    let chunks = map_document_chunks(documents, |chunk| {
        emit_weighted_chunk(chunk, &model, Some(&fit_feature_order))
    });
    crate::util::print_timing("tv pass_b_emit", t2);

    let t3 = crate::util::start_timing();
    let (data, indices, indptr) = assemble_chunks(chunks, documents.len());
    crate::util::print_timing("tv assemble_csr", t3);

    Ok(FitOutput {
        model,
        data,
        indices,
        indptr,
        vocabulary_order,
    })
}

fn fit_fixed_vocabulary<S: AsRef<str> + Sync>(
    documents: &[S],
    options: Options,
    terms: Vec<String>,
) -> Result<FitOutput, Error> {
    if terms.is_empty() {
        return Err(Error::EmptyVocabulary);
    }
    ensure_feature_count_fits_i32(terms.len())?;

    let n_features = terms.len();
    let vocabulary = fixed_vocabulary_from_terms(&terms)?;
    let vocabulary_order = (0..n_features as i32).collect();

    // One tokenize pass for TF. Each feature appears at most once per row,
    // so the indices double as DF observations.
    let t0 = crate::util::start_timing();
    let chunks = map_document_chunks(documents, |chunk| {
        emit_tf_chunk(chunk, &vocabulary, n_features, &options)
    });
    crate::util::print_timing("tv map_chunks", t0);

    let t1 = crate::util::start_timing();
    let idf = fixed_vocabulary_idf(&chunks, n_features, documents.len(), &options);

    let (mut data, indices, indptr) = assemble_chunks(chunks, documents.len());
    apply_idf_and_normalize(&mut data, &indices, &indptr, &idf, &options);
    crate::util::print_timing("tv assemble_csr", t1);

    let model = Arc::new(Model {
        vocabulary,
        terms,
        idf,
        options,
    });

    Ok(FitOutput {
        model,
        data,
        indices,
        indptr,
        vocabulary_order,
    })
}

fn vocabulary_from_terms(terms: &[String]) -> HashMap<String, i32> {
    let mut vocabulary = HashMap::with_capacity(terms.len());
    for (feature, term) in terms.iter().enumerate() {
        vocabulary.insert(term.clone(), feature as i32);
    }
    vocabulary
}

fn fixed_vocabulary_from_terms(terms: &[String]) -> Result<HashMap<String, i32>, Error> {
    let mut vocabulary = HashMap::with_capacity(terms.len());
    for (feature, term) in terms.iter().enumerate() {
        if vocabulary.insert(term.clone(), feature as i32).is_some() {
            return Err(Error::DuplicateVocabularyTerm(term.clone()));
        }
    }
    Ok(vocabulary)
}

/// Target chunk count: a few tasks per Rayon thread for load balance, but
/// never so many that chunks fall below [`CHUNK_DOCS_MIN`] (over-parallelizes
/// small corpora). Also capped at `n_docs`.
fn target_chunk_count(n_docs: usize) -> usize {
    if n_docs == 0 {
        return 0;
    }
    let by_threads = worker_threads().max(1).saturating_mul(CHUNKS_PER_THREAD);
    let by_min_docs = (n_docs / CHUNK_DOCS_MIN).max(1);
    by_threads.min(by_min_docs).min(n_docs).max(1)
}

/// Documents per chunk from [`target_chunk_count`], never larger than `n_docs`.
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

fn accumulate_chunk_stats<S: AsRef<str>>(
    documents: &[S],
    options: &Options,
) -> HashMap<String, TermStats> {
    let mut stats: HashMap<String, TermStats> = HashMap::new();
    let mut next_discovery_position = 0;

    for (document_index, document) in documents.iter().enumerate() {
        for_each_document_term(document.as_ref(), options, |term| {
            if let Some(term_stats) = stats.get_mut(term) {
                let first_in_document = term_stats.last_document != document_index;
                if first_in_document {
                    term_stats.document_frequency += 1;
                    term_stats.last_document = document_index;
                }
                if !options.binary || first_in_document {
                    term_stats.term_frequency += 1;
                }
                return;
            }

            stats.insert(
                term.to_owned(),
                TermStats {
                    document_frequency: 1,
                    term_frequency: 1,
                    first_occurrence: DiscoveryOrder {
                        chunk: 0,
                        position: next_discovery_position,
                    },
                    last_document: document_index,
                },
            );
            next_discovery_position += 1;
        });
    }

    // The marker has no meaning outside this chunk and must not affect tests
    // or merged statistics.
    for term_stats in stats.values_mut() {
        term_stats.last_document = usize::MAX;
    }
    stats
}

/// Merge per-chunk term stats.
///
/// Hash-shard terms so each key is owned by one worker. A tree `reduce` would
/// rehash the growing vocabulary at every level; this stays linear in the sum
/// of chunk-map sizes. Semantics (DF/TF, earliest discovery) are unchanged.
fn merge_term_stats(chunk_stats: Vec<HashMap<String, TermStats>>) -> HashMap<String, TermStats> {
    match chunk_stats.len() {
        0 => return HashMap::new(),
        1 => return chunk_stats.into_iter().next().unwrap(),
        _ => {}
    }

    // Power of two so we can mask; capped so tiny pools don't over-shard.
    let shards = worker_threads()
        .max(1)
        .next_power_of_two()
        .min(MAX_MERGE_SHARDS);
    if shards == 1 {
        return merge_shard(chunk_stats.into_iter().flatten());
    }

    // Fixed seeds: same term always lands in the same shard for this merge.
    let hasher = ahash::RandomState::with_seeds(0, 0, 0, 0);
    let shard_mask = shards - 1;
    let shard_of = |term: &str| (hasher.hash_one(term) as usize) & shard_mask;

    let merge = || {
        let per_chunk: Vec<Vec<Vec<(String, TermStats)>>> = chunk_stats
            .into_par_iter()
            .map(|map| {
                let mut buckets: Vec<Vec<(String, TermStats)>> =
                    (0..shards).map(|_| Vec::new()).collect();
                for (term, stats) in map {
                    let shard = shard_of(&term);
                    buckets[shard].push((term, stats));
                }
                buckets
            })
            .collect();

        let mut shard_inputs: Vec<Vec<Vec<(String, TermStats)>>> = (0..shards)
            .map(|_| Vec::with_capacity(per_chunk.len()))
            .collect();
        for chunk_buckets in per_chunk {
            for (shard, bucket) in chunk_buckets.into_iter().enumerate() {
                shard_inputs[shard].push(bucket);
            }
        }

        // Disjoint keys per shard, so the final extend just concatenates.
        let shard_maps: Vec<HashMap<String, TermStats>> = shard_inputs
            .into_par_iter()
            .map(|buckets| merge_shard(buckets.into_iter().flatten()))
            .collect();

        let total = shard_maps.iter().map(|shard_map| shard_map.len()).sum();
        let mut merged = HashMap::with_capacity(total);
        for shard_map in shard_maps {
            merged.extend(shard_map);
        }
        merged
    };

    match get_thread_pool() {
        Some(pool) => pool.install(merge),
        None => merge(),
    }
}

/// Sum term statistics from an iterator of `(term, stats)` pairs into one map.
fn merge_shard<I>(entries: I) -> HashMap<String, TermStats>
where
    I: Iterator<Item = (String, TermStats)>,
{
    let mut map: HashMap<String, TermStats> = HashMap::new();
    for (term, stats) in entries {
        let entry = map.entry(term).or_default();
        entry.document_frequency += stats.document_frequency;
        entry.term_frequency += stats.term_frequency;
        entry.first_occurrence = entry.first_occurrence.min(stats.first_occurrence);
    }
    map
}

fn emit_weighted_chunk<S: AsRef<str>>(
    documents: &[S],
    model: &Model,
    feature_order: Option<&[DiscoveryOrder]>,
) -> CsrChunk {
    let mut scratch = ChunkScratch::new(model.n_features());
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(documents.len() + 1);
    indptr.push(0);

    for document in documents {
        count_features_into(
            document.as_ref(),
            &model.options,
            &model.vocabulary,
            &mut scratch,
        );
        if let Some(feature_order) = feature_order {
            scratch
                .features
                .sort_unstable_by_key(|&(feature, _)| feature_order[feature as usize]);
        } else {
            scratch
                .features
                .sort_unstable_by_key(|&(feature, _)| feature);
        }

        let row_start = data.len();
        for &(feature, count) in &scratch.features {
            let mut value = tf_value(count, &model.options);
            if model.options.use_idf {
                value *= model.idf[feature as usize];
            }
            indices.push(feature);
            data.push(value);
        }
        normalize(&mut data[row_start..], model.options.norm);
        indptr.push(indices.len() as i64);
    }

    CsrChunk {
        data,
        indices,
        indptr,
    }
}

fn emit_tf_chunk<S: AsRef<str>>(
    documents: &[S],
    vocabulary: &HashMap<String, i32>,
    n_features: usize,
    options: &Options,
) -> CsrChunk {
    let mut scratch = ChunkScratch::new(n_features);
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(documents.len() + 1);
    indptr.push(0);

    for document in documents {
        count_features_into(document.as_ref(), options, vocabulary, &mut scratch);
        scratch
            .features
            .sort_unstable_by_key(|&(feature, _)| feature);

        for &(feature, count) in &scratch.features {
            indices.push(feature);
            data.push(tf_value(count, options));
        }
        indptr.push(indices.len() as i64);
    }

    CsrChunk {
        data,
        indices,
        indptr,
    }
}

fn fixed_vocabulary_idf(
    chunks: &[CsrChunk],
    n_features: usize,
    document_count: usize,
    options: &Options,
) -> Vec<f64> {
    if !options.use_idf {
        return vec![1.0; n_features];
    }

    let mut df = vec![0u64; n_features];
    for chunk in chunks {
        for &feature in &chunk.indices {
            df[feature as usize] += 1;
        }
    }
    df.into_iter()
        .map(|document_frequency| compute_idf(document_count as u64, document_frequency, options))
        .collect()
}

fn assemble_chunks(chunks: Vec<CsrChunk>, n_docs: usize) -> (Vec<f64>, Vec<i32>, Vec<i64>) {
    let nonzero_count = chunks.iter().map(|chunk| chunk.indices.len()).sum();
    let mut data = Vec::with_capacity(nonzero_count);
    let mut indices = Vec::with_capacity(nonzero_count);
    let mut indptr = Vec::with_capacity(n_docs + 1);
    indptr.push(0);

    let mut nnz_offset = 0i64;
    for chunk in chunks {
        let row_count = chunk.indptr.len().saturating_sub(1);
        for row in 0..row_count {
            indptr.push(nnz_offset + chunk.indptr[row + 1]);
        }
        nnz_offset += chunk.indices.len() as i64;
        data.extend(chunk.data);
        indices.extend(chunk.indices);
    }

    debug_assert_eq!(indptr.len(), n_docs + 1);
    (data, indices, indptr)
}

fn apply_idf_and_normalize(
    data: &mut [f64],
    indices: &[i32],
    indptr: &[i64],
    idf: &[f64],
    options: &Options,
) {
    let n_rows = indptr.len().saturating_sub(1);
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        if options.use_idf {
            for offset in start..end {
                data[offset] *= idf[indices[offset] as usize];
            }
        }
        normalize(&mut data[start..end], options.norm);
    }
}

#[inline]
fn tf_value(count: u64, options: &Options) -> f64 {
    let mut value = if options.binary { 1.0 } else { count as f64 };
    if options.sublinear_tf {
        value = value.ln() + 1.0;
    }
    value
}

#[cfg(test)]
fn count_document_into(document: &str, options: &Options, counts: &mut HashMap<String, u64>) {
    counts.clear();
    for_each_document_term(document, options, |term| {
        if let Some(count) = counts.get_mut(term) {
            *count += 1;
        } else {
            counts.insert(term.to_owned(), 1);
        }
    });
}

fn count_features_into(
    document: &str,
    options: &Options,
    vocabulary: &HashMap<String, i32>,
    scratch: &mut ChunkScratch,
) {
    scratch.features.clear();
    match &mut scratch.feature_counts {
        FeatureCounts::Dense(counts) => {
            for_each_document_term(document, options, |term| {
                if let Some(&feature) = vocabulary.get(term) {
                    let count = &mut counts[feature as usize];
                    if *count == 0 {
                        scratch.features.push((feature, 0));
                    }
                    *count += 1;
                }
            });
            for (feature, count) in &mut scratch.features {
                let dense_count = &mut counts[*feature as usize];
                *count = *dense_count;
                *dense_count = 0;
            }
        }
        FeatureCounts::Sparse(counts) => {
            counts.clear();
            for_each_document_term(document, options, |term| {
                if let Some(&feature) = vocabulary.get(term) {
                    *counts.entry(feature).or_insert(0) += 1;
                }
            });
            scratch.features.extend(counts.drain());
        }
    }
}

fn for_each_document_term<F: FnMut(&str)>(document: &str, options: &Options, mut visit: F) {
    debug_assert!(document.is_ascii());

    if !options.lowercase {
        for_each_term_from_ascii_text(document, options, &mut visit);
        return;
    }

    if document.as_bytes().iter().any(u8::is_ascii_uppercase) {
        // ASCII upper: byte lowercasing matches `str.lower()` without Unicode tables.
        let lowered = document.to_ascii_lowercase();
        for_each_term_from_ascii_text(&lowered, options, &mut visit);
    } else {
        for_each_term_from_ascii_text(document, options, &mut visit);
    }
}

fn for_each_term_from_ascii_text<F: FnMut(&str)>(text: &str, options: &Options, visit: &mut F) {
    if options.nmin == 1 && options.nmax == 1 {
        for_each_word_token_ascii(text, visit);
        return;
    }

    let mut tokens = Vec::with_capacity(16);
    word_tokens_ascii(text, &mut tokens);
    for_each_term_from_tokens(&tokens, options, visit);
}

fn for_each_term_from_tokens<F: FnMut(&str)>(tokens: &[&str], options: &Options, visit: &mut F) {
    if options.nmin == 1 {
        for &token in tokens {
            visit(token);
        }
    }
    if options.nmax >= 2 {
        for_each_word_ngram(tokens, options.nmin.max(2), options.nmax, |ngram| {
            visit(ngram);
        });
    }
}

#[cfg(test)]
fn fill_counts_from_ascii_text(text: &str, options: &Options, counts: &mut HashMap<String, u64>) {
    for_each_term_from_ascii_text(text, options, &mut |term| {
        if let Some(count) = counts.get_mut(term) {
            *count += 1;
        } else {
            counts.insert(term.to_owned(), 1);
        }
    });
}

#[inline]
fn ensure_feature_count_fits_i32(n_features: usize) -> Result<(), Error> {
    if n_features > i32::MAX as usize {
        return Err(Error::TooManyFeatures);
    }
    Ok(())
}

fn select_terms<'a>(
    stats: impl Iterator<Item = (&'a String, &'a TermStats)>,
    min_document_count: f64,
    max_document_count: f64,
    max_features: Option<usize>,
    exact_integer_limit: u64,
) -> Result<Vec<String>, Error> {
    let mut candidates: Vec<(&String, &TermStats)> = stats
        .filter(|(_, term_stats)| {
            let df = term_stats.document_frequency as f64;
            df >= min_document_count && df <= max_document_count
        })
        .collect();

    if candidates.is_empty() {
        return Err(Error::NoTermsAfterPruning);
    }

    // Lexicographic order first (sklearn), then use it as the TF tie-breaker.
    candidates.sort_unstable();
    if let Some(limit) = max_features {
        if candidates.len() > limit {
            candidates.sort_by(|(left_term, left_stats), (right_term, right_stats)| {
                right_stats
                    .term_frequency
                    .cmp(&left_stats.term_frequency)
                    .then_with(|| left_term.cmp(right_term))
            });
            // sklearn casts counts to the configured matrix dtype before
            // summing them for max_features. Above the dtype's consecutive
            // integer range, rounding can change which term lies at the
            // boundary. The exact u64 ranking cannot safely predict that
            // result, so delegate this rare case to sklearn.
            if candidates[limit - 1].1.term_frequency > exact_integer_limit {
                return Err(Error::AmbiguousMaxFeatures);
            }
            // NumPy's unstable sort can break TF ties either way across releases.
            // Fall back when a tie sits on the max_features boundary.
            if candidates[limit - 1].1.term_frequency == candidates[limit].1.term_frequency {
                return Err(Error::AmbiguousMaxFeatures);
            }
            candidates.truncate(limit);
            candidates.sort_unstable();
        }
    }

    Ok(candidates
        .into_iter()
        .map(|(term, _)| term.clone())
        .collect())
}

fn compute_idf(document_count: u64, document_frequency: u64, options: &Options) -> f64 {
    if !options.use_idf {
        return 1.0;
    }

    let smoothing = u64::from(options.smooth_idf);
    let numerator = (document_count + smoothing) as f64;
    let denominator = (document_frequency + smoothing) as f64;
    (numerator / denominator).ln() + 1.0
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

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Options {
        Options {
            nmin: 1,
            nmax: 1,
            lowercase: true,
            binary: false,
            norm: Norm::L2,
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
            output_dtype: OutputDtype::Float64,
        }
    }

    #[test]
    fn feature_count_above_i32_max_is_rejected() {
        assert!(matches!(
            ensure_feature_count_fits_i32(i32::MAX as usize),
            Ok(())
        ));
        assert!(matches!(
            ensure_feature_count_fits_i32((i32::MAX as usize) + 1),
            Err(Error::TooManyFeatures)
        ));
    }

    #[test]
    fn fit_orders_vocabulary_lexicographically() {
        let documents = vec!["zebra apple apple".to_owned(), "pear".to_owned()];
        let output = fit(&documents, defaults(), None, 1.0, 2.0, None).unwrap();
        assert_eq!(output.model.terms(), &["apple", "pear", "zebra"]);
        assert_eq!(output.indices, vec![2, 0, 1]);
        assert_eq!(output.indptr, vec![0, 2, 3]);
    }

    #[test]
    fn learned_fit_preserves_discovery_order_after_feature_remapping() {
        let documents = vec!["one two three four".to_owned()];
        let output = fit(
            &documents,
            Options {
                nmax: 2,
                norm: Norm::None,
                use_idf: false,
                ..defaults()
            },
            None,
            1.0,
            1.0,
            None,
        )
        .unwrap();

        assert_eq!(
            output.model.terms(),
            &[
                "four",
                "one",
                "one two",
                "three",
                "three four",
                "two",
                "two three",
            ]
        );
        assert_eq!(output.vocabulary_order, vec![1, 5, 3, 0, 2, 6, 4]);
        assert_eq!(output.indices, vec![1, 5, 3, 0, 2, 6, 4]);
    }

    #[test]
    fn learned_fit_preserves_discovery_order_across_chunks() {
        // Put "apple" only in the last doc (second chunk) so its discovery
        // order is after "zebra" from the first chunk.
        let n_docs = CHUNK_DOCS_MIN * 2;
        assert!(
            target_chunk_count(n_docs) >= 2,
            "test needs multiple chunks"
        );
        let mut documents = vec!["zebra".to_owned(); n_docs];
        documents[n_docs - 1] = "zebra apple".to_owned();
        let output = fit(
            &documents,
            Options {
                norm: Norm::None,
                use_idf: false,
                ..defaults()
            },
            None,
            1.0,
            documents.len() as f64,
            None,
        )
        .unwrap();

        assert_eq!(output.model.terms(), &["apple", "zebra"]);
        assert_eq!(output.vocabulary_order, vec![1, 0]);
        assert_eq!(&output.indices[output.indices.len() - 2..], &[1, 0],);
    }

    #[test]
    fn max_features_uses_corpus_term_frequency() {
        let documents = vec!["zebra apple apple".to_owned(), "pear apple".to_owned()];
        let output = fit(&documents, defaults(), None, 1.0, 2.0, Some(1)).unwrap();
        assert_eq!(output.model.terms(), &["apple"]);
    }

    #[test]
    fn float32_max_features_falls_back_beyond_exact_integer_range() {
        let stats = HashMap::from([
            (
                "aa".to_owned(),
                TermStats {
                    document_frequency: 1,
                    term_frequency: 1 << 24,
                    ..TermStats::default()
                },
            ),
            (
                "bb".to_owned(),
                TermStats {
                    document_frequency: 1,
                    term_frequency: (1 << 24) + 1,
                    ..TermStats::default()
                },
            ),
        ]);

        let float32_result = select_terms(
            stats.iter(),
            1.0,
            1.0,
            Some(1),
            OutputDtype::Float32.exact_integer_limit(),
        );
        assert!(matches!(float32_result, Err(Error::AmbiguousMaxFeatures)));

        let float64_result = select_terms(
            stats.iter(),
            1.0,
            1.0,
            Some(1),
            OutputDtype::Float64.exact_integer_limit(),
        )
        .unwrap();
        assert_eq!(float64_result, vec!["bb"]);
    }

    #[test]
    fn chunk_stats_count_binary_tf_once_per_document() {
        let documents = ["apple apple banana", "apple banana banana"];
        let mut options = defaults();
        options.binary = true;

        let stats = accumulate_chunk_stats(&documents, &options);
        assert_eq!(stats["apple"].document_frequency, 2);
        assert_eq!(stats["apple"].term_frequency, 2);
        assert_eq!(stats["banana"].document_frequency, 2);
        assert_eq!(stats["banana"].term_frequency, 2);

        options.binary = false;
        let stats = accumulate_chunk_stats(&documents, &options);
        assert_eq!(stats["apple"].document_frequency, 2);
        assert_eq!(stats["apple"].term_frequency, 3);
        assert_eq!(stats["banana"].document_frequency, 2);
        assert_eq!(stats["banana"].term_frequency, 3);
    }

    #[test]
    fn dense_and_sparse_feature_counters_match() {
        let vocabulary = HashMap::from([("apple".to_owned(), 0), ("banana".to_owned(), 1)]);
        let options = defaults();

        let mut dense = ChunkScratch::new(vocabulary.len());
        count_features_into(
            "banana apple apple unknown",
            &options,
            &vocabulary,
            &mut dense,
        );
        dense.features.sort_unstable();

        let mut sparse = ChunkScratch::new(DENSE_COUNT_FEATURE_LIMIT + 1);
        count_features_into(
            "banana apple apple unknown",
            &options,
            &vocabulary,
            &mut sparse,
        );
        sparse.features.sort_unstable();

        assert_eq!(dense.features, vec![(0, 2), (1, 1)]);
        assert_eq!(sparse.features, dense.features);
    }

    #[test]
    fn max_features_zero_returns_an_error_instead_of_panicking() {
        let documents = vec!["apple banana".to_owned()];
        let result =
            std::panic::catch_unwind(|| fit(&documents, defaults(), None, 1.0, 1.0, Some(0)));
        assert!(result.is_ok(), "max_features=0 must not panic");
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn transform_ignores_unknown_terms_and_normalizes_rows() {
        let documents = vec!["known token".to_owned(), "known".to_owned()];
        let output = fit(&documents, defaults(), None, 1.0, 2.0, None).unwrap();
        let transformed = output
            .model
            .transform(&["known missing missing".to_owned()]);
        assert_eq!(transformed.1.len(), 1);
        assert!((transformed.0[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fixed_vocabulary_keeps_unseen_terms() {
        let documents = vec!["known token".to_owned()];
        let terms = vec!["known".to_owned(), "unseen".to_owned()];
        let output = fit(&documents, defaults(), Some(terms), 1.0, 1.0, None).unwrap();
        assert_eq!(output.model.n_features(), 2);
        assert_eq!(output.vocabulary_order, vec![0, 1]);
        assert!(output.model.idf()[1] > output.model.idf()[0]);
    }

    #[test]
    fn fixed_vocabulary_rejects_duplicate_terms() {
        let documents = vec!["duplicate".to_owned()];
        let terms = vec!["duplicate".to_owned(), "duplicate".to_owned()];
        let result = fit(&documents, defaults(), Some(terms), 1.0, 1.0, None);

        assert!(matches!(
            result,
            Err(Error::DuplicateVocabularyTerm(term)) if term == "duplicate"
        ));
    }

    #[test]
    fn fixed_vocabulary_document_frequency_counts_each_feature_once_per_document() {
        let documents = vec![
            "known known known shared".to_owned(),
            "known shared shared".to_owned(),
            "shared irrelevant-token".to_owned(),
        ];
        let terms = vec!["known".to_owned(), "shared".to_owned(), "unseen".to_owned()];
        let vocabulary = vocabulary_from_terms(&terms);
        let chunk = emit_tf_chunk(&documents, &vocabulary, terms.len(), &defaults());
        let mut df = vec![0u64; terms.len()];
        for feature in chunk.indices {
            df[feature as usize] += 1;
        }
        assert_eq!(df, vec![2, 3, 0]);
    }

    #[test]
    fn fixed_vocabulary_idf_reduces_sparse_indices_across_chunks() {
        let chunks = vec![
            CsrChunk {
                data: vec![1.0, 1.0],
                indices: vec![0, 1],
                indptr: vec![0, 2],
            },
            CsrChunk {
                data: vec![1.0],
                indices: vec![1],
                indptr: vec![0, 1],
            },
        ];
        let options = Options {
            norm: Norm::None,
            smooth_idf: false,
            ..defaults()
        };
        let idf = fixed_vocabulary_idf(&chunks, 3, 2, &options);

        assert_eq!(idf[0], (2.0f64).ln() + 1.0);
        assert_eq!(idf[1], 1.0);
        assert!(idf[2].is_infinite());
    }

    #[test]
    fn fixed_vocabulary_skips_document_frequency_scan_when_idf_is_disabled() {
        let chunks = vec![CsrChunk {
            data: vec![1.0],
            // This deliberately invalid index would panic if the DF scan ran.
            indices: vec![i32::MAX],
            indptr: vec![0, 1],
        }];
        let options = Options {
            use_idf: false,
            ..defaults()
        };
        assert_eq!(
            fixed_vocabulary_idf(&chunks, 2, 1, &options),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn fixed_vocabulary_weighting_combinations_preserve_sorted_sparse_rows() {
        let documents = vec!["banana banana apple".to_owned(), "banana carrot".to_owned()];
        let terms = vec![
            "apple".to_owned(),
            "banana".to_owned(),
            "carrot".to_owned(),
            "unseen".to_owned(),
        ];

        for &binary in &[false, true] {
            for &sublinear_tf in &[false, true] {
                for &use_idf in &[false, true] {
                    for &smooth_idf in &[false, true] {
                        let options = Options {
                            binary,
                            sublinear_tf,
                            use_idf,
                            smooth_idf,
                            norm: Norm::None,
                            ..defaults()
                        };
                        let output =
                            fit(&documents, options, Some(terms.clone()), 1.0, 2.0, None).unwrap();
                        assert_eq!(output.indices, vec![0, 1, 1, 2]);
                        assert_eq!(output.indptr, vec![0, 2, 4]);
                        assert_eq!(output.model.n_features(), 4);
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_vocabulary_document_frequency_is_correct_across_chunks() {
        let n_docs = CHUNK_DOCS_MIN * 2;
        assert!(
            target_chunk_count(n_docs) >= 2,
            "test needs multiple chunks"
        );
        let documents: Vec<String> = (0..n_docs)
            .map(|index| {
                if index % 2 == 0 {
                    "alpha alpha shared".to_owned()
                } else {
                    "beta shared shared".to_owned()
                }
            })
            .collect();
        let terms = vec![
            "alpha".to_owned(),
            "beta".to_owned(),
            "shared".to_owned(),
            "unseen".to_owned(),
        ];
        let output = fit(
            &documents,
            Options {
                norm: Norm::None,
                smooth_idf: false,
                ..defaults()
            },
            Some(terms),
            1.0,
            documents.len() as f64,
            None,
        )
        .unwrap();

        let document_count = documents.len() as f64;
        let alpha_df = documents.len().div_ceil(2) as f64;
        let beta_df = (documents.len() / 2) as f64;
        assert!((output.model.idf()[0] - ((document_count / alpha_df).ln() + 1.0)).abs() < 1e-12);
        assert!((output.model.idf()[1] - ((document_count / beta_df).ln() + 1.0)).abs() < 1e-12);
        assert_eq!(output.model.idf()[2], 1.0);
        assert!(output.model.idf()[3].is_infinite());
    }

    #[test]
    fn chunk_count_respects_threads_and_min_docs() {
        assert_eq!(chunk_size(1), 1);
        assert_eq!(target_chunk_count(1), 1);
        // Below CHUNK_DOCS_MIN → a single chunk regardless of thread count.
        assert_eq!(target_chunk_count(CHUNK_DOCS_MIN - 1), 1);
        assert_eq!(chunk_size(CHUNK_DOCS_MIN - 1), CHUNK_DOCS_MIN - 1);

        let by_threads = worker_threads().max(1) * CHUNKS_PER_THREAD;
        // 10k docs: min-docs cap is 10, so high thread counts cannot over-split.
        assert_eq!(
            target_chunk_count(10_000),
            by_threads.min(10_000 / CHUNK_DOCS_MIN)
        );
        // 1M docs: thread cap binds before the min-docs cap.
        assert_eq!(
            target_chunk_count(1_000_000),
            by_threads.min(1_000_000 / CHUNK_DOCS_MIN)
        );
        assert!(chunk_size(10_000) >= CHUNK_DOCS_MIN);
        assert!(target_chunk_count(100) <= 100);
        assert!(chunk_size(100) * target_chunk_count(100) >= 100);
    }

    #[test]
    fn merge_term_stats_sums_overlapping_terms_across_many_chunks() {
        // Overlapping terms across many chunks so the merge has to sum.
        let terms = ["apple", "banana", "cherry", "date", "elderberry", "fig"];
        let chunk_count = 200;
        let mut chunk_stats = Vec::with_capacity(chunk_count);
        let mut expected: std::collections::BTreeMap<String, TermStats> = Default::default();

        for chunk in 0..chunk_count {
            let mut map: HashMap<String, TermStats> = HashMap::new();
            for (index, term) in terms.iter().enumerate() {
                // Vary participation so different terms accrue different totals.
                if (chunk + index) % 3 != 0 {
                    let stats = TermStats {
                        document_frequency: 1 + index as u64,
                        term_frequency: 2 + (chunk as u64 % 5),
                        first_occurrence: DiscoveryOrder {
                            chunk,
                            position: index,
                        },
                        last_document: usize::MAX,
                    };
                    map.insert((*term).to_owned(), stats);
                    let entry = expected.entry((*term).to_owned()).or_default();
                    entry.document_frequency += stats.document_frequency;
                    entry.term_frequency += stats.term_frequency;
                    entry.first_occurrence = entry.first_occurrence.min(stats.first_occurrence);
                }
            }
            chunk_stats.push(map);
        }

        let merged = merge_term_stats(chunk_stats);
        assert_eq!(merged.len(), expected.len());
        for (term, stats) in expected {
            let got = merged
                .get(term.as_str())
                .unwrap_or_else(|| panic!("missing term {term}"));
            assert_eq!(
                got.document_frequency, stats.document_frequency,
                "df {term}"
            );
            assert_eq!(got.term_frequency, stats.term_frequency, "tf {term}");
            assert_eq!(
                got.first_occurrence, stats.first_occurrence,
                "first occurrence {term}"
            );
        }
    }

    #[test]
    fn merge_term_stats_handles_empty_and_single_chunk() {
        assert!(merge_term_stats(Vec::new()).is_empty());

        let mut single: HashMap<String, TermStats> = HashMap::new();
        single.insert(
            "solo".to_owned(),
            TermStats {
                document_frequency: 2,
                term_frequency: 5,
                first_occurrence: DiscoveryOrder {
                    chunk: 0,
                    position: 3,
                },
                last_document: usize::MAX,
            },
        );
        let merged = merge_term_stats(vec![single]);
        assert_eq!(merged.len(), 1);
        let stats = merged.get("solo").unwrap();
        assert_eq!(stats.document_frequency, 2);
        assert_eq!(stats.term_frequency, 5);
        assert_eq!(
            stats.first_occurrence,
            DiscoveryOrder {
                chunk: 0,
                position: 3
            }
        );
    }

    fn reference_counts(document: &str, options: &Options) -> HashMap<String, u64> {
        let mut counts: HashMap<String, u64> = HashMap::new();
        if options.lowercase {
            let lowered = document.to_ascii_lowercase();
            fill_counts_from_ascii_text(&lowered, options, &mut counts);
        } else {
            fill_counts_from_ascii_text(document, options, &mut counts);
        }
        counts
    }

    fn assert_counts_match(document: &str, options: &Options) {
        let mut got: HashMap<String, u64> = HashMap::new();
        count_document_into(document, options, &mut got);
        let expected = reference_counts(document, options);
        assert_eq!(got, expected, "document {document:?} options {options:?}");
    }

    fn tokenizer_option_sets() -> Vec<Options> {
        let mut sets = Vec::new();
        for &lowercase in &[true, false] {
            for &(nmin, nmax) in &[(1usize, 1usize), (1, 2), (2, 3)] {
                sets.push(Options {
                    nmin,
                    nmax,
                    lowercase,
                    ..defaults()
                });
            }
        }
        sets
    }

    #[test]
    fn ascii_tokenizer_matches_reference_on_curated_corpus() {
        let corpus = [
            "the quick brown fox jumps over the lazy dog",
            "The Quick BROWN Fox",
            "HELLO, WORLD! hello.",
            "MIXED123 under_score X_Y a b cc",
            "",
            "a",
            "42 4 seven_eight under_score",
            "  spaced   OUT   text  ",
        ];
        for document in corpus {
            for options in tokenizer_option_sets() {
                assert_counts_match(document, &options);
            }
        }
    }

    #[test]
    fn ascii_tokenizer_matches_reference_on_random_documents() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let alphabet: Vec<char> = "abcdeXYZ0129_ \t.,!?-".chars().collect();
        let option_sets = tokenizer_option_sets();
        let mut rng = StdRng::seed_from_u64(0x05EE_D1DF);

        for _ in 0..3000 {
            let length = rng.random_range(0..24);
            let document: String = (0..length)
                .map(|_| alphabet[rng.random_range(0..alphabet.len())])
                .collect();
            for options in &option_sets {
                assert_counts_match(&document, options);
            }
        }
    }
}
