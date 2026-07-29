use crate::tokenize;
use crate::util::{print_timing, start_timing};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

fn sort_vocab_lexi_inplace(vocabulary: &mut FxHashMap<String, usize>, j_indices: &mut Vec<usize>) {
    let n = vocabulary.len();
    let mut sorted: Vec<(&String, usize)> =
        vocabulary.iter().map(|(term, &old)| (term, old)).collect();
    sorted.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut remap = vec![0usize; n];
    for (new_index, &(_, old_index)) in sorted.iter().enumerate() {
        remap[old_index] = new_index;
    }
    for idx in vocabulary.values_mut() {
        *idx = remap[*idx];
    }
    for col in j_indices.iter_mut() {
        *col = remap[*col];
    }
}


struct Partial {
    vocab: FxHashMap<String, usize>,
    values: Vec<i32>,
    j_indices: Vec<usize>,
    indptr: Vec<usize>,
}
pub fn compute_count_vectorizer_fit(
    corpus: Vec<String>,
    stopwords: FxHashSet<String>,
    n_chunks: usize,
) -> (FxHashMap<String, usize>, Vec<i32>, Vec<usize>, Vec<usize>) {
    let chunk_size = (corpus.len() / n_chunks).max(1);
    // for reduction
    let identity = || Partial {
        vocab: FxHashMap::default(),
        values: vec![],
        j_indices: vec![],
        indptr: vec![0],
    };
    let start_multithreading = start_timing();
    let processed_chunks: Vec<Partial> = corpus
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(idx, chunk)| {
            let chunk_start = start_timing();
            let chunksize = chunk.len();
            let mut vocab: FxHashMap<String, usize> = FxHashMap::default();
            let mut values: Vec<i32> = Vec::new();
            let mut j_indices: Vec<usize> = Vec::new();
            let mut indptr: Vec<usize> = Vec::with_capacity(chunk.len() + 1);
            indptr.push(0);

            let mut feature_counter: FxHashMap<usize, usize> = FxHashMap::default();
            let mut token_buf: Vec<&str> = Vec::new();

            for text in chunk.iter() {
                feature_counter.clear();
                token_buf.clear();
                tokenize::word_tokens(text, &mut token_buf);
                for tok in token_buf.drain(..) {
                    // to lowercase transforms to String from str
                    let token = tok.to_lowercase();
                    if stopwords.contains(&token) {
                        continue;
                    };
                    let idx = match vocab.get(&token).copied() {
                        Some(i) => i,
                        None => {
                            let i = vocab.len();
                            vocab.insert(token, i);
                            i
                        }
                    };

                    // we need to own the number that our vocabulary returns
                    *feature_counter.entry(idx).or_insert(0) += 1;
                }
                for (&col, &count) in feature_counter.iter() {
                    j_indices.push(col);
                    values.push(count as i32);
                }
                indptr.push(j_indices.len());
            }
            print_timing(
                &format!("count_vectorize_fit chunk_{idx}_{chunksize}"),
                chunk_start,
            );
            Partial {
                vocab,
                values,
                j_indices,
                indptr,
            }
        })
        .collect();
    print_timing("count_vectorize_fit map", start_multithreading);

    let start_reduce = start_timing();
    let mut result = processed_chunks
        .into_par_iter()
        .reduce(identity, |mut a, b| {
            let mut remap = vec![0 as usize; b.vocab.len()];
            for (token, &local) in b.vocab.iter() {
                let next = a.vocab.len();
                let global = *a.vocab.entry(token.clone()).or_insert(next);
                remap[local] = global;
            }
            let offset = a.j_indices.len();
            a.values.extend(b.values);
            a.j_indices.extend(b.j_indices.iter().map(|&c| remap[c]));
            a.indptr.extend(b.indptr[1..].iter().map(|&p| p + offset));
            a
        });
    print_timing("count_vectorize_fit reduce", start_reduce);

    let start_sorting = start_timing();
    sort_vocab_lexi_inplace(&mut result.vocab, &mut result.j_indices);
    print_timing("count_vectorize_fit sorting", start_sorting);

    (result.vocab, result.values, result.j_indices, result.indptr)
}

pub fn compute_count_vectorizer_transform(
    corpus: &[String],
    vocabulary: &FxHashMap<String, usize>,
    stopwords: &FxHashSet<String>,
    n_chunks: usize,
) -> (Vec<i32>, Vec<usize>, Vec<usize>) {
    let n_rows = corpus.len();
    let chunk_size = (n_rows / n_chunks).max(1);
    let start_multithreading = start_timing();

    let fragments: Vec<(Vec<i32>, Vec<usize>, Vec<usize>)> = corpus
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut data: Vec<i32> = Vec::new();
            let mut indices: Vec<usize> = Vec::new();
            let mut indptr: Vec<usize> = Vec::with_capacity(chunk.len());
            let mut feature_counter: FxHashMap<usize, i32> = FxHashMap::default();
            let mut token_buf: Vec<&str> = Vec::new();
            for text in chunk.iter() {
                feature_counter.clear(); // reuse across rows: keeps capacity, one alloc per chunk
                token_buf.clear();
                tokenize::word_tokens(text, &mut token_buf);
                for tok in token_buf.drain(..) {
                    let token = tok.to_lowercase();
                    if stopwords.contains(&token) {
                        continue;
                    }
                    if let Some(&col) = vocabulary.get(&token) {
                        *feature_counter.entry(col).or_insert(0) += 1;
                    }
                }
                for (&col, &count) in feature_counter.iter() {
                    indices.push(col);
                    data.push(count);
                }
                indptr.push(indices.len());
            }

            (data, indices, indptr)
        })
        .collect();
    print_timing("count_vectorize_transform map", start_multithreading);

    let start_reduce = start_timing();
    // reduce: concatenate fragments in order, shifting each indptr by the running nnz
    let total_nnz: usize = fragments.iter().map(|(d, _, _)| d.len()).sum();
    let mut data = Vec::with_capacity(total_nnz);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut indptr = Vec::with_capacity(n_rows + 1);
    indptr.push(0);
    for (fd, fi, fp) in fragments {
        let offset = data.len();
        data.extend(fd);
        indices.extend(fi);
        indptr.extend(fp.iter().map(|&p| p + offset));
    }
    print_timing("count_vectorize_transform reduce", start_reduce);

    (data, indices, indptr)
}
#[pyfunction]
#[pyo3(signature = (corpus, vocabulary, stopwords, n_chunks = 1))]
pub fn count_vectorize_transform(
    py: Python<'_>,
    corpus: Vec<String>,
    vocabulary: FxHashMap<String, usize>,
    stopwords: FxHashSet<String>,
    n_chunks: usize,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    if n_chunks == 0 {
        return Err(PyValueError::new_err("n_chunks must be >= 1"));
    }

    let (data, indices, indptr) = py.detach(|| {
        compute_count_vectorizer_transform(&corpus, &vocabulary, &stopwords, n_chunks)
    });

    let indices: Vec<i64> = indices.into_iter().map(|x| x as i64).collect();
    let indptr: Vec<i64> = indptr.into_iter().map(|x| x as i64).collect();

    let data = Py::from(data.into_pyarray(py).to_owned());
    let indices = Py::from(indices.into_pyarray(py).to_owned());
    let indptr = Py::from(indptr.into_pyarray(py).to_owned());
    Ok((data, indices, indptr))
}

#[pyfunction]
#[pyo3(signature = (corpus, stopwords, n_chunks = 1))]
pub fn count_vectorize_fit(
    corpus: Vec<String>,
    stopwords: FxHashSet<String>,
    n_chunks: usize,
) -> PyResult<FxHashMap<String, usize>> {
    if n_chunks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_chunks must be >= 1",
        ));
    }

    let (vocabulary, _, _, _) = compute_count_vectorizer_fit(corpus, stopwords, n_chunks);

    Ok(vocabulary)
}

#[pyfunction]
#[pyo3(signature = (corpus, stopwords, n_chunks = 1))]
pub fn count_vectorize_fit_transform(
    py: Python<'_>,
    corpus: Vec<String>,
    stopwords: FxHashSet<String>,
    n_chunks: usize,
) -> PyResult<(
    FxHashMap<String, usize>,
    Py<PyArray1<i32>>,
    Py<PyArray1<usize>>,
    Py<PyArray1<usize>>,
)> {
    let start = start_timing();
    if n_chunks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_chunks must be >= 1",
        ));
    }

    let (vocabulary, data, j_indices, indptr) =
        compute_count_vectorizer_fit(corpus, stopwords, n_chunks);

    let data = data.into_pyarray(py).unbind();
    let indices = j_indices.into_pyarray(py).unbind();
    let indptr = indptr.into_pyarray(py).unbind();
    print_timing("count_vectorize_fit_transform total", start);

    Ok((vocabulary, data, indices, indptr))
}
