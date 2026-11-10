use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Analyzer {
    Char,
    Char_wb,
}

pub fn parse_analyzer(s: &str) -> Result<Analyzer> {
    match s {
        "char" => Ok(Analyzer::Char),
        "char_wb" => Ok(Analyzer::Char_wb),
        _ => bail!("Unsupported Analyzer: {}", s),
    }
}

pub fn char_ngrams<'a>(s: &'a str, nmin: usize, nmax: usize, buf: &mut Vec<&'a str>) {
    // Collect byte offsets at char boundaries (safe UTF-8 slicing)
    let mut idx: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
    idx.push(s.len());

    // Gather all n-grams for all n between nmin and nmax
    for n in nmin..=nmax {
        if n == 0 || idx.len() < n + 1 {
            //coherence check
            continue;
        }
        for start in 0..=idx.len().saturating_sub(n + 1) {
            let end = start + n;
            let i = idx[start];
            let j = idx[end];
            buf.push(&s[i..j]); // slice is valid UTF-8 by construction
        }
    }
}

pub fn char_wb_ngrams(str: &str, nmin: usize, nmax: usize, buf: &mut Vec<String>) {
    // Pad with spaces at word boundaries and then extract like char_ngrams
    // TODO: Implement proper word-boundary padding
    let padded = format!(" {} ", str);
    for n in nmin..=nmax {
        if n == 0 || n > padded.len() {
            continue;
        }
        for i in 0..=padded.len().saturating_sub(n) {
            let j = i + n;
            buf.push(padded[i..j].to_string()); //FIXME (perf): allocations
        }
    }
}

/// sklearn default `\w`: alphanumeric or underscore.
#[inline(always)]
pub fn is_word_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Tokenize like sklearn's default `token_pattern=r"(?u)\b\w\w+\b"`.
/// Tokens borrow from `text` (no per-token allocation).
pub fn word_tokens<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let mut start = None;
    let mut char_count = 0usize;

    for (offset, ch) in text.char_indices() {
        if is_word_char(ch) {
            if start.is_none() {
                start = Some(offset);
                char_count = 0;
            }
            char_count += 1;
        } else if let Some(token_start) = start.take() {
            if char_count >= 2 {
                out.push(&text[token_start..offset]);
            }
        }
    }

    if let Some(token_start) = start {
        if char_count >= 2 {
            out.push(&text[token_start..]);
        }
    }
}
