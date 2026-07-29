use anyhow::{bail, Result};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Analyzer {
    Char,
    Char_wb,
}

#[allow(dead_code)]
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

/// sklearn default `\w`: alphanumeric or underscore.
#[cfg(test)]
#[inline(always)]
pub fn is_word_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Tokenize like sklearn's default `token_pattern=r"(?u)\b\w\w+\b"`.
/// Tokens borrow from `text` (no per-token allocation).
#[cfg(test)]
pub fn word_tokens<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let mut start: Option<usize> = None;
    let mut nchars: usize = 0;
    for (i, c) in text.char_indices() {
        if is_word_char(c) {
            if start.is_none() {
                start = Some(i);
                nchars = 0;
            }
            nchars += 1;
        } else if let Some(s) = start.take() {
            if nchars >= 2 {
                out.push(&text[s..i]);
            }
        }
    }
    if let Some(s) = start {
        if nchars >= 2 {
            out.push(&text[s..]);
        }
    }
}

/// Tokenize ASCII text like sklearn's default
/// `token_pattern=r"(?u)\b\w\w+\b"`.
///
/// Callers that have already established ASCII input can avoid UTF-8
/// decoding and Unicode character classification with this path.
#[inline]
pub fn word_tokens_ascii<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    for_each_word_token_ascii(text, |token| out.push(token));
}

/// Visit tokens from [`word_tokens_ascii`] without first materializing a token
/// vector. This is the common unigram-only fast path.
#[inline]
pub fn for_each_word_token_ascii<'a, F>(text: &'a str, mut visit: F)
where
    F: FnMut(&'a str),
{
    debug_assert!(text.is_ascii());

    let bytes = text.as_bytes();
    let mut token_start = 0usize;
    let mut in_token = false;

    for (offset, &byte) in bytes.iter().enumerate() {
        let is_word = byte.is_ascii_alphanumeric() || byte == b'_';
        if is_word {
            if !in_token {
                token_start = offset;
                in_token = true;
            }
        } else if in_token {
            if offset - token_start >= 2 {
                visit(&text[token_start..offset]);
            }
            in_token = false;
        }
    }

    if in_token && bytes.len() - token_start >= 2 {
        visit(&text[token_start..]);
    }
}

/// Word n-grams in sklearn order. Unigrams are borrowed; higher-order grams
/// are joined with a shared scratch buffer.
pub fn for_each_word_ngram<F: FnMut(&str)>(tokens: &[&str], nmin: usize, nmax: usize, mut f: F) {
    let ntok = tokens.len();
    let start = nmin.max(1);
    let end = nmax.min(ntok);
    if start > end {
        return;
    }

    let mut buf = String::new();
    for n in start..=end {
        if n == 1 {
            for &t in tokens {
                f(t);
            }
        } else {
            for i in 0..=ntok - n {
                buf.clear();
                for (j, &t) in tokens[i..i + n].iter().enumerate() {
                    if j > 0 {
                        buf.push(' ');
                    }
                    buf.push_str(t);
                }
                f(&buf);
            }
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

#[cfg(test)]
mod tests {
    use super::{for_each_word_ngram, for_each_word_token_ascii, word_tokens, word_tokens_ascii};

    fn collect_ngrams(tokens: &[&str], nmin: usize, nmax: usize) -> Vec<String> {
        let mut ngrams = Vec::new();
        for_each_word_ngram(tokens, nmin, nmax, |ngram| {
            ngrams.push(ngram.to_owned());
        });
        ngrams
    }

    #[test]
    fn default_word_pattern_drops_single_character_tokens() {
        let mut tokens = Vec::new();
        word_tokens("a fox 42 _ x_y café", &mut tokens);
        assert_eq!(tokens, vec!["fox", "42", "x_y", "café"]);
    }

    #[test]
    fn ascii_fast_path_matches_generic_tokenizer() {
        for text in [
            "",
            "a fox 42 _ x_y cafe",
            " punctuation!around,tokens ",
            "UPPER lower MiXeD123",
            "one\ttwo\nthree\r\nfour",
        ] {
            let mut generic = Vec::new();
            word_tokens(text, &mut generic);

            let mut buffered = Vec::new();
            word_tokens_ascii(text, &mut buffered);
            assert_eq!(buffered, generic);

            let mut visited = Vec::new();
            for_each_word_token_ascii(text, |token| visited.push(token));
            assert_eq!(visited, generic);
        }
    }

    #[test]
    fn ascii_fast_path_matches_generic_tokenizer_on_random_documents() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let alphabet = b"abcdeXYZ0129_ \t.,!?-";
        let mut rng = StdRng::seed_from_u64(0x05EE_D1DF);

        for _ in 0..3000 {
            let length = rng.random_range(0..24);
            let document: String = (0..length)
                .map(|_| alphabet[rng.random_range(0..alphabet.len())] as char)
                .collect();

            let mut expected = Vec::new();
            word_tokens(&document, &mut expected);

            let mut buffered = Vec::new();
            word_tokens_ascii(&document, &mut buffered);
            assert_eq!(buffered, expected, "document {document:?}");

            let mut visited = Vec::new();
            for_each_word_token_ascii(&document, |token| visited.push(token));
            assert_eq!(visited, expected, "document {document:?}");
        }
    }

    #[test]
    fn empty_tokens_ignore_unbounded_nmax() {
        assert!(collect_ngrams(&[], 1, usize::MAX).is_empty());
    }

    #[test]
    fn one_token_ignores_unbounded_nmax() {
        assert_eq!(collect_ngrams(&["token"], 1, usize::MAX), vec!["token"]);
    }

    #[test]
    fn nmin_above_token_count_produces_no_ngrams() {
        assert!(collect_ngrams(&["one", "two"], 3, usize::MAX).is_empty());
    }

    #[test]
    fn ordinary_word_ngram_ranges_are_unchanged() {
        let tokens = ["one", "two", "three"];
        assert_eq!(collect_ngrams(&tokens, 1, 1), vec!["one", "two", "three"]);
        assert_eq!(
            collect_ngrams(&tokens, 1, 2),
            vec!["one", "two", "three", "one two", "two three"]
        );
        assert_eq!(
            collect_ngrams(&tokens, 2, 3),
            vec!["one two", "two three", "one two three"]
        );
    }
}
