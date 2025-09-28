from __future__ import annotations
import numpy as np

from skrub import StringEncoder as _SE  # base class from vanilla skrub
from .. import _rust_backend as rb
from .. config import get_config
from skrub._string_encoder import scaling_factor
from skrub import _dataframe as sbd

def _rust_supported_subset(enc: _SE) ->tuple[bool, str]:
    # Supports vectorizer="hashing" with char/char_wb analyzer, no stopwords.
    if getattr(enc, "vectorizer", None) != "hashing":
        return False, "vectorizer != hashing"
    if getattr(enc, "stop_words", None) is not None:
        return False, "stop_words not supported yet"
    if getattr(enc, "analyzer", None) not in ("char", "char_wb"):
        return False, "analyzer not in {char, char_wb}"
    ngr = getattr(enc, "ngram_range", (3, 5))
    if not (isinstance(ngr, tuple) and len(ngr) == 2 and 1 <= ngr[0] <= ngr[1]):
        return False, f"invalid ngram_range {ngr!r}"
    return True, ""

def _clean_strings(x_list):
    # Fill null/NaN → "", and coerce to str
    out = []
    for v in x_list:
        if v is None:
            out.append("")
            continue
        try:
            # Handle NaN (float)
            if isinstance(v, float) and np.isnan(v):
                out.append("")
                continue
        except Exception:
            pass
        out.append("" if v is None else str(v))
    return out

def _prep_strings(X):
    try: #from skrub's _string_encoder.py
        from skrub._to_str import ToStr
        to_str = ToStr(convert_category=True)
        X_filled = to_str.fit_transform(X)
        X_filled = sbd.fill_nulls(X_filled, "")
        return rb._to_list(X_filled)
    except Exception: #fallback
        return _clean_strings(rb._to_list(X))

# Create a subclass of StringEncoder
class RustyStringEncoder(_SE):
    """Drop-in StringEncoder that prefers the Rust fastpath where supported."""

    def fit_transform(self, X, y=None):
        # Check kill-switch and feature flag at call time
        rc = get_config()
        if not (rc["allow_patch"] and rc["rust_backend"] and rb.HAVE_RUST):
            return super().fit_transform(X, y)
        # Check if the rust modules are available
        if getattr(rb, "hashing_tfidf_csr", None) is None or getattr(rb, "fd_embedding", None) is None:
            return super().fit_transform(X, y)

        # Prepare inputs for Rust
        strings = _prep_strings(X)
        ngram_min, ngram_max = self.ngram_range
        analyzer = self.analyzer    #"char" or "char_wb"
        n_features = 1 << 20    #TODO: expose via parameter

        # Call Rust function. Returns CSR parts + idf vector (float32)
        t0 = rb.start_timing()
        print("INFO: Delegating StringEncoder to Rust backend") #TODO: proper logging
        try:
            data, indices, indptr, n_rows, n_cols, idf = rb.hashing_tfidf_csr(
                strings, analyzer, int(ngram_min), int(ngram_max), int(n_features)
            )
        except Exception as e:
            # Never fail, just fallback
            print(f"WARNING: Rust hashing_tfidf_csr failed, falling back. Error: {e}")
            return super().fit_transform(X, y)
        rb.print_timing("hashing_tfidf_csr", t0)

        # Maintain states for transform (in future)
        self._rust_state_ = {
            "backend": "rust",
            "path": "hashing->tfidf",
            "n_features": n_features,
            "idf": idf,
        }

        # Frequent Directions (FD) path in Rust.
        # TODO: Write a truncated SVD in Rust for sklearn equivalent result
        print("INFO: Taking FD path in Rust")
        t0 = rb.start_timing()
        try:
            Z = rb.fd_embedding(data, indices, indptr, int(n_rows), int(n_cols),
                                int(self.n_components), 16, self.random_state)
        except Exception as e:
            print(f"WARNING: Rust fd_embedding failed, falling back. Error: {e}")
            return super().fit_transform(X, y)
        result = np.asarray(Z, dtype=np.float32, order="C")
        rb.print_timing("fd_embedding", t0)

        # Block normalize as original
        self.scaling_factor_ = scaling_factor(result)
        result /= self.scaling_factor_
        self.n_components_ = result.shape[1]
        self.input_name_ = sbd.name(X) or "string_enc"
        self.all_outputs_ = self.get_feature_names_out()
        return self._post_process(X, result)