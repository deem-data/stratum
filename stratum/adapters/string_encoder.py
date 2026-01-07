from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix

from skrub import StringEncoder as _SE  # base class from vanilla skrub
from .. import _rust_backend as rb
from .. _config import get_config
from skrub._string_encoder import scaling_factor
from skrub import _dataframe as sbd

# File-internal config flags
_DEBUG_INFO = False
_FD_PATH = True

def _rust_supported_subset(enc: _SE) ->tuple[bool, str]:
    # Supports vectorizer="hashing/tfidf" with char/char_wb analyzer, no stopwords.
    if getattr(enc, "vectorizer", None) not in ("hashing", "tfidf"):
        return False, "vectorizer not in {hashing, tfidf}"
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

def _prep_strings_transform(X):
    try: #from skrub's _string_encoder.py
        from skrub._to_str import ToStr
        to_str = ToStr(convert_category=True)
        # FIXME: Do we need to store the to_str from fit for transform?
        X_filled = to_str.transform(X)
        X_filled = sbd.fill_nulls(X_filled, "")
        return rb._to_list(X_filled)
    except Exception: #fallback
        return _clean_strings(rb._to_list(X))

# Create a subclass of StringEncoder
"""Drop-in StringEncoder that prefers the Rust fastpath where supported."""
class RustyStringEncoder(_SE):
    def __init__(self, vectorizer="tfidf", analyzer="char_wb", ngram_range=(3,4), n_components=30, **kwargs):
        super().__init__(vectorizer=vectorizer, analyzer=analyzer, ngram_range=ngram_range, n_components=n_components, **kwargs)
        self._rust_state_ = None

    def fit_transform(self, X, y=None):
        # Check supported parameters
        if not _rust_supported_subset(_SE):
            return super().fit_transform(X, y)
        # Check kill-switch and feature flag at call time
        rc = get_config()
        if not (rc["allow_patch"] and rc["rust_backend"] and rb.HAVE_RUST):
            return super().fit_transform(X, y)

        # Prepare inputs for Rust
        strings = _prep_strings(X)
        ngram_min, ngram_max = self.ngram_range
        analyzer = self.analyzer    #"char" or "char_wb"
        n_features = 1 << 20    #TODO: expose via parameter

        # Call Rust function. Returns CSR parts + idf vector for "hashing"
        t0 = rb.start_timing()
        if _DEBUG_INFO: print("INFO: Delegating StringEncoder to Rust backend") #TODO: proper logging
        try:
            if self.vectorizer == "hashing":
                data, indices, indptr, n_rows, n_cols, idf = rb.hashing_tfidf_fit(
                    strings, analyzer, int(ngram_min), int(ngram_max), int(n_features)
                )
            if self.vectorizer == "tfidf":
                tfidf_model_id, data, indices, indptr, n_rows, n_cols = rb.tfidf_fit(
                    strings, analyzer, int(ngram_min), int(ngram_max)
                )
        except Exception as e:
            # Never fail, just fallback
            print(f"WARNING: Rust tfidf_fit_csr failed, falling back. Error: {e}")
            return super().fit_transform(X, y)
        if self.vectorizer == "hashing": rb.print_timing("hashing_tfidf_fit", t0)
        if self.vectorizer == "tfidf": rb.print_timing("tfidf_fit", t0)

        # FD/TruncatedSVD path in Rust (randomized SVD)
        t0 = rb.start_timing()
        try:
            if _FD_PATH: #Frequent Directions path
                if _DEBUG_INFO: print("INFO: Taking FD path in Rust")
                svd_model_id, Z = rb.fd_fit(data, indices, indptr, int(n_rows),
                                            int(n_cols), int(self.n_components), 16, self.random_state)
            else: #TruncatedSVD path
                if _DEBUG_INFO: print("INFO: Taking TruncatedSVD path in Rust")
                svd_model_id, Z = rb.truncated_svd_fit(data, indices, indptr, int(n_rows),
                                            int(n_cols), int(self.n_components), self.random_state)
        except Exception as e:
            print(f"WARNING: Rust truncated_svd_from_csr failed, falling back. Error: {e}")
            return super().fit_transform(X, y)
        result = np.asarray(Z, dtype=np.float32, order="C")
        if _FD_PATH: rb.print_timing("fd_fit", t0)
        else: rb.print_timing("truncated_svd_fit", t0)

        # Pad to exactly self.n_components
        if result.shape[1] < self.n_components:
            padded = np.zeros((result.shape[0], self.n_components), dtype=np.float32)
            padded[:, : result.shape[1]] = result
            result = padded

        # Block normalize as original
        self.scaling_factor_ = scaling_factor(result)
        result /= self.scaling_factor_

        # Maintain states for transform
        self._rust_state_ = {
            "backend": "rust",
            "n_features": n_features,
            "idf": None,
            "tfidf_model_id": None,
            "svd_model_id": svd_model_id
        }
        if self.vectorizer == "hashing":
            self._rust_state_["idf"] = idf
        if self.vectorizer == "tfidf":
            self._rust_state_["tfidf_model_id"] = tfidf_model_id

        # Mark fitted attributes
        self.n_components_ = result.shape[1]
        self.input_name_ = sbd.name(X) or "string_enc"
        self.all_outputs_ = self.get_feature_names_out()
        return self._post_process(X, result)

    def transform(self, X):
        # Check kill-switch and feature flag at call time
        rc = get_config()
        if not (rc["allow_patch"] and rc["rust_backend"] and rb.HAVE_RUST):
            return super().transform(X)
        # Check if we have stored state from fit_transform
        if not hasattr(self, "_rust_state_") or self._rust_state_ is None:
            return super().transform(X)
        if not hasattr(self, "scaling_factor_") or self.scaling_factor_ is None:
            return super().transform(X)
        if not hasattr(self, "n_components_") or self.n_components_ is None:
            return super().transform(X)

        # Prepare inputs for Rust
        strings = _prep_strings_transform(X) #FIXME
        ngram_min, ngram_max = self.ngram_range
        analyzer = self.analyzer    #"char" or "char_wb"
        n_features = self._rust_state_["n_features"]
        if self.vectorizer == "hashing":
            idf = self._rust_state_["idf"]  #Get pre-computed IDF from fit
        model_id = self._rust_state_["tfidf_model_id"] #Get model id stored in the Rust space
        svd_model_id = self._rust_state_["svd_model_id"] #Get the tsvd/fd model id stored in the Rust space

        # Call Rust function
        t0 = rb.start_timing()
        if _DEBUG_INFO: print("INFO: Delegating StringEncoder transform to Rust backend") #TODO: proper logging
        try:
            if self.vectorizer == "hashing":
                data, indices, indptr, n_rows, n_cols = rb.hashing_tfidf_transform(
                    strings, analyzer, int(ngram_min), int(ngram_max), int(n_features), idf
                )
            if self.vectorizer == "tfidf":
                data, indices, indptr, n_rows, n_cols = rb.tfidf_transform(model_id, strings)
        except Exception as e:
            # Never fail, just fallback
            print(f"WARNING: Rust tfidf_transform_csr failed, falling back. Error: {e}")
            return super().transform(X)
        if self.vectorizer == "hashing": rb.print_timing("hashing_tfidf_transform", t0)
        if self.vectorizer == "tfidf": rb.print_timing("tfidf_transform", t0)

        # FD/TruncatedSVD path in Rust (randomized SVD)
        t0 = rb.start_timing()
        try:
            if _FD_PATH:
                if _DEBUG_INFO: print("INFO: Taking FD path in Rust")
                Z = rb.fd_transform(svd_model_id, data, indices, indptr, int(n_rows), int(n_cols))
            else:
                if _DEBUG_INFO: print("INFO: Taking TruncatedSVD path in Rust")
                Z = rb.truncated_svd_transform(svd_model_id, data, indices, indptr, int(n_rows), int(n_cols))
        except Exception as e:
            print(f"WARNING: Rust truncated_svd_from_csr failed, falling back. Error: {e}")
            return super().transform(X)
        result = np.asarray(Z, dtype=np.float32, order="C")
        if _FD_PATH: rb.print_timing("fd_transform", t0)
        else: rb.print_timing("truncated_svd_transform", t0)

        # Ensure fixed width output
        if result.shape[1] < self.n_components_:
            padded = np.zeros((result.shape[0], self.n_components_), dtype=np.float32)
            padded[:, : result.shape[1]] = result
            result = padded

        # Block normalize using stored scaling_factor_ from fit
        result /= self.scaling_factor_
        return self._post_process(X, result)