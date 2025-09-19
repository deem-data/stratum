import os

USE_RUST = os.environ.get("SKRUB_RUST", "0") == "1"
try:
    #import skrub_rust
    from . import _rust_backend_native as native
    HAVE_RUST = True
except Exception as e:
    native = False
    HAVE_RUST = False
    _import_error = e

def _to_list(col):
    # pandas or polars series -> list (best-effort, minimal overhead)
    try:
        return col.tolist()
    except Exception:
        pass
    try:
        return col.to_list()
    except Exception:
        pass
    return list(col)

# Re-export compiled functions if present
hashing_tfidf_csr = getattr(native, "hashing_tfidf_csr", None)
fd_embedding = getattr(native, "fd_embed_from_csr", None)