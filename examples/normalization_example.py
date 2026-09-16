"""
Normalization example using the Rust backend.

Demonstrates row-wise normalization (L2, L1, max), which scales each sample
independently, via both the raw Rust kernels and the public ``stratum`` API.

Run after building the extension:
    cd _rust && maturin develop --release && cd ..
    SKRUB_RUST=1 python examples/normalization_example.py
"""
import os
os.environ.setdefault("SKRUB_RUST", "1")

import numpy as np
from sklearn.preprocessing import normalize as sk_normalize

import stratum
from stratum import _rust_backend as rb

if not rb.HAVE_RUST:
    raise SystemExit(
        "Rust backend not available. Build it first:\n"
        "  cd _rust && maturin develop --release"
    )


def section(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


rng = np.random.default_rng(42)

# ── Row-wise normalizations ───────────────────────────────────────────────────

section("L2 normalization (each row becomes unit vector)")

embeddings = rng.standard_normal((5, 4)).astype(np.float32)
print("Input:\n", embeddings.round(3))

l2_out = rb.normalize_l2(embeddings)
print("Output:\n", l2_out.round(4))
print("Row L2 norms:", np.linalg.norm(l2_out, axis=1).round(6))  # all ~1.0


section("L1 normalization (each row sums to 1 in absolute value)")

counts = rng.integers(0, 10, size=(4, 6)).astype(np.float32)
print("Input (pseudo-counts):\n", counts)

l1_out = rb.normalize_l1(counts)
print("Output:\n", l1_out.round(4))
print("Row L1 norms:", np.abs(l1_out).sum(axis=1).round(6))  # all ~1.0


section("Max normalization (each row divided by its absolute max)")

data = np.array([[2.0, -6.0, 3.0], [10.0, 1.0, 5.0]], dtype=np.float32)
print("Input:\n", data)

max_out = rb.normalize_max(data)
print("Output:\n", max_out.round(4))
print("Row max abs:", np.abs(max_out).max(axis=1))  # all 1.0


# ── Public API ──────────────────────────────────────────────

section("stratum.normalize / stratum.Normalizer (drop-in for sklearn)")

X = rng.standard_normal((6, 3)).astype(np.float32)

# Matches sklearn.preprocessing.normalize elementwise on the supported subset.
print("stratum.normalize:\n", stratum.normalize(X, norm="l2").round(4))
print("sklearn.normalize:\n", sk_normalize(X, norm="l2").round(4))
assert np.allclose(stratum.normalize(X, norm="l2"), sk_normalize(X, norm="l2"),
                   rtol=1e-5, atol=1e-6)

# The transformer is stateless, so fit() is a no-op and transform() does the work.
print("Normalizer().fit_transform:\n",
      stratum.Normalizer(norm="l1").fit_transform(X).round(4))


# ── In-place kernels ────────────────────────────────────────

section("In-place variants (no output allocation)")

buf = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
print("Before:\n", buf)
rb.normalize_l2_inplace(buf)          # writes through the caller's buffer
print("After: \n", buf.round(4))
print("Row L2 norms:", np.linalg.norm(buf, axis=1).round(6))


# ── Edge cases ────────────────────────────────────────────────────────────────

section("Edge cases")

# All-zero row: normalization leaves it untouched
zero_row = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
out = rb.normalize_l2(zero_row)
assert np.all(out[0] == 0), "Zero row should stay zero"
print("Zero row handled correctly (stays zero).")

# Zero row is left untouched by every norm, in place as well as by copy
zero_row_ip = zero_row.copy()
rb.normalize_l1_inplace(zero_row_ip)
assert np.all(zero_row_ip[0] == 0), "Zero row should stay zero in place too"
print("Zero row handled correctly in place (stays zero).")

# Unsupported input (float64) falls back to sklearn rather than converting
out64 = stratum.normalize(X.astype(np.float64), norm="l2")
assert out64.dtype == np.float64, "Fallback should preserve sklearn's dtype"
print("float64 input fell back to sklearn correctly (dtype preserved).")

print("\nAll checks passed.")
