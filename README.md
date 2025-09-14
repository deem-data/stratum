# Stratum

**Stratum** is an experimental fork of [skrub](https://github.com/skrub-data/skrub) with a **Rust backend** for compute-heavy operations, while keeping the high-level Python API intact.

---

## Goals

- Provide an **opt-in Rust backend** for performance-critical parts of skrub.
- Preserve skrub’s **Python API**; developers can flip an environment variable to enable Rust.
- Build cross-platform wheels (Windows / Linux / macOS) so users can install without a Rust toolchain.

---

## Installation

For now, you need to build from source.

Requirements:
- Python **3.10+**
- [Rust toolchain](https://rustup.rs/) (nightly not required; stable is fine)
- [maturin](https://www.maturin.rs/) (`pip install maturin`)

---

## Usage

At runtime, enable the Rust backend by setting an environment variable:

```bash
export SKRUB_RUST=1     # Linux / macOS
setx SKRUB_RUST 1       # Windows PowerShell
```

#### Test Code

```Python

import os
import pandas as pd
# os.environ['SKRUB_RUST'] = '1'

from skrub import StringEncoder
s = pd.Series(["foo", "bar", None, "lorem ipsum dolor"]) # nulls handled upstream
enc = StringEncoder(vectorizer='hashing', analyzer='char', ngram_range=(3,5), n_components=2)
Z = enc.fit_transform(s)
print(type(Z), Z.shape)
assert Z.shape[0] == len(s)
```
---

## Repository Layout

```bash
stratum/
├─ pyproject.toml           # Python + Rust build config (maturin)
├─ skrub/                   # Python sources (fork of skrub)
│   ├─ __init__.py
│   ├─ _string_encoder.py
│   ├─ _rust_backend.py     # shim: checks env, exposes USE_RUST/HAVE_RUST
│   └─ …
└─ _rust/                   # Rust crate (PyO3 extension)
   ├─ Cargo.toml
   └─ src/lib.rs            # defines #[pymodule] fn _rust_backend_native
```
---

## Developer Instructions

#### Local Dev Install (Editable)

```bash
maturin develop				# Debug mode
maturin develop --release	# Optimized dev build
```

#### Building Wheels

This produces redistributable `.whl` files under `dist/`.

```bash
maturin build --release -o dist --interpreter python3.10 --compatibility linux		# Linux/macOS
maturin build --release -o dist		# Windows
```
Then install with:

```bash
pip install ./dist/stratum-*.whl
```

---

## License
BSD-3-Clause (inherited from skrub).






