# Stratum

[![CI](https://github.com/deem-data/stratum/actions/workflows/test.yml/badge.svg)](https://github.com/deem-data/stratum/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/deem-data/stratum/graph/badge.svg?token=QQDTC0RXUN)](https://codecov.io/gh/deem-data/stratum)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Stratum** is an experimental fork of [skrub](https://github.com/skrub-data/skrub) with a **Rust backend** for compute-heavy operations, while keeping the high-level Python API intact.

---

## Goals

- Provide an **opt-in Rust backend** for performance-critical parts of skrub.
- Preserve skrub’s **Python API**; developers can flip a flag to enable Rust.
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

To enable the Rust backend and other related features, import stratum as skrub and enable the Rust backend:

Replace
```Python
import skrub
from skrub import StringEncoder
```
with
```Python
import stratum as skrub
from stratum import StringEncoder
skrub.set_config(rust_backend=True)
```

#### Test Code

```Python

import os
import pandas as pd
import stratum as skrub
from stratum import StringEncoder
skrub.set_config(rust_backend=True)

#skrub.set_config(debug_timing=True, num_threads=0) # other rust flags
s = pd.Series(["foo", "bar", None, "lorem ipsum dolor"])
enc = StringEncoder(vectorizer='hashing', analyzer='char', ngram_range=(3,5), n_components=2)
Z = enc.fit_transform(s)
print(type(Z), Z.shape)
assert Z.shape[0] == len(s)
```
---

## Repository Layout

```bash  
stratum/
├─ pyproject.toml             # Python + Rust build config (maturin)
├─ stratum/
│ ├─ __init__.py              # Façade over skrub
│ ├─ config.py                # set_config/get_config + env sync
│ ├─ _rust_backend.py         # Python <-> Rust shim (re-exports native fns)
│ ├─ adapters/                # Public API (dispatch to Rust or fallback to skrub)
│ │ └─ string_encoder.py      # RustyStringEncoder (subclass)
│ └─ _rust_backend_native.*   # Compiled PyO3 extension (built)
└─ _rust/                     # Rust crate (PyO3 extension)
├─ Cargo.toml
└─ src/lib.rs                 # Defines #[pymodule] fn _rust_backend_native(...)
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






