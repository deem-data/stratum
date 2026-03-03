# Stratum

[![CI](https://github.com/deem-data/stratum/actions/workflows/test.yml/badge.svg)](https://github.com/deem-data/stratum/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/deem-data/stratum/graph/badge.svg?token=QQDTC0RXUN)](https://codecov.io/gh/deem-data/stratum)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Stratum** is an ML system for efficiently executing **large-scale agentic pipeline search**. It integrates with ML agents by representing batches of agent-generated pipelines as lazily evaluated DAGs, applying logical and runtime optimizations, and executing them across heterogeneous backends, including a Rust runtime.
Stratum builds on [skrub's](https://skrub-data.org/stable) operator abstraction.

---

## Design Principles

- Provide seamless and unrestricted support for **arbitrary ML libraries** without operator porting.
- Enable **lazy evaluation** and provide operator semantics that enable logical rewrites and **cost-based** optimizations.
- Implement a runtime with **efficient operator kernels** (in Rust), scheduling across CPUs, GPUs, and distributed backends, plus runtime optimizations such as **buffer pools, reuse of intermediates, and inter- and intra-operator parallelization**.

---

## Installation

For now, you need to build stratum from source.

**Requirements:**
- Python **3.10+**
- [skrub](https://skrub-data.org/stable/)
- [Rust toolchain](https://rustup.rs/) (nightly not required; stable is fine)
- [maturin](https://www.maturin.rs/) (`pip install maturin`)

From the repository root, install the extension in editable (development) mode:

```bash
maturin develop --release
```

For more details (including building wheels), see the **Developer Instructions** section below.

---

## Usage

To leverage stratum, agent prompts or pipelines need minor changes.
Prompts should be modified to generate code following [skrub DataOps](https://skrub-data.org/stable/reference/data_ops.html) syntax.

Stratum can also significantly speed up human-written skrub code.

The following flags enable different features of Stratum. These flags can be set via environment variables or directly in code:

```python
import stratum

stratum.set_config(
    rust_backend=True,
    scheduler=True,
    stats=True,
    debug_timing=False,
)
```
---

## Repository Layout

```bash
stratum/
├─ pyproject.toml          # Project metadata + Python/Rust build config (maturin)
├─ README.md
├─ LICENSE
├─ _rust/                  # Rust crate (PyO3 extension)
│  ├─ Cargo.toml
│  └─ src/lib.rs           # Defines #[pymodule] fn _rust_backend_native(...)
└─ stratum/                # Python package
   ├─ __init__.py          # Façade over skrub + automatic patching
   ├─ _config.py           # set_config/get_config + runtime/env sync
   ├─ _api.py              # High-level grid search / evaluate helpers
   ├─ _rust_backend.py     # Python <-> Rust shim (re-exports native fns)
   ├─ adapters/            # Public API (dispatch to Rust or fall back to skrub)
   │  ├─ string_encoder.py # RustyStringEncoder
   │  └─ one_hot_encoder.py # RustyOneHotEncoder
   ├─ logical_optimizer/   # DAG representation + logical rewrites
   ├─ runtime/             # Schedulers and runtime execution
   ├─ patching/            # Hooks that patch upstream skrub
   └─ tests/               # Test suite
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
# Linux / macOS
maturin build --release -o dist --interpreter python3.10 --compatibility linux

# Windows
maturin build --release -o dist
```
Then install with:

```bash
pip install ./dist/stratum-*.whl
```

---

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.






