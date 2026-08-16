<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/stratum_logo_dark.png">
    <img src="docs/repository-card.png" alt="Stratum logo" width="50%">
  </picture>
</p>

[![Python CI](https://github.com/deem-data/stratum/actions/workflows/python_tests.yml/badge.svg)](https://github.com/deem-data/stratum/actions/workflows/python_tests.yml)
[![Rust CI](https://github.com/deem-data/stratum/actions/workflows/rust_tests.yml/badge.svg)](https://github.com/deem-data/stratum/actions/workflows/rust_tests.yml)
[![codecov](https://codecov.io/gh/deem-data/stratum/graph/badge.svg?token=QQDTC0RXUN)](https://codecov.io/gh/deem-data/stratum)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

**Stratum** is an ML system for efficiently executing **large-scale agentic pipeline search**. It integrates with MLE agents by representing batches of agent-generated pipelines as lazily evaluated DAGs, applying logical and runtime optimizations, and executing them across heterogeneous backends, including a Rust-based runtime.
Stratum builds on [skrub's](https://skrub-data.org/stable) operator abstraction and is under active development.

> 📄 Paper: The motivation and vision behind stratum are described in our [VLDB 2026 paper](https://arxiv.org/pdf/2603.03589).

---

## Design Principles

- Provide seamless and unrestricted support for **arbitrary ML libraries** without operator porting.
- A semantic abstraction built on a minimal set of logical operators that enables rewrites and lazy evaluation with physical operator independence.
- Implement a runtime with **efficient operator kernels** (in Rust), scheduling across CPUs, GPUs, and distributed backends, plus runtime optimizations such as **buffer pools, reuse of intermediates, and inter- and intra-operator parallelization**.

---

## Installation

Install the current development build from PyPI:

```bash
python -m pip install "stratum-ai==0.0.0.dev1"
```

Pre-built Rust wheels are provided for the supported CPython 3.11+ platforms, so a Rust toolchain is not required for a normal installation. The source-build instructions below are for contributors developing Stratum itself.

For source development, you need Python **3.11+**, a [Rust toolchain](https://rustup.rs/) (nightly not required; stable is fine), and [maturin](https://www.maturin.rs/) (`python -m pip install maturin`).

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
    scheduler=True,
    implementation_selector="greedy",
    explain=True,
    stats=True,
)
```
### Example Code

```python
import stratum as skrub #drop-in replacement
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def main():
    # Collect and prepare datasets
    dataset = skrub.datasets.fetch_employee_salaries()
    df = skrub.as_data_op(dataset.employee_salaries).skb.subsample()
    df_clean = df.dropna()
    y = df_clean["current_annual_salary"].skb.mark_as_y()
    X = df_clean.drop(columns=["current_annual_salary"]).skb.mark_as_X()

    # Apply feature transformations
    skrub.set_config(scheduler=True, implementation_selector="greedy", explain=True, stats=True)
    tv = skrub.TableVectorizer()
    X_enc = X.skb.apply(tv)
    print(f"Encoded data shape: {X_enc.shape.skb.preview()}")

    # Training and cross-validation
    pred = X_enc.skb.apply(LinearRegression(), y=y)
    search = pred.skb.make_grid_search(cv=3, fitted=True, scoring="r2", refit=False)
    print(search.results_)

if __name__ == "__main__":
    main()
```
---

## Developer Instructions

### Running the Tests

Install all extras and run the full test suite:

```bash
uv sync --all-extras
pytest -v stratum/tests
```

Or, more concisely:

```bash
uv run pytest
```

---

## Local Dev Install (Editable, without `uv`)

```bash
maturin develop				# Debug mode
maturin develop --release	# Optimized dev build
```

#### Building Wheels

This produces redistributable `.whl` files under `dist/`.

```bash
# Linux / macOS
maturin build --release -o dist --interpreter python3.11 --compatibility linux

# Windows
maturin build --release -o dist
```
Then install with:

```bash
python -m pip install ./dist/stratum_ai-*.whl
```

---

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.



