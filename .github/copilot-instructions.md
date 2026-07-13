# Review Instructions — Stratum (project-wide)

## Project context

Stratum is an ML system for efficiently executing large-scale agentic pipeline search.
It represents batches of agent-generated pipelines as lazily evaluated DAGs, applies
logical and runtime optimizations, and executes them on heterogeneous backends,
including a Rust runtime. It builds on skrub's operator abstraction (skrub is pinned
in `pyproject.toml`), and performance is a core product goal — treat performance
regressions as bugs, not style issues.

Layout:

- `stratum/` — Python source (optimizer, runtime, adapters, patching, utils)
- `stratum/tests/` — pytest suite, mirrors the source layout
- `_rust/` — Rust backend, exposed to Python via maturin/pyo3

## Review priorities (in order)

1. **Correctness of optimizations** — a rewrite or optimization that changes results
   is the worst possible bug in this codebase.
2. **Performance regressions** — see `.github/instructions/stratum.instructions.md`.
3. **Missing or shallow tests** — see `.github/instructions/stratum.instructions.md`.
4. Everything else (style, naming, docs).

## General rules

- **Lazy evaluation must stay lazy.** Flag any change that forces evaluation of a
  lazy DAG or materializes intermediate data earlier than necessary (e.g., calling
  `.compute()`/eval-like methods, converting lazy expressions to concrete
  DataFrames, iterating rows) outside the runtime layer.
- **Python/Rust parity.** If a change alters the semantics of an operation that has
  both a Python and a Rust implementation, flag it unless both sides are updated
  (or the divergence is explicitly justified in the PR description).
- **Monkeypatching is fragile.** Changes under `stratum/patching/` depend on skrub
  internals of the pinned skrub version. Flag any patching change that assumes
  behavior not guaranteed by the pinned version, and any dependency version bump
  that could invalidate existing patches.
- **Public API changes** (anything imported in `stratum/__init__.py` or `stratum/_api.py`)
  should be reflected in docs/README and covered by tests.
