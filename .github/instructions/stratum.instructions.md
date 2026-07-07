---
applyTo: "stratum/**"
---

# Review Instructions — `stratum/` (Python source)

## Testing rules

Every behavioral change under `stratum/` (excluding `stratum/tests/`) must come with
meaningful tests under `stratum/tests/`, in the directory mirroring the source location
(e.g., `stratum/optimizer/` → `stratum/tests/logical_optimizer/`,
`stratum/runtime/` → `stratum/tests/runtime/`). Codecov enforces line coverage, so do
not comment on raw coverage numbers — instead review test *quality*:

- **Flag changes with no test changes.** If a PR modifies source behavior but touches
  no test file, add a finding asking for tests (unless the change is a pure refactor,
  comment/docs change, or is explicitly justified in the PR description).
- **Flag shallow tests.** Tests that merely execute the new code path without asserting
  on the result, assert only that no exception is raised, or duplicate an existing test
  with renamed variables do not count as meaningful coverage.
- **Require edge-case coverage** appropriate to the change. For this codebase the
  recurring edge cases are:
  - empty DataFrames / zero-row and zero-column inputs
  - null/NaN/None values, and NaN-sensitive semantics in numeric rewrites
  - single-element and duplicate-value inputs
  - mixed dtypes, non-default indexes, and unordered/unsorted data
  - both pandas and polars inputs where the code is backend-generic
  - both the Python fallback and the Rust backend where both implement the operation
- **Optimizer rewrites need equivalence tests.** Any new or modified rewrite in
  `stratum/optimizer/` must have a test asserting that optimized and unoptimized
  plans produce identical results (not just identical plan shapes), including on
  inputs designed to break the rewrite (NaN, negative values, empty inputs, nested
  expressions that trigger rule interaction).
- **Tests must fail without the fix.** For bug-fix PRs, check that the added test
  actually exercises the fixed behavior; if the test would pass on the pre-fix code,
  flag it.

## Performance rules

Stratum's purpose is performance. For every change, actively look for ways it could
perform poorly — and flag them even when the code is functionally correct:

- **Asymptotic complexity.** Flag anything that scales worse than the code it replaces:
  nested loops over rows or DAG nodes, `O(n²)` pairwise comparisons, repeated linear
  scans inside a loop, list `in` checks where a set is warranted. Optimizer passes run
  over potentially large DAGs (batches of pipelines) — passes should stay roughly
  linear in graph size.
- **Unnecessary materialization and copies.** Flag avoidable `df.copy()`, conversions
  between pandas/polars/numpy back and forth, `.to_list()`/`.tolist()` on large
  columns, building large intermediate lists instead of iterators, and row-wise
  operations (`iterrows`, `apply` with a Python lambda) where a vectorized or
  Rust-backed alternative exists.
- **Memory behavior.** Anything touching `stratum/runtime/` (buffer pool, serialization,
  scheduler) must be reviewed for: unbounded growth of caches/dicts, objects kept alive
  past their use (preventing buffer reuse or eviction), reference cycles involving large
  buffers, and size-accounting mistakes (LRU eviction relies on `_object_size.py` being
  accurate).
- **Hot paths vs. cold paths.** Per-row, per-value, or per-node work is hot; per-fit or
  per-plan work is cold. Flag expensive work (logging with eager string formatting,
  exception handling as control flow, redundant validation, repeated attribute lookups
  of invariant values) added to hot paths.
- **Degenerate inputs.** Ask how the change behaves on: very wide frames (thousands of
  columns), very tall frames, high-cardinality string columns (adapters/encoders!),
  deeply nested expression trees, and large batches of near-identical pipelines
  (the CSE/common-subexpression machinery should benefit, not choke).
- **Ask for evidence when it matters.** If a change plausibly affects runtime or memory
  of a hot path and the PR contains no benchmark numbers or reasoning, add a finding
  requesting a micro-benchmark or a justification.
