---
applyTo: "_rust/**"
---

# Review Instructions — `_rust/` (Rust backend)

- **Semantic parity with Python.** Rust kernels must match the semantics of the
  Python implementation they accelerate (null handling, NaN ordering, dtype
  promotion, empty-input behavior). Flag divergence unless it is tested and
  documented.
- **No panics across the FFI boundary.** Flag `unwrap()`/`expect()`/indexing that can
  panic on user-controlled data in code reachable from Python; errors should be
  converted to Python exceptions.
- **GIL discipline.** Long-running or parallel kernels should release the GIL
  (`py.allow_threads`); flag compute loops that hold it.
- **Avoid needless copies at the boundary.** Flag conversions that copy whole
  columns/buffers between Python and Rust when a zero-copy view (e.g., Arrow,
  numpy views) is possible.
- **Allocation in hot loops.** Flag per-element `String`/`Vec` allocation, repeated
  `collect()` into intermediates, or hashing with default hashers in hot kernels
  where a faster alternative is already used elsewhere in the crate.
- **Tests on both sides.** Kernel changes need Rust unit tests and a Python-side
  test in `stratum/tests/` exercising the kernel through the public API.
