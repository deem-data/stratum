# Stratum integrates as a drop-in skrub replacement

`import stratum as skrub` is the integration contract. `stratum/__init__.py` patches
upstream skrub at import, re-exports a small number of Stratum subclasses under skrub's
own names, and forwards every other attribute to real skrub through a module-level
`__getattr__`. Agent-generated and human-written skrub code therefore runs under Stratum
unchanged, which is the point: requiring an explicit Stratum API would mean rewriting
agent prompts and every existing pipeline.

## Considered options

**Fork or vendor skrub.** Rejected: the maintenance burden of tracking upstream, and
divergence would break the claim that Stratum accelerates ordinary skrub code.

**Expose an explicit Stratum API.** Rejected: it defeats the unchanged-code property
that makes integration with an MLE agent cheap.

**Upstream the changes into skrub.** Too slow for research iteration, and much of what
Stratum does (a logical/physical split, a Rust runtime) is out of scope for skrub.

## Consequences

The patch surface depends on internals of the pinned skrub version (`skrub==0.8.0` in
`pyproject.toml`), so a version bump can silently invalidate it. That surface is kept
deliberately minimal: physical-operator registration replaced import-time class
replacement, and one method-level patch remains,
`SkrubNamespace.make_grid_search`. A nearly empty `stratum/patching/` module is the
expected steady state, not an unfinished migration.
