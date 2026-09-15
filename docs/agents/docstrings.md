# Docstring and comment policy

Two tiers. The distinction is what the reader needs at that level, not a length budget.

## Module docstrings: rationale

A module docstring explains why the module exists, what invariant it holds, and what
would break if that invariant were violated. It may run long where the design needs it.
These are the highest-value documentation in the repo; do not compress them into
one-liners.

Reference examples, all current:

- `stratum/optimizer/ir/_base.py`, why the two operator hierarchies share a base and
  what must stay out of it
- `stratum/optimizer/physical/_physical_ops.py`, abstract versus concrete physical
  operators and why selection mutates in place
- `stratum/optimizer/ir/_scoring.py`, why a metric is scored from the values a plan
  produced rather than through scikit-learn's scorer
- `stratum/optimizer/_op_cse.py`, which operators CSE must not merge

Required for every module under `stratum/optimizer/` and `stratum/runtime/`.

## Class and function docstrings: what, plus the non-obvious

One to three lines. State what the thing produces, and any constraint a caller cannot
infer from the signature. No parameter tables except on the public config surface
(`stratum/_config.py`), where the parameter list is the documentation.

Required for every public class. Internal helpers only when the name does not already
say it.

Current baseline for reference: 287 docstrings, median 3 lines, 46 percent one-liners.
That distribution is the target, not something to fix.

## Comments

Comment the why, never the what. A comment restating the line above it is noise. Comment
density should match the surrounding file.

## Vocabulary

Use the canonical term from `CONTEXT.md`. Do not use a term listed there under `_Avoid_`.
When a docstring names a concept that is not in `CONTEXT.md` yet, that is a signal:
either the term is being invented and should not be, or the glossary has a gap worth
filling.

## Enforcement

Docstring rules are unenforced today; there is no lint configuration in the repo at all.
A minimal ruff config covering module and public-class docstrings is tracked as its own
issue. Until it lands, this file is convention only, which will not survive a large
contributor pool.
