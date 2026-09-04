# AGENTS.md

## Vocabulary

Read `CONTEXT.md` before writing code or docs. It is the project's glossary, in two
tiers: the research framing (pipeline, trajectory, batch, merging) and the system
internals (logical operator, physical operator, plan, lowering, fusion, CSE). Use the
canonical term; do not use a term listed there under `_Avoid_`.

Architectural decisions that read as bugs without context are recorded in `docs/adr/`.

## Writing docs and comments

See `docs/agents/docstrings.md`. In short: module docstrings carry design rationale and
may run long, class and function docstrings are one to three lines, comments explain why
and never what.

## Review rules

Path-scoped review rules live under `.github/`. They are written in Copilot's format but
apply to any agent working here:

- `.github/copilot-instructions.md` project-wide priorities. Correctness of
  optimizations first, then performance regressions, then test quality.
- `.github/instructions/stratum.instructions.md` for `stratum/**`. Testing rules
  (rewrites need equivalence tests, tests must fail without the fix) and performance
  rules (asymptotic complexity, materialization, hot versus cold paths).
- `.github/instructions/rust.instructions.md` for `_rust/**`. Python parity, no panics
  across FFI, GIL discipline, allocation in hot loops.

## Contributing conventions

See `CONTRIBUTING.md` for commit message tags (`[Fix]`, `[Perf]`, `[Docs]`,
`[_FEATURE_NAME_]`, ...) and the parent-issue-with-sub-issues workflow.

## Agent skills

### Issue tracker

Issues live as GitHub issues on `deem-data/stratum` (the `upstream` remote), driven with
the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
