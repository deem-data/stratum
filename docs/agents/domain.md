# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root. The glossary, in two tiers: the research framing the paper and README speak in, and the system internals the code speaks in.
- **`docs/adr/`**. Read the ADRs that touch the area you are about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Stratum is a single-context repo: one glossary and one ADR directory, both at the root.

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-in-place-implementation-selection.md
│   └── 0002-drop-in-skrub-replacement.md
└── stratum/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR 0002 (drop-in skrub replacement), but worth reopening because…_
