# Contributing to Stratum AI

First off, thank you for taking the time to contribute!

## Workflow

The typical contribution flow looks like this:

1. **Open a GitHub issue** describing the work — a bug report, feature request, or improvement proposal. For larger features, create a parent issue with sub-issues that break the work into trackable pieces.
2. **Discuss the approach** in the issue (or in a GitHub Discussion for more open-ended topics) before writing code.
3. **Implement** the change, committing with the appropriate tags and issue references (see below).
4. **Open a Pull Request** for review. Link it to the issue and keep commits clean — each commit should be a logical, self-contained unit.
5. **Merge** once the PR is approved and CI passes.

## Commit Message Tags

We use commit message tags to keep our git history organized and scannable. Every commit message should start with one of these tags:

| Tag | Purpose | Example |
|-----|---------|---------|
| `[Minor]` | Small changes with no significant side effects | `[Minor] update .gitignore` |
| `[Fix]` | Bug fixes | `[Fix] correct DAG cycle detection for nested pipelines` |
| `[Perf]` | Performance enhancements (single commit) | `[Perf] vectorize batch scoring in Rust runtime` |
| `[Test]` | Test-only changes | `[Test] add coverage for batch executor edge cases` |
| `[Docs]` | Documentation changes | `[Docs] add API reference for PipelineBuilder` |
| `[_FEATURE_NAME_]` | Part of a multi-commit feature | `[LazyDAG] add topological sort for execution planning` |

### Feature Tags

For larger features built across multiple commits, use a shared feature tag with the feature name. This makes it easy to trace all commits related to a feature:

```
[LazyDAG] add node deduplication pass (#16)
[LazyDAG] implement partition-aware scheduling (#17)
[LazyDAG] add integration tests (#18)
```

### Referencing Issues

Most commits should have an associated GitHub issue. The only exception is `[Minor]` — small, self-explanatory changes that don't need an issue.

When a commit relates to a GitHub issue, reference it at the end of the commit title:

```
[Fix] correct DAG cycle detection for nested pipelines (#42)
[LazyDAG] add topological sort for execution planning (#18)
```

Bigger features should be planned as GitHub issues with sub-issues to break the work into trackable pieces. Each sub-issue maps to one or more commits sharing the same feature tag:

```
#15 LazyDAG execution engine        (parent issue)
  ├── #16 node deduplication pass    → [LazyDAG] add node deduplication pass #16
  ├── #17 partition-aware scheduling → [LazyDAG] implement partition-aware scheduling #17
  └── #18 integration tests          → [LazyDAG] add integration tests #18
```
