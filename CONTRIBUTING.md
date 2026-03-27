# Contributing to Stratum AI

First off, thank you for taking the time to contribute!

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
[LazyDAG] add node deduplication pass
[LazyDAG] implement partition-aware scheduling
[LazyDAG] add integration tests
```

### Referencing Issues

When a commit relates to a GitHub issue, reference it at the end of the commit title:

```
[Fix] correct DAG cycle detection for nested pipelines #42
[LazyDAG] add topological sort for execution planning #18
```

Bigger features should be planned as GitHub issues with sub-issues to break the work into trackable pieces. Each sub-issue maps to one or more commits sharing the same feature tag:

```
#15 LazyDAG execution engine        (parent issue)
  ├── #16 node deduplication pass    → [LazyDAG] add node deduplication pass #16
  ├── #17 partition-aware scheduling → [LazyDAG] implement partition-aware scheduling #17
  └── #18 integration tests          → [LazyDAG] add integration tests #18
```

## Getting Started

_TODO: Add setup instructions._

## Submitting Changes

_TODO: Add PR workflow._
