"""Fast graph extraction from a skrub DataOp DAG.

Drop-in replacement for ``skrub._data_ops._evaluation._Graph().run(dag)``
that avoids the heavyweight generator-based ``_DataOpTraversal`` machinery.
We only need the DataOp-to-DataOp adjacency; choices, estimators, slices etc.
are irrelevant for graph structure and can be skipped.
"""

from collections import defaultdict
from skrub._data_ops import DataOp
from skrub._data_ops._choosing import BaseChoice, Choice, Match


_BUILTIN_SEQ = (list, tuple, frozenset, set)


def _collect_child_data_ops(value):
    """Yield all DataOp objects reachable from *value*.

    Handles DataOps stored directly in a field, or nested inside the built-in
    container types that skrub uses (tuple, list, dict, set, frozenset),
    as well as skrub Choice/Match wrappers.
    """
    if isinstance(value, DataOp):
        yield value
    elif isinstance(value, Match):
        yield from _collect_child_data_ops(value.choice)
        yield from _collect_child_data_ops(value.outcome_mapping)
    elif isinstance(value, Choice):
        for outcome in value.outcomes:
            yield from _collect_child_data_ops(outcome)
    elif isinstance(value, BaseChoice):
        pass
    elif isinstance(value, dict):
        for v in value.values():
            yield from _collect_child_data_ops(v)
    elif isinstance(value, _BUILTIN_SEQ):
        for item in value:
            yield from _collect_child_data_ops(item)


def _unique(seq):
    """Deduplicate while preserving order."""
    return list(dict.fromkeys(seq))


def build_graph(data_op):
    """Build the graph dict for a DataOp DAG.

    Returns the same ``{"nodes", "children", "parents"}`` dict produced by
    ``skrub._data_ops._evaluation._Graph().run()``, with integer ids starting
    from 0.

    Uses an iterative stack-based DFS that only visits DataOp nodes,
    skipping the generator protocol and all non-DataOp node types.
    """
    raw_nodes = {}
    raw_children = defaultdict(list)
    raw_parents = defaultdict(list)

    stack = [data_op]
    visited = set()

    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        raw_nodes[node_id] = node

        impl = node._skrub_impl
        for field_name in impl._fields:
            attr = getattr(impl, field_name)
            for child in _collect_child_data_ops(attr):
                child_id = id(child)
                raw_children[node_id].append(child_id)
                raw_parents[child_id].append(node_id)
                if child_id not in visited:
                    stack.append(child)

    short = {obj_id: i for i, obj_id in enumerate(raw_nodes)}
    nodes = {short[k]: v for k, v in raw_nodes.items()}
    children = {
        short[k]: [short[c] for c in _unique(v)]
        for k, v in raw_children.items()
    }
    parents = {
        short[k]: [short[p] for p in _unique(v)]
        for k, v in raw_parents.items()
    }
    return {"nodes": nodes, "children": children, "parents": parents}
