"""Dataframe map rewrites: fuse consecutive assign maps into one kernel."""
from __future__ import annotations

from stratum.optimizer.ir._column_expr import substitute_cols
from stratum.optimizer.ir._map_ops import AssignMapOp
from stratum.optimizer.ir._ops import Op
from stratum.optimizer._op_utils import topological_iterator


def _is_self_contained(op: AssignMapOp) -> bool:
    """True when every entry is free of ``OperandLeaf`` / external refs."""
    return all(not any(expr.iter_operand_refs()) for expr in op.entries.values())


def _is_fusible_map(op: Op) -> bool:
    """Whether ``op`` may participate in assign-map fusion.

    Barriers: non-exact type, metadata-bearing maps, multi-input maps (external
    operands), and expressions that still need an intermediate frame object.
    """
    if type(op) is not AssignMapOp:
        return False
    if op.is_X or op.is_y or op.is_split_op:
        return False
    if len(op.inputs) != 1:
        return False
    return _is_self_contained(op)


def _collect_chain(head: AssignMapOp) -> list[AssignMapOp]:
    """Extend ``head`` downstream while each stage has a single map consumer."""
    chain = [head]
    while True:
        cur = chain[-1]
        if len(cur.outputs) != 1:
            break
        nxt = cur.outputs[0]
        if not _is_fusible_map(nxt):
            break
        if nxt.inputs[0] is not cur:
            break
        chain.append(nxt)
    return chain


def _flatten_chain(chain: list[AssignMapOp]) -> dict:
    """Inline sequential assigns into one source-relative ``entries`` dict."""
    bindings: dict = {}
    entries: dict = {}
    for stage in chain:
        rewritten = {
            name: substitute_cols(expr, bindings)
            for name, expr in stage.entries.items()
        }
        for name, expr in rewritten.items():
            # Overwrite keeps the first insertion slot (Python dict order).
            entries[name] = expr
            bindings[name] = expr
    return entries


def _fuse_chain(chain: list[AssignMapOp], root: Op) -> Op:
    """Replace ``chain`` with one ``AssignMapOp`` against the chain source."""
    head, tail = chain[0], chain[-1]
    source = head.inputs[0]
    fused = AssignMapOp(entries=_flatten_chain(chain), inputs=[source], outputs=[])

    source.outputs = [o for o in source.outputs if o is not head]
    source.add_output(fused)

    for consumer in list(tail.outputs):
        consumer.replace_input(tail, fused)
        fused.add_output(consumer)

    for stage in chain:
        for inp in stage.inputs:
            inp.outputs = [o for o in inp.outputs if o is not stage]
        stage.inputs = []
        stage.outputs = []

    if root is tail:
        return fused
    return root


def fuse_assign_maps(root: Op) -> Op:
    """Collapse maximal fusible ``AssignMapOp`` chains into one map each.

    Walks each node once; fused prefixes are marked so chain matching stays
    roughly linear in the number of map nodes.
    """
    seen: set[int] = set()
    for op in list(topological_iterator(root)):
        if id(op) in seen:
            continue
        if not _is_fusible_map(op):
            continue
        chain = _collect_chain(op)
        for stage in chain:
            seen.add(id(stage))
        if len(chain) < 2:
            continue
        root = _fuse_chain(chain, root)
    return root
