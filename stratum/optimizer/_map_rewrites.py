from __future__ import annotations

from stratum.optimizer.logical._column_expr import (
    Col, ColumnMethodExpr, Const, iter_postorder, substitute_cols)
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._ops import Op
from stratum.optimizer._op_utils import topological_iterator


def _is_self_contained(op: AssignMapOp) -> bool:
    return all(not any(expr.iter_operand_refs()) for expr in op.entries.values())


def _is_fusible_map(op: Op) -> bool:
    if type(op) is not AssignMapOp:
        return False
    if op.is_X or op.is_y or op.is_split_op:
        return False
    if len(op.inputs) != 1:
        return False
    return _is_self_contained(op)


def _reads_dtype_of(op: AssignMapOp, names: set[str]) -> bool:
    """Whether an entry of ``op`` applies a dtype-reading column method directly
    to one of the columns ``names``.

    Fusing ``op`` behind the stages that assign those columns would replace the
    ``Col`` operand with the assigned expression, whose dtype the polars kernel
    cannot resolve, changing how the method treats NaN.
    """
    for node in iter_postorder(op.entries.values()):
        if (isinstance(node, ColumnMethodExpr) and node.reads_operand_dtype
                and isinstance(node.operand, Col) and node.operand.name in names):
            return True
    return False


def _reads_column_of(op: AssignMapOp, names: set[str]) -> bool:
    """Whether an entry of ``op`` reads any of the columns ``names``."""
    if not names:
        return False
    return any(isinstance(node, Col) and node.name in names
               for node in iter_postorder(op.entries.values()))


def _update_constant_names(constant: set[str], op: AssignMapOp) -> None:
    """Track which columns the chain so far binds to a ``Const`` entry.

    A ``Const`` evaluates to a plain Python scalar on pandas, while the unfused
    plan reads the assigned column back as a broadcast Series. Inlining one
    into a later stage would run Python scalar semantics instead (``~True`` is
    ``-2``, ``1 / 0`` raises, ``"x".str`` does not exist), so a stage reading
    such a column ends the chain.
    """
    for name, expr in op.entries.items():
        if type(expr) is Const:
            constant.add(name)
        else:
            constant.discard(name)


def _collect_chain(head: AssignMapOp) -> list[AssignMapOp]:
    chain = [head]
    assigned = set(head.entries)
    constant: set[str] = set()
    _update_constant_names(constant, head)
    while True:
        cur = chain[-1]
        if len(cur.outputs) != 1:
            break
        nxt = cur.outputs[0]
        if not _is_fusible_map(nxt):
            break
        if nxt.inputs[0] is not cur:
            break
        if _reads_dtype_of(nxt, assigned) or _reads_column_of(nxt, constant):
            break
        chain.append(nxt)
        assigned.update(nxt.entries)
        _update_constant_names(constant, nxt)
    return chain


def _flatten_chain(chain: list[AssignMapOp]) -> dict:
    """Inline sequential assigns into source-relative expression DAGs.

    Every stage's entries are rewritten against the entries of the stages
    before it, with one memo per stage so sub-expressions shared between a
    stage's entries stay shared after inlining. The physical kernels rely on
    that sharing to evaluate each shared node once.
    """
    entries: dict = {}
    for stage in chain:
        memo: dict = {}
        rewritten = {
            name: substitute_cols(expr, entries, memo)
            for name, expr in stage.entries.items()
        }
        entries.update(rewritten)
    return entries


def _fuse_chain(chain: list[AssignMapOp], root: Op) -> Op:
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
    """Collapse maximal fusible ``AssignMapOp`` chains into one map each."""
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
