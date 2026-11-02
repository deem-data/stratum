"""Index use: whether the pandas index on a plan edge matters to the result.

A polars frame has no index, so converting one to pandas numbers its rows
``0..n-1``. That is exact where the pandas plan numbers them so too
(:func:`~stratum.optimizer.physical._conversion.has_default_index`), and also
where nothing downstream uses the index: no op reads it or pairs rows by it, and
it does not reach the plan's result. Implementation selection converts on such
an edge instead of running its producer on pandas.

An op uses the index of its input ``k`` when it

* reads it (``reads_index``);
* pairs rows by it: ``keyed_inputs``, else ``aligns_by_label``, or, where that
  is ``None``, pandas' alignment of several inputs that may carry an index
  (:func:`carries_index`);
* or passes it on to an output whose index is used: ``index_passed_from``, else
  by ``index_effect``. KEEP and PREFIX pass the first input's, NEW and COMMON
  every input's, DEFAULT and FRESH none. Only a frame or a series has an index, or a value whose kind is
  not known.

A fitted state is not data, and its index is not used.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from stratum.optimizer.logical._base import FRAME_TYPES, IndexEffect, IRNode, OutputType
from stratum.optimizer.logical._ops import ValueOp


def carries_index(op: IRNode) -> bool:
    """Whether ``op``'s output may carry an index: a frame or a series, or a value
    of a kind not known at plan time, which recorded code may make a series."""
    if op.output_type in FRAME_TYPES:
        return True
    if op.output_type is not OutputType.UNKNOWN:
        return False
    if isinstance(op, ValueOp):
        return isinstance(op.value, (pd.Series, pd.DataFrame))
    return True


class IndexUse:
    """Index use over a plan's ops, given in topological order.

    The plan's result is shown to the user, so its index is used. An op outside
    the plan counts as using every index it is given.
    """

    def __init__(self, ops: list[IRNode]):
        self._output_used: dict[int, bool] = {}
        for op in reversed(ops):
            self._output_used[id(op)] = carries_index(op) and (
                not op.outputs or any(
                    self.uses_input(consumer, k) for consumer in op.outputs
                    for k, input_ in enumerate(consumer.inputs) if input_ is op))

    def output_used(self, op: IRNode) -> bool:
        """Whether ``op``'s output index is used."""
        return self._output_used.get(id(op), True)

    def uses_input(self, op: IRNode, k: int) -> bool:
        """Whether ``op`` uses the index of its input ``k``."""
        if op.reads_state(k):
            return False
        if op.reads_index:
            return True
        if op.keyed_inputs is not None:
            if k in op.keyed_inputs:
                return True
        elif _pairs_rows_by_label(op):
            return True
        return k in _passed_inputs(op) and self.output_used(op)


def _pairs_rows_by_label(op: IRNode) -> bool:
    if op.aligns_by_label is not None:
        return op.aligns_by_label
    # pandas aligns elementwise operands; one frame read twice pairs with itself.
    frames = {id(i) for k, i in enumerate(op.inputs)
              if carries_index(i) and not op.reads_state(k)}
    return len(frames) > 1


def _passed_inputs(op: IRNode) -> Sequence[int]:
    """The inputs whose index reaches ``op``'s output index."""
    if op.index_passed_from is not None:
        return op.index_passed_from
    effect = op.index_effect
    if effect in (IndexEffect.DEFAULT, IndexEffect.FRESH):
        return range(0)
    if effect in (IndexEffect.KEEP, IndexEffect.PREFIX):
        return range(1)
    return range(len(op.inputs))
