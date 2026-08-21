from stratum.optimizer.ir._ops import OperandRef, OutputType, MethodCallOp, Op
from stratum.optimizer.ir import _schema


class AggregateOp(Op):
    """Fused ``groupby(...).agg(...)`` operation.

    Captures a ``DataFrame.groupby(by)`` followed by a single aggregation call
    (e.g. ``.agg("mean")``, ``.sum()``, ``.mean()``, ``.count()``) as one op.
    Both the direct methods and ``.agg(spec)`` are normalized to ``aggregations``
    so ``grouped.agg(aggregations)`` reproduces the original result.

    Pure config -- execution is provided by the physical impls in
    ``physical/_aggregation_execs.py`` (PandasAggregateOp; polars pending),
    selected at plan time.
    """
    logical_family = "Aggregation"
    fields = ["grouping_attributes", "aggregations", "groupby_kwargs"]

    def __init__(self, grouping_attributes: str | list[str] | OperandRef,
                 aggregations: str | list[str] | dict | OperandRef,
                 groupby_kwargs: dict | None = None,
                 inputs: list[Op] | None = None, outputs: list[Op] | None = None):
        # by/agg go in the name so the base helper renders them for both the
        # logical family ("Aggregation(by=..., agg=...)") and the bound physical
        # impl ("PandasAggregateOp(by=..., agg=...)").
        super().__init__(name=f"by={grouping_attributes}, agg={aggregations}",
                         inputs=inputs, outputs=outputs)
        self.grouping_attributes = grouping_attributes
        self.aggregations = aggregations
        self.groupby_kwargs = groupby_kwargs or {}
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        """Best-effort: only a dict aggregation spec gives statically known output
        columns (its keys); a bare function name or list spec is left unknown.
        See :func:`_schema.aggregate_schema` for the exact rules."""
        self.output_schema = _schema.aggregate_schema(
            self.inputs[0].output_schema,
            self.grouping_attributes,
            self.aggregations,
            self.groupby_kwargs.get("as_index"),
        )


class GroupedDataframeOp(Op):
    def __init__(self, ops: list[Op]):
        super().__init__(name="GROUPED_DATAFRAME", is_X=False, is_y=False)
        self.ops = ops
        self.output_type = OutputType.FRAME

    def process(self, mode: str, inputs: list):  # pragma: no cover
        # TODO: GroupedDataframeOp is experimental and not integrated yet.
        # Needs proper refactoring to collect sub-op inputs from the pool.
        raise NotImplementedError("GroupedDataframeOp is not integrated yet.")


# Aggregation methods callable directly on a groupby (no .agg wrapper needed).
_AGG_METHODS = {"sum", "mean", "count", "min", "max", "median", "std", "var",
                "first", "last", "prod", "size", "nunique", "sem"}
# Generic aggregation entrypoints that take the aggregation spec as an argument.
_AGG_FUNCS = {"agg", "aggregate"}


def _is_groupby_op(op: Op) -> bool:
    return isinstance(op, MethodCallOp) and op.method_name == "groupby"


def _is_aggregation(op: MethodCallOp) -> bool:
    """True for a `groupby(...).<agg>()` pair that can fuse into an AggregateOp.

    Requires the aggregation to consume a `groupby` op directly (no GetItem or
    other op in between) and that groupby to have a single consumer.
    """
    if not op.inputs or not _is_groupby_op(op.inputs[0]):
        return False
    if len(op.inputs[0].outputs) != 1:
        return False
    if _extract_grouping(op.inputs[0]) is None:
        return False
    if op.method_name in _AGG_METHODS:
        return True
    # `.agg(spec)` / `.aggregate(spec)`: only the positional-spec form is supported.
    return op.method_name in _AGG_FUNCS and bool(op.args)


def _extract_grouping(groupby_op: MethodCallOp) -> str | list[str] | OperandRef:
    if groupby_op.args:
        return groupby_op.args[0]
    if groupby_op.kwargs and "by" in groupby_op.kwargs:
        return groupby_op.kwargs["by"]
    return None


def _extract_aggregations(op: MethodCallOp) -> str | list[str] | OperandRef:
    if op.method_name in _AGG_FUNCS:
        return op.args[0]
    # direct method such as .mean()/.sum()/.count() -> normalize to its name
    return op.method_name


def make_aggregate_op(op: MethodCallOp) -> AggregateOp:
    """Fuse `groupby(by).agg(...)` (or `.sum()/.mean()/...`) into an AggregateOp."""
    groupby_op = op.inputs[0]
    df = groupby_op.inputs[0]

    grouping_attributes = _extract_grouping(groupby_op)
    aggregations = _extract_aggregations(op)

    # Inputs in resolution order: the frame, then any placeholder operands of the
    # grouping key, then any placeholder operands of the aggregation spec.
    inputs = [df] + list(groupby_op.inputs[1:]) + list(op.inputs[1:])

    # OperandRefs in aggregations index into op.inputs. After prepending
    # groupby_op.inputs[1:], those refs need to shift by that slice's length.
    offset = len(groupby_op.inputs) - 1
    if isinstance(aggregations, OperandRef):
        aggregations = OperandRef(aggregations.k + offset)

    # All groupby kwargs except 'by', which is captured in grouping_attributes.
    groupby_kwargs = {k: v for k, v in (groupby_op.kwargs or {}).items() if k != "by"}

    new_op = AggregateOp(
        grouping_attributes=grouping_attributes,
        aggregations=aggregations,
        groupby_kwargs=groupby_kwargs,
        inputs=inputs,
        outputs=op.outputs,
    )
    # Bypass the now-orphaned groupby op: rewire the frame and grouping-key
    # producers, plus any aggregation-arg producers, to feed the new op.
    groupby_op.replace_output_of_inputs(new_op)
    for extra in op.inputs[1:]:
        extra.replace_output(op, new_op)
    groupby_op.outputs.remove(op)
    return new_op
