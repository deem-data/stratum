from polars import List

from stratum.optimizer._op_utils import replace_op_in_outputs
from stratum.optimizer.ir._base import remap_operand_refs
from stratum.optimizer.ir._ops import OperandRef, OutputType, MethodCallOp, Op
from typing import TypedDict





# Two caller types: "plain" (Series/DataFrame) and "grouped"
# (SeriesGroupBy/DataFrameGroupBy).
# a mechanisim to handle method signature mismatch depending on the caller type has been created.
#
# there are cases where Series vs Dataframe or SeriesGroupby vs DataFrameGroupBy method variants disagrre on signature.
# the entry is the union. the DataFrame/DataFrameGroupby form is a superset in almost every case. this fact shall be exploited once
# schema propagation is integrated, to ensure the correct dispatch of fields_list.
#
# current implementation does not handle UDF aggregate functions

def _expand(spec: dict) -> dict:
    """Flatten {(method, ...): params} into {method: params}."""
    out = {}
    for methods, params in spec.items():
        for m in methods:
            assert m not in out, f"duplicate signature entry for {m!r}"
            out[m] = params
    return out

_GROUPBY_POSITIONAL = ("by", "level", "as_index", "sort",
                       "group_keys", "observed", "dropna")

ENGINE    = ("engine", "engine_kwargs")
AXIS_SKIP = ("axis", "skipna")


# --- plain Series / DataFrame ----------------------------------------------
_PLAIN = {
    ("sum", "prod"):                     AXIS_SKIP + ("numeric_only", "min_count"),
    ("mean", "median", "min", "max",
     "skew", "kurt", "idxmin", "idxmax"): AXIS_SKIP + ("numeric_only",),
    ("std", "var", "sem"):               AXIS_SKIP + ("ddof", "numeric_only"),
    ("all", "any"):                      ("axis", "bool_only", "skipna"),
    ("quantile",):                       ("q", "axis", "numeric_only",
                                          "interpolation", "method"),
    ("nunique",):                        ("axis", "dropna"),
    ("count",):                          ("axis", "numeric_only"),
    # first / last / size have no plain method in 3.0.2; `.size` is a property
    # and arrives as a GetAttrOp, never as a MethodCallOp.
}


# --- SeriesGroupBy / DataFrameGroupBy --------------------------------------
_GROUPED = {
    ("sum", "min", "max"):     ("numeric_only", "min_count", "skipna") + ENGINE,
    ("prod", "first", "last"): ("numeric_only", "min_count", "skipna"),
    ("mean",):                 ("numeric_only", "skipna") + ENGINE,
    ("median",):               ("numeric_only", "skipna"),
    ("std", "var"):            ("ddof",) + ENGINE + ("numeric_only", "skipna"),
    ("sem",):                  ("ddof", "numeric_only", "skipna"),
    ("skew", "kurt"):          ("skipna", "numeric_only"),
    ("idxmin", "idxmax"):      ("skipna", "numeric_only"),
    ("all", "any"):            ("skipna",),
    ("quantile",):             ("q", "interpolation", "numeric_only"),
    ("nunique",):              ("dropna",),
    ("count", "size"):         (),
}



_AGG_SPEC_POSITIONAL = {
    "plain":   ("func", "axis"),
    "grouped": ("func",),
}

_AGG_GROUPED_KEYWORD_ONLY = ENGINE


_AGG_FUNCS = {"agg", "aggregate"}


_POSITIONAL = {
    "plain":   _expand(_PLAIN),
    "grouped": _expand(_GROUPED),
}


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
    fields = ["grouped", "grouping_kwargs", "agg_method", "aggregation_kwargs"]

    def __init__(self, grouped: bool, grouping_kwargs: dict, agg_method: str, aggregation_kwargs, inputs: list[Op] | None = None, outputs: list[Op] | None = None):
        super().__init__(name="Aggregation", inputs=inputs, outputs=outputs)
        self.grouped = grouped
        self.grouping_kwargs = grouping_kwargs
        self.agg_method = agg_method
        self.aggregation_kwargs = aggregation_kwargs



def _is_groupby_op(op: Op) -> bool:
    return isinstance(op, MethodCallOp) and op.method_name == "groupby"



def _make_grouped_agg_op(agg_method: str, op: MethodCallOp, agg_fields: tuple[str, ...], operand: MethodCallOp, groupby_kwargs: dict) -> AggregateOp | None:
    if len(operand.outputs) != 1:
        return None

    agg_kwargs = dict(zip(agg_fields, op.args))
    if op.kwargs:
        agg_kwargs.update(op.kwargs)

    op_inp = op.inputs[1:]

    offset = len(operand.inputs) - 1
    mapping = {k: k + offset for k in range(1, len(op.inputs))}
    agg_kwargs = remap_operand_refs(agg_kwargs, mapping)

    new_op = AggregateOp(grouped=True, grouping_kwargs=groupby_kwargs, agg_method=agg_method, aggregation_kwargs=agg_kwargs, inputs=operand.inputs + op_inp, outputs=op.outputs)

    operand.replace_output_of_inputs(new_op)

    for _op in op_inp:
        _op.replace_output(op, new_op)

    operand.outputs.remove(op)

    return new_op


def _make_plain_agg_op(agg_method: str, op: MethodCallOp, agg_fields: tuple[str, ...]) -> AggregateOp:
    agg_kwargs = dict(zip(agg_fields, op.args))
    if op.kwargs:
        agg_kwargs.update(op.kwargs)

    new_op = AggregateOp(grouped=False, grouping_kwargs={},  agg_method=agg_method, aggregation_kwargs=agg_kwargs, inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op

def make_aggregate_op(op: MethodCallOp) -> AggregateOp | None:
    agg_method = "agg" if op.method_name in _AGG_FUNCS else op.method_name

    operand = op.inputs[0]
    grouped_operand = _is_groupby_op(operand)

    if grouped_operand:
        groupby_params = dict(zip(_GROUPBY_POSITIONAL, operand.args))
        if operand.kwargs:
            groupby_params.update(operand.kwargs)

        if agg_method == "agg":
            agg_fields = _AGG_SPEC_POSITIONAL["grouped"]
        else:
            if agg_method not in _POSITIONAL["grouped"]:
                return None
            agg_fields = _POSITIONAL["grouped"][agg_method]

        return _make_grouped_agg_op(agg_method, op, agg_fields, operand, groupby_params)

    if agg_method == "agg":
        agg_fields = _AGG_SPEC_POSITIONAL["plain"]
    else:
        if agg_method not in _POSITIONAL["plain"]:
            return None
        agg_fields = _POSITIONAL["plain"][agg_method]

    return _make_plain_agg_op(agg_method, op, agg_fields)
