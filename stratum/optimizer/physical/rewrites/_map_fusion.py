"""Fuse Polars projections and maps after implementation selection.

Intermediate results used outside the fused subgraph remain materialized.
Fusion stops at conversions, opaque calls, and stateful operators. The logical
plan represents projections, maps, and assignments as separate operators.
"""
from __future__ import annotations

from stratum.optimizer._op_utils import _iter_operand_refs, topological_iterator
from stratum.optimizer.logical import _schema
from stratum.optimizer.logical._base import OperandRef, OutputType
from stratum.optimizer.logical._column_expr import (
    AggExpr, Col, DatetimeExpr, DtExpr, OperandLeaf, StrExpr)
from stratum.optimizer.logical._map_ops import (
    DatetimeConversionOp, DatetimeFieldOp, ExpressionMapOp, series_name)
from stratum.optimizer.physical._aggregation_execs import PolarsFusedScalarAggregateOp, PolarsScalarAggregateOp
from stratum.optimizer.physical._combine_execs import PolarsAssignOp
from stratum.optimizer.physical._expressions import polars_unsupported
from stratum.optimizer.physical._fused_map_execs import PolarsFusedAssignOp, PolarsFusedMapOp
from stratum.optimizer.physical._projection_execs import PolarsColumnProjectionOp, PolarsDropOp
from stratum.optimizer.physical._representation import (
    FrameLib, check_frame_representations, frame_lib)


def _template(op):
    """An expression over local inputs, or None at a fusion boundary."""
    if (frame_lib(op) is not FrameLib.POLARS or op.output_type is not OutputType.SERIES
            or op.has_state or op.state_ref is not None or op.is_split_op):
        return None
    receiver = OperandLeaf(OperandRef(0))
    if isinstance(op, ExpressionMapOp):
        if (isinstance(op.expr, StrExpr)
                and (not op.expr.preserves_columns or op.expr.has_parameter_exprs)):
            return None
        return op.expr
    if isinstance(op, DatetimeFieldOp):
        return DtExpr(receiver, op.field)
    if isinstance(op, DatetimeConversionOp):
        if any(_iter_operand_refs([getattr(op, name) for name in op.fields])):
            return None
        return DatetimeExpr(receiver, op.format, op.exact, op.cache, op.errors)
    return None


def _column_source(op):
    if (isinstance(op, PolarsColumnProjectionOp) and isinstance(op.key, str)
            and len(op.inputs) == 1 and op.inputs[0].output_type is OutputType.FRAME
            and frame_lib(op.inputs[0]) is FrameLib.POLARS):
        return op.inputs[0]
    return None


def _find_source(op):
    """Choose the first frame read through supported maps and projections."""
    seen, stack = set(), [op]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        source = _column_source(node)
        if source is not None:
            return source
        if _template(node) is not None:
            stack.extend(reversed(node.inputs))
    return None


def _assignment_source(op):
    """The base frame and optional projection to perform in the fused select."""
    base = op.inputs[0]
    if base.is_X or base.is_y:
        return base, None
    if isinstance(base, PolarsColumnProjectionOp) and isinstance(base.key, list):
        columns = base.key
        if len(set(columns)) == len(columns):
            return base.inputs[0], list(columns)
    if (isinstance(base, PolarsDropOp) and base.level is None and base.errors == "raise"
            and _schema.is_known(base.inputs[0].output_schema)):
        dropped = _schema.as_column_list(base.columns)
        schema = base.inputs[0].output_schema
        if dropped is not None and set(dropped) <= set(schema):
            return base.inputs[0], [name for name in schema if name not in dropped]
    return base, None


class _Fusion:
    def __init__(self, source, sink, roots):
        self.source, self.sink = source, sink
        self.templates = {}
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node is source or node in self.templates:
                continue
            if node is not sink and (node.is_X or node.is_y):
                continue
            expr = Col(node.key) if _column_source(node) is source else _template(node)
            if expr is not None and polars_unsupported(expr) is None:
                self.templates[node] = expr
                stack.extend(node.inputs[ref.k] for ref in expr.iter_operand_refs())

        # Removing a shared producer also excludes its private predecessors.
        # A producer shared only by branches of this region is still absorbable.
        keep = set(self.templates)
        stack = [node for node in keep if node is not sink
                 and any(out is not sink and out not in keep for out in node.outputs)]
        while stack:
            node = stack.pop()
            if node not in keep:
                continue
            keep.remove(node)
            stack.extend(node.inputs)
        self.absorbed = keep
        self.inputs = [source]
        self.memo = {}

    def expression(self, node):
        if node in self.memo:
            return self.memo[node]
        if node not in self.absorbed:
            if node not in self.inputs:
                self.inputs.append(node)
            return OperandLeaf(OperandRef(self.inputs.index(node)))
        expr = self.templates[node]
        refs = {ref.k: self.expression(node.inputs[ref.k]) for ref in expr.iter_operand_refs()}
        result = expr.remap_operand_refs(refs)
        self.memo[node] = result
        return result

    def install(self, replacement):
        """Keep the sink's identity and metadata; replace its inputs and kernel."""
        old_inputs = list(self.sink.inputs)
        old_fields = set(type(self.sink).fields)
        self.sink.__class__ = replacement
        for name in old_fields - set(replacement.fields):
            delattr(self.sink, name)
        self.sink.inputs = self.inputs
        for node in self.inputs:
            node.add_output(self.sink)
        for node in self.absorbed - {self.sink}:
            node.detach()
        for node in old_inputs:
            if node not in self.inputs:
                node.outputs = [out for out in node.outputs if out is not self.sink]
                if not node.outputs:
                    node.detach_and_prune()


def fuse_maps(root, *, map_aggregate_fusion: bool = False):
    """Fuse selected Polars maps before unrolling and runtime lifetime planning."""
    for op in reversed(list(topological_iterator(root))):
        if not op.outputs and op is not root:
            continue
        if isinstance(op, PolarsAssignOp):
            source, columns = _assignment_source(op)
            if frame_lib(source) is not FrameLib.POLARS:
                continue
            roots = [op.inputs[value.k] for value in op.kwargs.values()
                     if isinstance(value, OperandRef)]
            fusion = _Fusion(source, op, roots)
            if not fusion.absorbed:
                continue
            entries = {name: fusion.expression(op.inputs[value.k])
                       if isinstance(value, OperandRef) else value
                       for name, value in op.kwargs.items()}
            fusion.install(PolarsFusedAssignOp)
            op.entries, op.columns, op.name = entries, columns, "assign"
        elif (map_aggregate_fusion and isinstance(op, PolarsScalarAggregateOp)
              and _template(op.inputs[0]) is not None):
            series = op.inputs[0]
            source = _find_source(series)
            if source is None:
                continue
            fusion = _Fusion(source, op, [series])
            expr = fusion.expression(series)
            # A fused scalar aggregate reads one input relation. Shared producers and
            # columns from other relations remain explicit operands.
            if len(fusion.inputs) != 1:
                continue
            fusion.install(PolarsFusedScalarAggregateOp)
            name, agg = op.aggregations[0]
            op.aggregations = ((name, AggExpr(agg.func, expr, agg.params)),)
            op.update_name()
        elif _template(op) is not None:
            source = _find_source(op)
            if source is None:
                continue
            fusion = _Fusion(source, op, [op])
            if len(fusion.absorbed) < 2:
                continue
            expr = fusion.expression(op)
            result_name = series_name(op)
            effect = op.index_effect
            fusion.install(PolarsFusedMapOp)
            op.expr = expr
            op.result_name = result_name if isinstance(result_name, str) else None
            op.index_effect = effect
            op.name = repr(expr)
    check_frame_representations(topological_iterator(root))
    return root
