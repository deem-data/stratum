"""Plan-time evaluation program for a column map.

A map's ``entries`` are source-relative ``ColumnExpr`` DAGs; after assign-map
fusion, later entries embed earlier ones as shared sub-expressions. Evaluating
each entry as a tree would recompute every shared node once per path to it
(quadratic on towers, exponential on diamonds). :func:`plan_map_program` turns
the entries into a :class:`MapProgram`: a list of steps, each computed once,
that the backend kernels execute.

Planning, in order:

1. **Interning.** Structurally equal nodes are merged into one object, so equal
   sub-expressions written in different stages are shared too.
2. **Materialization.** A non-leaf node is materialized as a step when it has
   two or more uses (parents, plus one use per output entry naming it), or
   when inlining it would nest :data:`MAX_INLINE_DEPTH` levels deep. Other
   nodes are inlined into their consumer; leaves (``Col``, ``Const``,
   ``OperandLeaf``) are always referenced directly. The operand of a column
   method that reads its operand's dtype
   (:attr:`~stratum.optimizer.logical._column_expr.ColumnMethodExpr.reads_operand_dtype`)
   stays inline in that method even when it is also a step, if its dtype is
   resolvable from its shape
   (:func:`~stratum.optimizer.logical._column_expr.has_static_polars_dtype`),
   so the kernel can still resolve it.
3. **Lowering.** Each step's expression and each output's expression reference
   earlier steps through :class:`TempRef`. Steps get private names, so a later
   overwrite of a column cannot change what an earlier step reads: every
   ``Col`` still resolves against the map's input frame.
4. **Levels.** A step's level is the number of steps it transitively waits on
   along its longest path, so steps of one level are independent and can be
   evaluated together.
5. **Stored outputs.** A step whose value is exactly the output ``name``
   records it as :attr:`MapStep.output` when no expression reads ``name`` from
   the input frame. Backends that stage steps as frame columns store such a
   step under ``name`` directly instead of copying a private column into it
   afterwards; since nothing reads the input's ``name``, writing it early
   cannot change any other value. When equal outputs share a step, the first
   in assignment order is stored.

An output used only by itself is not a step: it is evaluated directly when the
outputs are assigned, so a map without shared nodes has no steps at all.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from stratum.optimizer.logical._column_expr import (
    Col, ColumnExpr, ColumnMethodExpr, has_static_polars_dtype, iter_postorder)

#: Prefix of the private column names steps use on backends that stage them as
#: frame columns (polars). Input frames must not contain such columns.
TEMP_PREFIX = "__stratum_map_tmp_"

#: Nesting depth at which an inlined expression is cut into a step. Evaluation
#: recurses once per nesting level, so a deep single-use chain (e.g. a long run
#: of fused ``a = a + 1`` stages) must not stay one expression. A step inlines
#: at most one level more, when a dtype-reading method keeps its cast inline.
MAX_INLINE_DEPTH = 64


class TempRef(ColumnExpr):
    """Reference to the value of an already evaluated :class:`MapStep`.

    Exists only inside a :class:`MapProgram`, never in logical entries. Kernels
    expose each step's value through ``ctx.temps``: the evaluated column on
    pandas, a column reference into the lazy plan on polars.
    """
    __slots__ = ("index",)

    def __init__(self, index: int):
        self.index = index

    def _key(self):
        return self.index

    def __repr__(self):
        return f"TempRef({self.index})"

    def children(self):
        return ()

    def with_children(self, children):
        if children:
            raise TypeError(f"{type(self).__name__} has no children")
        return self

    def to_pandas(self, ctx):
        return ctx.temps[self.index]

    def to_polars(self, ctx):
        return ctx.temps[self.index]


@dataclass(frozen=True, slots=True)
class MapStep:
    """One materialized node: evaluate ``expr`` and store it as ``TempRef(index)``.

    ``output`` names the output this step may be stored under directly (see
    the module docstring), or is ``None`` when it needs a private column.
    """
    index: int
    expr: ColumnExpr
    level: int
    output: str | None = None

    @property
    def name(self) -> str:
        """Private column name of the step."""
        return f"{TEMP_PREFIX}{self.index}"

    @property
    def column(self) -> str:
        """Frame column the step is staged in."""
        return self.output if self.output is not None else self.name


@dataclass(frozen=True, slots=True)
class MapProgram:
    """Steps in dependency order plus the lowered output expressions.

    ``outputs`` preserves the entries' assignment order.
    """
    steps: tuple[MapStep, ...]
    outputs: dict[str, ColumnExpr]

    @property
    def levels(self) -> list[list[MapStep]]:
        grouped: list[list[MapStep]] = []
        for step in self.steps:
            while len(grouped) <= step.level:
                grouped.append([])
            grouped[step.level].append(step)
        return grouped

    @property
    def temp_names(self) -> list[str]:
        """Private columns of the steps not stored under an output name."""
        return [step.name for step in self.steps if step.output is None]

    @property
    def unstored_outputs(self) -> dict[str, ColumnExpr]:
        """Outputs no step is stored under, in assignment order."""
        stored = {step.output for step in self.steps}
        return {name: expr for name, expr in self.outputs.items()
                if name not in stored}


def _intern(roots: list[ColumnExpr]) -> dict[int, ColumnExpr]:
    """Map ``id`` of every node under ``roots`` to its canonical equal node.

    Children are canonicalized first, so comparing two candidates only
    compares canonical children, which are identical objects when equal.
    """
    canonical: dict[int, ColumnExpr] = {}
    table: dict[ColumnExpr, ColumnExpr] = {}
    for node in iter_postorder(roots):
        kids = node.children()
        new_kids = tuple(canonical[id(c)] for c in kids)
        candidate = (node if all(a is b for a, b in zip(kids, new_kids))
                     else node.with_children(new_kids))
        canonical[id(node)] = table.setdefault(candidate, candidate)
    return canonical


def plan_map_program(entries: dict[str, ColumnExpr]) -> MapProgram:
    """Compile a map's entries into a :class:`MapProgram` (see module docstring)."""
    canonical = _intern(list(entries.values()))
    roots = {name: canonical[id(expr)] for name, expr in entries.items()}
    order = iter_postorder(roots.values())

    uses: dict[int, int] = {}
    for node in order:
        for child in node.children():
            uses[id(child)] = uses.get(id(child), 0) + 1
    for root in roots.values():
        uses[id(root)] = uses.get(id(root), 0) + 1

    steps: list[MapStep] = []
    lowered: dict[int, ColumnExpr] = {}
    # Lowered form of a node before it is replaced by its step's TempRef.
    inline: dict[int, ColumnExpr] = {}
    # Number of step levels that must run before a node's value is available.
    ready: dict[int, int] = {}
    # Nesting depth of ``lowered`` / ``inline`` (leaves and TempRefs are 0).
    lowered_depth: dict[int, int] = {}
    inline_depth: dict[int, int] = {}
    for node in order:
        kids = node.children()
        if not kids:
            lowered[id(node)] = node
            ready[id(node)] = 0
            lowered_depth[id(node)] = 0
            continue
        new_kids = [lowered[id(c)] for c in kids]
        kid_depths = [lowered_depth[id(c)] for c in kids]
        if isinstance(node, ColumnMethodExpr) and node.reads_operand_dtype:
            operand = inline.get(id(kids[0]))
            if operand is not None and has_static_polars_dtype(operand):
                new_kids[0] = operand
                kid_depths[0] = inline_depth[id(kids[0])]
        expr = (node if all(a is b for a, b in zip(kids, new_kids))
                else node.with_children(tuple(new_kids)))
        depth = 1 + max(kid_depths)
        inline[id(node)] = expr
        inline_depth[id(node)] = depth
        level = max(ready[id(c)] for c in kids)
        if uses[id(node)] >= 2 or depth >= MAX_INLINE_DEPTH:
            step = MapStep(index=len(steps), expr=expr, level=level)
            steps.append(step)
            lowered[id(node)] = TempRef(step.index)
            ready[id(node)] = level + 1
            lowered_depth[id(node)] = 0
        else:
            lowered[id(node)] = expr
            ready[id(node)] = level
            lowered_depth[id(node)] = depth

    outputs = {name: lowered[id(root)] for name, root in roots.items()}

    read = {node.name for node in order if isinstance(node, Col)}
    stored: dict[int, str] = {}
    for name, expr in outputs.items():
        if type(expr) is TempRef and expr.index not in stored and name not in read:
            stored[expr.index] = name
    steps = [replace(step, output=stored[step.index]) if step.index in stored
             else step for step in steps]
    return MapProgram(steps=tuple(steps), outputs=outputs)
