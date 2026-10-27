"""Logical row ordering (``sort_values`` / polars ``sort``).

Kept apart from :class:`~stratum.optimizer.logical._selection_ops.SelectionOp`: a
selection decides which rows survive, a sort decides what order they come out
in. Splitting them is what lets ``value_counts`` be an aggregation *plus* an
ordering rather than an aggregation with an ordering baked into its options, and
it puts the ordering somewhere a later rewrite can drop it from once we can see
that the consumer never observes row order.
"""
from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._ops import MethodCallOp, Op, OutputType, substitute_op


class SortOp(Op):
    """Order rows by literal column names, or by the values of a series.

    ``by`` holds column labels rather than expressions. A computed sort key has
    no pandas spelling without materialising a column first, so an expression
    here would be a representation we could not lower; a map in front of the sort
    expresses the same thing. An empty ``by`` means "order by the values
    themselves", which is the only form a series has.

    ``ascending`` is one flag for every key or one per key, matching both
    backends. ``na_position`` is carried because the defaults disagree: pandas
    puts nulls last, polars puts them first. ``ignore_index`` only matters to
    pandas, whose sort otherwise keeps the permuted index labels.

    Pure config -- execution is provided by the physical impls in
    ``physical/_sort_execs.py``, selected at plan time.
    """
    logical_family = "Sort"
    fields = ["by", "ascending", "na_position", "ignore_index"]

    def __init__(self, by: tuple | list = (), ascending: bool | tuple | list = True,
                 na_position: str = "last",
                 inputs: list[Op] = None, outputs: list[Op] = None,
                 ignore_index: bool = False):
        super().__init__(name=_render_name(by, ascending),
                         inputs=inputs, outputs=outputs)
        self.by = tuple(by)
        self.ascending = (ascending if isinstance(ascending, bool)
                          else tuple(ascending))
        self.na_position = na_position
        self.ignore_index = ignore_index

    @property
    def index_effect(self):
        return IndexEffect.DEFAULT if self.ignore_index else IndexEffect.NEW
        # A sort reorders rows and changes nothing else, so it keeps its input's
        # kind; extraction overrides this with the propagated type.
        self.output_type = OutputType.FRAME

    def update_name(self):
        self.name = _render_name(self.by, self.ascending)


_SORT_KINDS = ("quicksort", "mergesort", "heapsort", "stable")


def make_sort_op(op: MethodCallOp) -> SortOp | None:
    """Extract ``sort_values`` by literal column names, or ``None`` to keep the call.

    Refused: a sort ``key`` (arbitrary code), ``inplace``, sorting columns
    (``axis=1``) and computed or non-string keys. Every ``kind`` is accepted:
    the impls sort stably, which is one valid tie order for every kind.
    """
    args, kwargs = tuple(op.args or ()), dict(op.kwargs or {})
    if (kwargs.pop("axis", 0) not in (0, "index")
            or kwargs.pop("inplace", False) is not False
            or kwargs.pop("key", None) is not None
            or kwargs.pop("kind", "quicksort") not in _SORT_KINDS):
        return None
    if op.inputs[0].output_type is OutputType.SERIES:
        # Series.sort_values takes no `by`; its first positional is the axis.
        if args or "by" in kwargs:
            return None
        by = ()
    else:
        if len(args) > 1 or (args and "by" in kwargs):
            return None
        by = args[0] if args else kwargs.pop("by", None)
        by = [by] if isinstance(by, str) else by
        if not isinstance(by, (list, tuple)) or not by or not all(isinstance(k, str) for k in by):
            return None
    ascending = kwargs.pop("ascending", True)
    na_position = kwargs.pop("na_position", "last")
    ignore_index = kwargs.pop("ignore_index", False)
    if (kwargs or type(ignore_index) is not bool or na_position not in ("first", "last")
            or not (type(ascending) is bool
                    or (isinstance(ascending, (list, tuple)) and len(ascending) == len(by)
                        and all(type(flag) is bool for flag in ascending)))):
        return None
    return substitute_op(op, SortOp(by=by, ascending=ascending, na_position=na_position,
                                    ignore_index=ignore_index), keep_kind=True)


def _render_name(by, ascending) -> str:
    keys = tuple(by) or ("values",)
    flags = ((ascending,) * len(keys) if isinstance(ascending, bool)
             else tuple(ascending))
    return ", ".join(f"{key} {'asc' if flag else 'desc'}"
                     for key, flag in zip(keys, flags))
