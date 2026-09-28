"""Logical row ordering (``sort_values`` / polars ``sort``).

Kept apart from :class:`~stratum.optimizer.logical._selection_ops.SelectionOp`: a
selection decides which rows survive, a sort decides what order they come out
in. Splitting them is what lets ``value_counts`` be an aggregation *plus* an
ordering rather than an aggregation with an ordering baked into its options, and
it puts the ordering somewhere a later rewrite can drop it from once we can see
that the consumer never observes row order.
"""
from stratum.optimizer.logical._ops import Op, OutputType


class SortOp(Op):
    """Order rows by literal column names, or by the values of a series.

    ``by`` holds column labels rather than expressions. A computed sort key has
    no pandas spelling without materialising a column first, so an expression
    here would be a representation we could not lower; a map in front of the sort
    expresses the same thing. An empty ``by`` means "order by the values
    themselves", which is the only form a series has.

    ``ascending`` is one flag for every key or one per key, matching both
    backends. ``na_position`` is carried because the defaults disagree: pandas
    puts nulls last, polars puts them first.

    Pure config -- execution is provided by the physical impls in
    ``physical/_sort_execs.py``, selected at plan time.
    """
    logical_family = "Sort"
    fields = ["by", "ascending", "na_position"]

    def __init__(self, by: tuple | list = (), ascending: bool | tuple | list = True,
                 na_position: str = "last",
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=_render_name(by, ascending),
                         inputs=inputs, outputs=outputs)
        self.by = tuple(by)
        self.ascending = (ascending if isinstance(ascending, bool)
                          else tuple(ascending))
        self.na_position = na_position
        # A sort reorders rows and changes nothing else, so it keeps its input's
        # kind; extraction overrides this with the propagated type.
        self.output_type = OutputType.FRAME

    def update_name(self):
        self.name = _render_name(self.by, self.ascending)


def _render_name(by, ascending) -> str:
    keys = tuple(by) or ("values",)
    flags = ((ascending,) * len(keys) if isinstance(ascending, bool)
             else tuple(ascending))
    return ", ".join(f"{key} {'asc' if flag else 'desc'}"
                     for key, flag in zip(keys, flags))
