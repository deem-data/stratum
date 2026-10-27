from collections.abc import Sequence

import pandas as pd

from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._ops import OperandRef, OutputType, MethodCallOp, Op, substitute_op
from stratum.optimizer.logical import _schema


class JoinOp(Op):
    """Logical join that combines both sides. Pure config -- execution is
    provided by the physical impls in ``physical/_join_execs.py``
    (Pandas/PolarsJoinOp), selected at plan time.

    The filtering joins (semi/anti) add no columns and so are a different
    operator: :class:`FilteringJoinOp`, in the same ``Join`` family.
    """
    logical_family = "Join"
    fields = ["how", "left_on", "right_on", "left_index", "right_index", "suffixes"]

    def __init__(
        self,
        how: str = "inner",
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: Sequence[str] = ("_x", "_y"),
        inputs: list[Op] | None = None,
        outputs: list[Op] | None = None,
    ):
        # Renders as the bare family label.
        super().__init__(name="", inputs=inputs, outputs=outputs)
        if how in ("semi", "anti"):
            raise ValueError(f"how={how!r} is a filtering join: use FilteringJoinOp.")
        self.how = how
        self.left_on = left_on
        self.right_on = right_on
        self.left_index = left_index
        self.right_index = right_index
        self.suffixes = suffixes
        self.output_type = OutputType.FRAME

    @property
    def index_effect(self):
        # A merge numbers its rows 0..n-1; an index join keeps labels.
        return IndexEffect.NEW if self.left_index or self.right_index else IndexEffect.DEFAULT

    @property
    def aligns_by_label(self):
        # Index joins key on labels; key joins on column values.
        return self.left_index or self.right_index

    def propagate_output_schema(self):
        """Merge the two sides; `how` is part of the schema, since a null-padded
        side loses its integer dtypes. See :func:`_schema.join_schema`."""
        left = self.inputs[0].output_schema
        right = self.inputs[1].output_schema
        if not _schema.is_known(left) or not _schema.is_known(right):
            self.output_schema = None
            return
        self.output_schema = _schema.join_schema(
            left, right, self._collapsed_keys(left, right), self.suffixes, self.how)

    def _collapsed_keys(self, left, right):
        """The key names that merge folds into a single output column.

        pandas pairs `left_on`/`right_on` *positionally* and collapses a pair only
        when its two names are equal; a pair of differing names keeps both columns
        (and suffixes them if they clash with the other side). So this is a
        positional walk, not a set intersection -- `left_on=["a","b"]` against
        `right_on=["b","a"]` shares both names yet collapses neither, and pandas
        emits `a_x, b_x, a_y, b_y`.

        With no key arguments at all, a bare `merge()` infers the keys as the two
        frames' common columns, and those do collapse.

        Returns `None` when the keys can't be worked out statically, which makes
        the whole schema unknown rather than a confidently wrong column set.
        """
        # A cross join matches every row pair and takes no keys, so nothing
        # collapses and every shared name is suffixed.
        if self.how == "cross":
            return set()

        if self.left_on is None and self.right_on is None:
            # An index-to-index join has no key *columns*; the index is not part
            # of the schema, so every shared column name is simply suffixed.
            if self.left_index or self.right_index:
                return set()
            # Bare merge(): pandas infers the keys from the common columns.
            return set(left) & set(right)

        # Mixing one side's key column with the other side's index makes pandas
        # emit an extra key column (`left_on="k", right_index=True` on two frames
        # that both hold `k` yields `k, k_x, k_y`). Not worth modelling.
        if self.left_on is None or self.right_on is None:
            return None
        if self.left_index or self.right_index:
            return None

        left_on = _schema.as_column_list(self.left_on)
        right_on = _schema.as_column_list(self.right_on)
        if left_on is None or right_on is None or len(left_on) != len(right_on):
            return None
        return {lhs for lhs, rhs in zip(left_on, right_on) if lhs == rhs}


class FilteringJoinOp(Op):
    """Semi or anti join: keeps the left rows that do (``anti=False``) or do not
    (``anti=True``) have a key match on the right, and adds no columns.

    ``left_on``/``right_on`` name the key columns (a name or a list of names,
    paired positionally). ``right_on`` may be ``None``, meaning the right input is
    itself the sequence of key values rather than a relation with a named key
    column; that is what an ``isin`` against a column produces. Missing keys
    match each other.
    """
    logical_family = "Join"
    fields = ["anti", "left_on", "right_on"]
    # Keeps the left rows' labels, and matches key values, never labels.
    index_effect = IndexEffect.NEW
    aligns_by_label = False

    def __init__(
        self,
        anti: bool = False,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        inputs: list[Op] | None = None,
        outputs: list[Op] | None = None,
    ):
        # "Join" alone cannot tell a semi from an anti, so name the direction.
        super().__init__(name="anti" if anti else "semi", inputs=inputs, outputs=outputs)
        self.anti = anti
        self.left_on = left_on
        self.right_on = right_on
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        """The left rows it keeps, with the left columns, whatever the build side holds."""
        self.output_schema = self.inputs[0].output_schema


#: The pandas arguments a :class:`JoinOp` field holds.
_JOIN_OP_FIELDS = {"how", "left_on", "right_on", "left_index", "right_index", "suffixes"}


def make_join_op(op: MethodCallOp) -> JoinOp | None:
    """Parse ``DataFrame.merge`` or ``DataFrame.join`` into a :class:`JoinOp`.

    The call is bound against pandas' own signature, so both methods' positional
    orders hold. ``join`` matches the right side's index, by default against the
    left index too, or against the ``on`` column; ``df.join([df2, df3])`` unrolls
    into a chain of index joins.

    Returns ``None`` (the call stays opaque and runs on pandas as recorded) for a
    call a ``JoinOp`` cannot hold: ``sort=True``, options such as ``indicator`` or
    ``validate``, a graph-fed option, a right side that is not a graph input, and
    a right side that is also the left one or repeats in a chain.
    """
    call = op.bind_pandas(owner=pd.DataFrame)
    if call is None:
        return None
    other = call.pop("right" if op.method_name == "merge" else "other", None)
    is_chained = op.method_name == "join" and isinstance(other, (list, tuple))
    rights = list(other) if is_chained else [other]
    # Each right side is its own input edge after the left one, and nothing else
    # is. A frame used twice de-duplicates to one edge, a literal frame has none,
    # and a graph-fed option adds one.
    if (not all(isinstance(r, OperandRef) for r in rights)
            or sorted(r.k for r in rights) != list(range(1, len(op.inputs)))):
        return None
    fields = _join_fields(call, is_chained) if op.method_name == "join" else _merge_fields(call)
    if fields is None or fields.pop("sort", False) is not False:
        return None
    if set(fields) - _JOIN_OP_FIELDS:
        return None

    if is_chained:
        return _make_chained_join_op(op, fields)

    return substitute_op(op, JoinOp(**fields))


def _merge_fields(call: dict) -> dict | None:
    """``merge``'s bound arguments as :class:`JoinOp` fields, or ``None`` when
    pandas would reject them."""
    # `on` applies to both sides when left_on/right_on are unset.
    on = call.pop("on", None)
    if "left_on" not in call and "right_on" not in call:
        call["left_on"] = call["right_on"] = on
    elif on is not None:
        return None  # pandas rejects on= next to left_on=/right_on=
    call.setdefault("suffixes", ("_x", "_y"))
    return call


def _join_fields(call: dict, is_chained: bool) -> dict:
    """``join``'s bound arguments as :class:`JoinOp` fields.

    ``join`` always matches the right side's index: against the ``on`` column of
    the left side, or against its index. A chained join is index-based on every
    link.
    """
    on = call.pop("on", None)
    call.setdefault("how", "left")
    if on is not None and not is_chained:
        call.update(left_on=on, left_index=False, right_index=True)
    else:
        call.update(left_index=True, right_index=True)
    # join names its suffixes lsuffix/rsuffix, both "" by default in pandas.
    call["suffixes"] = (call.pop("lsuffix", ""), call.pop("rsuffix", ""))
    return call


def _make_chained_join_op(op: MethodCallOp, join_kwargs: dict) -> JoinOp:
    """Unroll df1.join([df2, df3, ...]) into a chain of binary JoinOps."""
    dfs = op.inputs
    prev = dfs[0]
    final_join = None
    n_links = len(dfs) - 1
    for i, right in enumerate(dfs[1:]):
        is_last = i == n_links - 1
        join = JoinOp(**join_kwargs, inputs=[prev, right],
                      outputs=op.outputs if is_last else [])
        right.replace_output(op, join)
        if final_join is not None:
            final_join.outputs = [join]
        else:
            prev.replace_output(op, join)
        prev = join
        final_join = join
    return final_join
