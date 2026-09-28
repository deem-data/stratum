from collections.abc import Sequence
from stratum.optimizer.logical._ops import OperandRef, OutputType, MethodCallOp, Op
from stratum.optimizer.logical import _schema


#: Joins that filter the left side instead of combining both. They return the
#: left relation's columns unchanged, so they carry no suffixes, and their right
#: input is a build side whose values are only tested for membership.
FILTERING_JOINS = ("semi", "anti")


class JoinOp(Op):
    """Logical join. Pure config -- execution is provided by the physical impls
    in ``physical/_join_execs.py`` (Pandas/PolarsJoinOp), selected at plan time.

    ``how`` also covers the two *filtering* joins, ``semi`` and ``anti``, which
    keep the left rows that do (or do not) have a match and add no columns. They
    live here rather than in a separate operator so that join reordering and the
    cost model treat every join uniformly. For those, ``right_on`` may be ``None``,
    meaning the right input is itself the sequence of key values rather than a
    relation with a named key column; that is what an ``isin`` against a column
    produces.
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
        # Regular joins render as the bare family label; a filtering join names its
        # direction, since "Join" alone cannot tell a semi from an anti.
        super().__init__(name=how if how in FILTERING_JOINS else "",
                         inputs=inputs, outputs=outputs)
        if how in FILTERING_JOINS:
            # Nothing to disambiguate when no columns are added. Keeping a suffix
            # pair would be a dead field, and a false difference under CSE.
            if suffixes not in (None, ("_x", "_y")):
                raise ValueError(
                    f"suffixes is meaningless for how={how!r}: a filtering join "
                    f"returns the left columns unchanged.")
            suffixes = None
        self.how = how
        self.left_on = left_on
        self.right_on = right_on
        self.left_index = left_index
        self.right_index = right_index
        self.suffixes = suffixes
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        """Merge the two sides; `how` is part of the schema, since a null-padded
        side loses its integer dtypes. See :func:`_schema.join_schema`.

        A filtering join returns the left rows it keeps and adds no columns, so its
        schema is the left one, whatever the build side holds."""
        left = self.inputs[0].output_schema
        if self.how in FILTERING_JOINS:
            self.output_schema = left
            return
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


_MERGE_POSITIONAL = ["how", "on", "left_on", "right_on",
                    "left_index", "right_index", "sort", "suffixes"]
_JOIN_POSITIONAL = ["on", "how", "lsuffix", "rsuffix", "sort"]
_JOIN_OP_FIELDS = {"how", "left_on", "right_on", "left_index", "right_index", "suffixes"}


def make_join_op(op: MethodCallOp) -> JoinOp:
    # First positional arg is the right/other df; it's already in op.inputs
    pos_args = op.args[1:] if op.args else ()
    pos_names = _MERGE_POSITIONAL if op.method_name == "merge" else _JOIN_POSITIONAL

    params = dict(zip(pos_names, pos_args))
    if op.kwargs:
        params.update(op.kwargs)
    params.pop("other", None)

    other_arg = op.args[0] if op.args else None
    if other_arg is None and op.kwargs:
        other_arg = op.kwargs.get("other")

    if isinstance(other_arg, (list, tuple)):
        # Compare by operand index: a frame used twice de-duplicates to one input
        # edge (the same OperandRef.k), which the chained-join unrolling can't handle.
        keys = [x.k if isinstance(x, OperandRef) else id(x) for x in other_arg]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Duplicate right-hand frames in chained joins are not supported."
            )

    is_chained = (
        op.method_name == "join"
        and (isinstance(other_arg, (list, tuple)) or len(op.inputs) > 2)
    )

    if op.method_name == "join":
        # pandas .join() defaults to how="left" and matches against right's index.
        params.setdefault("how", "left")
        if is_chained:
            # Chained joins are always index-based on every link.
            params["left_index"] = True
            params["right_index"] = True
            params.pop("on", None)
            params.pop("left_on", None)
            params.pop("right_on", None)
        elif "on" in params:
            params["left_on"] = params.pop("on")
            params["left_index"] = False
            params["right_index"] = True
        else:
            params["left_index"] = True
            params["right_index"] = True
        # join uses lsuffix/rsuffix instead of suffixes; both default to "" in pandas.
        if "lsuffix" in params or "rsuffix" in params:
            params["suffixes"] = (params.pop("lsuffix", ""), params.pop("rsuffix", ""))
        params.setdefault("suffixes", ("", ""))
    else:
        # merge's `on` applies to both sides when left_on/right_on are unset.
        if "on" in params and "left_on" not in params and "right_on" not in params:
            shared = params.pop("on")
            params["left_on"] = shared
            params["right_on"] = shared
        else:
            params.pop("on", None)
        params.setdefault("suffixes", ("_x", "_y"))

    if params.pop("sort", False):
        raise NotImplementedError(
            "sort=True is not supported by JoinOp."
        )

    unsupported = [
        k for k in params
        if k not in _JOIN_OP_FIELDS and k not in ("right", "other")
    ]
    if unsupported:
        raise NotImplementedError(
            f"Unsupported arguments for {op.method_name}(): {', '.join(sorted(unsupported))}"
        )

    join_kwargs = {k: v for k, v in params.items() if k in _JOIN_OP_FIELDS}

    if is_chained:
        return _make_chained_join_op(op, join_kwargs)

    new_op = JoinOp(**join_kwargs, inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op


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
