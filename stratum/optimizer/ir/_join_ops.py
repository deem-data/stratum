from collections.abc import Sequence
from stratum.optimizer.ir._ops import OperandRef, OutputType, MethodCallOp, Op
from stratum._config import FLAGS


class JoinOp(Op):
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
        super().__init__(name="", inputs=inputs, outputs=outputs)
        self.how = how
        self.left_on = left_on
        self.right_on = right_on
        self.left_index = left_index
        self.right_index = right_index
        self.suffixes = suffixes
        self.output_type = OutputType.FRAME

    def process(self, mode: str, inputs: list):
        if len(inputs) != 2:
            raise ValueError(f"JoinOp expects exactly 2 inputs (left and right dataframes), got {len(inputs)}.")
        left_df = inputs[0]
        right_df = inputs[1]

        if FLAGS.force_polars:
            if self.left_index or self.right_index:
                raise NotImplementedError("JoinOp Polars backend does not support index-based joins.")
            if self.how not in ("inner", "left", "outer"):
                raise NotImplementedError(
                    f"JoinOp Polars backend does not support how={self.how!r}."
                )
            no_defined_join_columns = self.left_on is None and self.right_on is None
            if not no_defined_join_columns and not isinstance(self.left_on, (str, list, tuple)):
                raise NotImplementedError(
                    f"JoinOp Polars backend does not support left_on of type "
                    f"{type(self.left_on).__name__}."
                )

            left_columns_list = list(left_df.columns)
            common_columns = [col for col in right_df.columns if col in left_columns_list]
            how = "full" if self.how == "outer" else self.how
            left_on = common_columns if no_defined_join_columns else self.left_on
            right_on = common_columns if no_defined_join_columns else self.right_on

            result = left_df.join(
                right_df,
                how=how,
                left_on=left_on,
                right_on=right_on,
                suffix=self.suffixes[1],
                coalesce = self.left_on == self.right_on # keep the different key-rows and get rid off identical ones 
            )
            if no_defined_join_columns:
                return result
            if isinstance(self.left_on, str):
                key_cols = {self.left_on, self.right_on}
            else:  # list/tuple, validated above
                key_cols = set(self.left_on) | set(self.right_on)
            mapping = {
                col: col + self.suffixes[0]
                for col in common_columns
                if col not in key_cols
            }
            return result.rename(mapping=mapping)
        else:
            return left_df.merge(
                right_df,
                left_on=self.left_on,
                right_on=self.right_on,
                how=self.how,
                suffixes=self.suffixes,
                left_index=self.left_index,
                right_index=self.right_index
            )


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
