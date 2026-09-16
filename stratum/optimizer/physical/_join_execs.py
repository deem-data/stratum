"""Physical implementations of ``JoinOp`` (pandas merge / polars join).

Same-shape backend-variant family: the concrete impls subclass the logical
``JoinOp`` (plus :class:`PhysicalOp`), so ``isinstance(op, JoinOp)`` still
identifies a join anywhere in the plan. Selection swaps a logical ``JoinOp`` to
one of these per the plan context.
"""
from __future__ import annotations

from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


@physical_impl(of=JoinOp, backend="pandas")
class PandasJoinOp(JoinOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        if len(inputs) != 2:
            raise ValueError(f"JoinOp expects exactly 2 inputs (left and right dataframes), got {len(inputs)}.")
        left_df, right_df = inputs
        return left_df.merge(
            right_df,
            left_on=self.left_on,
            right_on=self.right_on,
            how=self.how,
            suffixes=self.suffixes,
            left_index=self.left_index,
            right_index=self.right_index,
        )


@physical_impl(of=JoinOp, backend="polars")
class PolarsJoinOp(JoinOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        if len(inputs) != 2:
            raise ValueError(f"JoinOp expects exactly 2 inputs (left and right dataframes), got {len(inputs)}.")
        left_df, right_df = inputs
        if self.left_index or self.right_index:
            raise NotImplementedError("JoinOp Polars backend does not support index-based joins.")
        if self.how not in ("inner", "left", "outer"):
            raise NotImplementedError(
                f"JoinOp Polars backend does not support how={self.how!r}.")
        no_defined_join_columns = self.left_on is None and self.right_on is None
        if not no_defined_join_columns and not isinstance(self.left_on, (str, list, tuple)):
            raise NotImplementedError(
                f"JoinOp Polars backend does not support left_on of type "
                f"{type(self.left_on).__name__}.")

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
            coalesce=self.left_on == self.right_on,  # keep distinct key-rows, drop identical ones
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
