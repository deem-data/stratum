"""Physical implementations of ``JoinOp`` (pandas merge / polars join).

Same-shape backend-variant family: the concrete impls subclass the logical
``JoinOp`` (plus :class:`PhysicalOp`), so ``isinstance(op, JoinOp)`` still
identifies a join anywhere in the plan. Selection swaps a logical ``JoinOp`` to
one of these per the plan context.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.logical._join_ops import FILTERING_JOINS, JoinOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl

@physical_impl(of=JoinOp, backend="pandas")
class PandasJoinOp(JoinOp, PhysicalOp):

    @classmethod
    def supports(cls, op: JoinOp, ctx) -> bool:
        # `merge` has no semi/anti spelling; the filtering impls below handle it.
        return op.how not in FILTERING_JOINS

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

    @classmethod
    def supports(cls, op: JoinOp, ctx) -> bool:
        return op.how not in FILTERING_JOINS

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


# --- Filtering joins (semi / anti) --------------------------------------------
#
# Single-key joins admit both implementations. Composite keys require merge;
# de-duplicating the build keys preserves the multiplicity of each left row.


class FilteringJoinExec(JoinOp, PhysicalOp):
    """Physical base for ``how="semi"`` / ``how="anti"``."""

    @property
    def _keep_matches(self) -> bool:
        return self.how == "semi"

@physical_impl(of=JoinOp, backend="pandas")
class PandasIsInSemiJoinOp(FilteringJoinExec):
    """Single-key filtering join through ``Series.isin``."""

    @classmethod
    def supports(cls, op: JoinOp, ctx) -> bool:
        single_left = (isinstance(op.left_on, str)
                       or isinstance(op.left_on, (list, tuple)) and len(op.left_on) == 1)
        single_right = (op.right_on is None or isinstance(op.right_on, str)
                        or isinstance(op.right_on, (list, tuple)) and len(op.right_on) == 1)
        return (op.how in FILTERING_JOINS and single_left and single_right
                and not op.left_index and not op.right_index)

    def process(self, mode: str, inputs: list):
        left, build = inputs
        if self.right_on is not None:
            key = self.right_on if isinstance(self.right_on, str) else self.right_on[0]
            build = build[key]
        key = self.left_on if isinstance(self.left_on, str) else self.left_on[0]
        mask = left[key].isin(build)
        return left[mask if self._keep_matches else ~mask]

@physical_impl(of=JoinOp, backend="pandas")
class PandasMergeSemiJoinOp(FilteringJoinExec):
    """Filtering join through a merge on de-duplicated keys, at any arity."""

    @classmethod
    def supports(cls, op: JoinOp, ctx) -> bool:
        left_keys = [op.left_on] if isinstance(op.left_on, str) else op.left_on
        right_keys = [op.right_on] if isinstance(op.right_on, str) else op.right_on
        return (op.how in FILTERING_JOINS and bool(left_keys)
                and (right_keys is None and len(left_keys) == 1
                     or right_keys is not None and len(left_keys) == len(right_keys))
                and not op.left_index and not op.right_index)

    def process(self, mode: str, inputs: list):
        left, build = inputs
        left_keys = [self.left_on] if isinstance(self.left_on, str) else list(self.left_on)
        right_keys = ([self.right_on] if isinstance(self.right_on, str)
                      else self.right_on)
        # Only keys enter the temporary merge, so user payload names (including
        # _merge) cannot collide with the indicator. Keep the original index.
        probe = left[left_keys].copy()
        keys = (pd.DataFrame({0: build.array if isinstance(build, (pd.Series, pd.Index))
                             else pd.Series(list(build), dtype=object)}) if right_keys is None
                else build[list(right_keys)].copy())
        probe.columns = keys.columns = list(range(len(left_keys)))
        # For a single key, handle missing sentinels with isin separately.
        # Otherwise distinct None/NaN/NA build values can multiply merge rows.
        merge_keys = keys.loc[keys[0].notna()] if len(left_keys) == 1 else keys
        merged = probe.merge(merge_keys.drop_duplicates(), how="left", sort=False,
                             on=list(probe.columns), indicator=True)
        matched = merged["_merge"].to_numpy() == "both"
        if len(left_keys) == 1:
            # pandas merge equates all missing sentinels, whereas isin on an
            # object Series distinguishes None, NaN and pd.NA.
            missing = probe[0].isna().to_numpy()
            matched[missing] = probe.loc[missing, 0].isin(keys[0]).to_numpy()
        return left[matched if self._keep_matches else ~matched]


@physical_impl(of=JoinOp, backend="polars")
class PolarsFilteringJoinOp(FilteringJoinExec):
    """polars runs semi/anti natively, whatever the key arity."""

    _BUILD_KEY = "_stratum_semi_key"

    @classmethod
    def supports(cls, op: JoinOp, ctx) -> bool:
        return (op.how in FILTERING_JOINS
                and not op.left_index and not op.right_index
                and op.left_on is not None)

    def process(self, mode: str, inputs: list):
        left, build = inputs
        left_on = [self.left_on] if isinstance(self.left_on, str) else list(self.left_on)
        if self.right_on is None:
            # A bare sequence of key values; polars joins relations, so give it
            # a one-column relation to join against.
            build = pl.DataFrame({self._BUILD_KEY: build})
            right_on = [self._BUILD_KEY]
        else:
            right_on = ([self.right_on] if isinstance(self.right_on, str)
                        else list(self.right_on))
        return left.join(build, left_on=left_on, right_on=right_on, how=self.how,
                         nulls_equal=True, maintain_order="left")
