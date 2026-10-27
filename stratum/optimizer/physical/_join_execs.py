"""Physical implementations of the join family (pandas merge / polars join).

Same-shape backend-variant families: the concrete impls subclass the logical
``JoinOp`` or ``FilteringJoinOp`` (plus :class:`PhysicalOp`), so ``isinstance``
on the logical class still identifies the op anywhere in the plan. Selection
swaps a logical op to one of these per the plan context.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.logical._join_ops import FilteringJoinOp, JoinOp
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
    #: The `how` values whose polars join matches pandas' merge.
    _HOW = ("inner", "left", "outer")

    @classmethod
    def supports(cls, op: JoinOp, ctx) -> bool:
        if op.left_index or op.right_index:
            op._unsupported_reason = "joins on index labels"
        elif op.how not in cls._HOW:
            op._unsupported_reason = f"how={op.how!r}"
        elif ((op.left_on is not None or op.right_on is not None)
              and not isinstance(op.left_on, (str, list, tuple))):
            op._unsupported_reason = f"left_on of type {type(op.left_on).__name__}"
        else:
            return True
        return False

    def process(self, mode: str, inputs: list):
        if len(inputs) != 2:
            raise ValueError(f"JoinOp expects exactly 2 inputs (left and right dataframes), got {len(inputs)}.")
        left_df, right_df = inputs
        no_defined_join_columns = self.left_on is None and self.right_on is None

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
        else:  # a list or tuple, as supports() admits
            key_cols = set(self.left_on) | set(self.right_on)
        mapping = {
            col: col + self.suffixes[0]
            for col in common_columns
            if col not in key_cols
        }
        return result.rename(mapping=mapping)


# --- FilteringJoinOp (semi / anti) --------------------------------------------
#
# Single-key joins admit both implementations. Composite keys require merge;
# de-duplicating the build keys preserves the multiplicity of each left row.


class FilteringJoinExec(FilteringJoinOp, PhysicalOp):
    """Physical base for the semi and anti joins."""

    @property
    def _keep_matches(self) -> bool:
        return not self.anti

@physical_impl(of=FilteringJoinOp, backend="pandas")
class PandasIsInSemiJoinOp(FilteringJoinExec):
    """Single-key filtering join through ``Series.isin``."""

    @classmethod
    def supports(cls, op: FilteringJoinOp, ctx) -> bool:
        single_left = (isinstance(op.left_on, str)
                       or isinstance(op.left_on, (list, tuple)) and len(op.left_on) == 1)
        single_right = (op.right_on is None or isinstance(op.right_on, str)
                        or isinstance(op.right_on, (list, tuple)) and len(op.right_on) == 1)
        return single_left and single_right

    def process(self, mode: str, inputs: list):
        left, build = inputs
        if self.right_on is not None:
            key = self.right_on if isinstance(self.right_on, str) else self.right_on[0]
            build = build[key]
        key = self.left_on if isinstance(self.left_on, str) else self.left_on[0]
        mask = left[key].isin(build)
        return left[mask if self._keep_matches else ~mask]

@physical_impl(of=FilteringJoinOp, backend="pandas")
class PandasMergeSemiJoinOp(FilteringJoinExec):
    """Filtering join through a merge on de-duplicated keys, at any arity."""

    @classmethod
    def supports(cls, op: FilteringJoinOp, ctx) -> bool:
        left_keys = [op.left_on] if isinstance(op.left_on, str) else op.left_on
        right_keys = [op.right_on] if isinstance(op.right_on, str) else op.right_on
        return (bool(left_keys)
                and (right_keys is None and len(left_keys) == 1
                     or right_keys is not None and len(left_keys) == len(right_keys)))

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


@physical_impl(of=FilteringJoinOp, backend="polars")
class PolarsFilteringJoinOp(FilteringJoinExec):
    """polars runs semi/anti natively, whatever the key arity."""

    _BUILD_KEY = "_stratum_semi_key"

    @classmethod
    def supports(cls, op: FilteringJoinOp, ctx) -> bool:
        return op.left_on is not None

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
        return left.join(build, left_on=left_on, right_on=right_on,
                         how="anti" if self.anti else "semi",
                         nulls_equal=True, maintain_order="left")
