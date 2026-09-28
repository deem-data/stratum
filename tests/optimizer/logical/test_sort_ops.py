import unittest

import pandas as pd
import polars as pl

from stratum.optimizer.logical._ops import OutputType
from stratum.optimizer.logical._sort_ops import SortOp
from .test_dataframe_ops import force_polars, run_op


def _series_sort(**kwargs) -> SortOp:
    """A sort with no key, i.e. ordering a series by its own values."""
    op = SortOp(**kwargs)
    op.output_type = OutputType.SERIES
    return op


class TestSortOpConfig(unittest.TestCase):
    """Field handling and the rendered name."""

    def test_defaults_sort_the_values_ascending(self):
        op = SortOp()
        self.assertEqual((), op.by)
        self.assertTrue(op.ascending)
        self.assertEqual("last", op.na_position)
        self.assertIn("values asc", op.name)

    def test_name_renders_every_key_with_its_direction(self):
        self.assertIn("count desc", SortOp(by=("count",), ascending=False).name)
        self.assertIn("a asc, b desc",
                      SortOp(by=("a", "b"), ascending=(True, False)).name)

    def test_lists_are_stored_as_tuples(self):
        # `clone_value` recurses tuples but passes a list through by reference.
        op = SortOp(by=["a", "b"], ascending=[True, False])
        self.assertEqual(("a", "b"), op.by)
        self.assertEqual((True, False), op.ascending)

    def test_clone_round_trips_the_config(self):
        op = SortOp(by=("count",), ascending=False, na_position="first")
        clone = op.clone()
        self.assertEqual(op.by, clone.by)
        self.assertEqual(op.ascending, clone.ascending)
        self.assertEqual(op.na_position, clone.na_position)

    def test_same_config_shares_a_structure_key(self):
        a, b = SortOp(by=("count",), ascending=False), SortOp(by=("count",),
                                                             ascending=False)
        self.assertEqual(a.structure_key()[2], b.structure_key()[2])
        self.assertNotEqual(a.structure_key()[2],
                            SortOp(by=("count",)).structure_key()[2])


class TestSortExecution(unittest.TestCase):
    """Both backends order rows the same way, ties included."""

    def setUp(self):
        # `c` and `b` tie on count, and `c` appears first, so a tie broken by
        # position is distinguishable from one broken by key.
        self.data = {"k": ["c", "a", "b", "d"], "count": [2, 3, 2, 1]}
        self.pdf = pd.DataFrame(self.data)
        self.pldf = pl.DataFrame(self.data)

    def test_frame_sorts_by_key_in_both_backends(self):
        op = SortOp(by=("count",), ascending=False)
        expected = ["a", "c", "b", "d"]
        self.assertEqual(expected, list(run_op(op, self.pdf)["k"]))
        with force_polars():
            self.assertEqual(expected,
                             run_op(SortOp(by=("count",), ascending=False),
                                    self.pldf)["k"].to_list())

    def test_series_sorts_by_its_own_values_in_both_backends(self):
        counts = pd.Series([2, 3, 2, 1], index=self.data["k"], name="count")
        got = run_op(_series_sort(ascending=False), counts)
        self.assertEqual(["a", "c", "b", "d"], list(got.index))
        with force_polars():
            out = run_op(_series_sort(ascending=False),
                         pl.Series("count", [2, 3, 2, 1]))
        self.assertEqual([3, 2, 2, 1], out.to_list())

    def test_ties_keep_input_order_in_both_backends(self):
        # An unstable sort is free to reorder ties, which would make the two
        # backends disagree on a result they both call correct.
        op = SortOp(by=("count",), ascending=False)
        pandas_order = list(run_op(op, self.pdf)["k"])
        with force_polars():
            polars_order = run_op(SortOp(by=("count",), ascending=False),
                                  self.pldf)["k"].to_list()
        self.assertEqual(pandas_order, polars_order)
        self.assertEqual(["c", "b"], [k for k in pandas_order if k in ("b", "c")])

    def test_multiple_keys_take_one_direction_each(self):
        data = {"a": [1, 1, 2], "b": [5, 7, 6]}
        got = run_op(SortOp(by=("a", "b"), ascending=(True, False)),
                     pd.DataFrame(data))
        self.assertEqual([7, 5, 6], list(got["b"]))
        with force_polars():
            out = run_op(SortOp(by=("a", "b"), ascending=(True, False)),
                         pl.DataFrame(data))
        self.assertEqual([7, 5, 6], out["b"].to_list())

    def test_na_position_agrees_across_backends(self):
        data = {"v": [2.0, None, 1.0]}
        for na_position, null_row in (("last", 2), ("first", 0)):
            with self.subTest(na_position=na_position):
                # polars puts nulls first by default and pandas last, so the two
                # only agree because the op always states the position.
                got = run_op(SortOp(by=("v",), na_position=na_position),
                             pd.DataFrame(data))
                self.assertEqual([null_row], list(got["v"].isna().to_numpy().nonzero()[0]))
                with force_polars():
                    out = run_op(SortOp(by=("v",), na_position=na_position),
                                 pl.DataFrame(data))
                self.assertEqual([null_row],
                                 [i for i, v in enumerate(out["v"].to_list())
                                  if v is None])


if __name__ == "__main__":
    unittest.main()
