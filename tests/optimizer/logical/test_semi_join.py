"""`isin` against a relation becomes a semi/anti join (#196)."""
import unittest

import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.logical.rewrites._relational import promote_isin_to_filtering_join
from stratum.optimizer.logical._join_ops import FilteringJoinOp, JoinOp
from stratum.optimizer.logical._ops import Op, OutputType
from stratum.optimizer.logical._selection_ops import MaskSelectionOp, SelectionOp
from stratum.optimizer.physical._impl_selection import bind_op, FlagBasedSelector
from stratum.optimizer.physical._join_execs import (
    PandasIsInSemiJoinOp, PandasJoinOp, PandasMergeSemiJoinOp,
    PolarsFilteringJoinOp)
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.logical._source_ops import InMemoryFrameOp
from .test_dataframe_ops import (
    force_polars, optimize, run_op)


def _ctx():
    return PlanContext(pandas_query=False, rechunk=False,
                       parallelism=1, rust_backend=False, allow_patch=False)


def _run(ops):
    pool = {}
    for op in ops:
        pool[id(op)] = op.process("fit_transform",
                                  [pool[id(i)] for i in op.inputs])
    return pool[id(ops[-1])]


class TestFilteringJoinConfig(unittest.TestCase):
    """Semi and anti joins are their own operator, with no suffixes to carry."""

    def test_a_regular_join_refuses_a_filtering_how(self):
        for how in ("semi", "anti"):
            with self.subTest(how=how), self.assertRaises(ValueError) as cm:
                JoinOp(how=how, left_on=["c"])
            self.assertIn("FilteringJoinOp", str(cm.exception))

    def test_a_regular_join_keeps_its_suffixes(self):
        self.assertEqual(("_x", "_y"), JoinOp(how="inner").suffixes)

    def test_a_filtering_join_is_in_the_join_family(self):
        self.assertEqual("Join", FilteringJoinOp.logical_family)

    def test_the_direction_is_visible_in_the_plan(self):
        # "Join" alone cannot tell a semi from an anti.
        self.assertIn("semi", str(FilteringJoinOp(left_on="c")))
        self.assertIn("anti", str(FilteringJoinOp(anti=True, left_on="c")))

    def test_clone_round_trips(self):
        clone = FilteringJoinOp(anti=True, left_on="c").clone()
        self.assertEqual((True, "c", None), (clone.anti, clone.left_on, clone.right_on))

    def test_the_schema_is_the_left_one(self):
        # A filtering join adds no columns, whatever the build side carries, and
        # it has no suffixes to apply to a shared name.
        left, build = Op(), Op()
        left.output_schema = pl.Schema({"c": pl.String, "v": pl.Int64})
        build.output_schema = pl.Schema({"c": pl.String, "z": pl.Int64})
        for anti, left_on, right_on in ((False, "c", None), (True, ["c"], ["c"])):
            with self.subTest(anti=anti):
                op = FilteringJoinOp(anti=anti, left_on=left_on, right_on=right_on,
                                     inputs=[left, build])
                op.propagate_output_schema()
                self.assertEqual(left.output_schema, op.output_schema)

    def test_semi_and_anti_do_not_share_a_structure_key(self):
        self.assertNotEqual(FilteringJoinOp(left_on="c").structure_key()[2],
                            FilteringJoinOp(anti=True, left_on="c").structure_key()[2])


class TestFilteringJoinExecution(unittest.TestCase):
    """Both backends match the plain pandas equivalent."""

    def setUp(self):
        self.left = pd.DataFrame({"c": ["a", "b", "c", "a", "d"],
                                  "v": [1, 2, 3, 4, 5]})
        # Duplicated on purpose: de-duplicating the build side is what makes this
        # a semi-join and not an inner join, so a repeated key must not double a
        # left row.
        self.build = pd.Series(["a", "c", "a"], name="c")

    def _expected(self, how):
        mask = self.left["c"].isin(self.build)
        return self.left[mask if how == "semi" else ~mask]

    def test_single_key_matches_pandas_in_both_backends(self):
        for how in ("semi", "anti"):
            with self.subTest(how=how):
                expected = self._expected(how)
                got = run_op(FilteringJoinOp(anti=how == "anti", left_on="c"), self.left, self.build)
                pd.testing.assert_frame_equal(got, expected)
                with force_polars():
                    out = run_op(FilteringJoinOp(anti=how == "anti", left_on="c"),
                                 pl.DataFrame(self.left),
                                 pl.Series("c", self.build.tolist()))
                self.assertEqual(list(expected["v"]), out["v"].to_list())

    def test_a_repeated_build_key_does_not_duplicate_left_rows(self):
        got = run_op(FilteringJoinOp(left_on="c"), self.left, self.build)
        self.assertEqual(len(self._expected("semi")), len(got))

    def test_a_composite_key_matches_pandas_in_both_backends(self):
        left = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "y", "x", "z"],
                             "v": [1, 2, 3, 4]})
        build = pd.DataFrame({"p": [1, 1, 2], "q": ["x", "x", "x"]})
        keys = set(zip(build["p"], build["q"]))
        hit = pd.Series([(a, b) in keys for a, b in zip(left["a"], left["b"])])
        for how, mask in (("semi", hit.to_numpy()), ("anti", ~hit.to_numpy())):
            with self.subTest(how=how):
                expected = left[mask]
                got = run_op(FilteringJoinOp(anti=how == "anti", left_on=["a", "b"],
                                             right_on=["p", "q"]), left, build)
                pd.testing.assert_frame_equal(got, expected)
                with force_polars():
                    out = run_op(FilteringJoinOp(anti=how == "anti", left_on=["a", "b"],
                                                 right_on=["p", "q"]),
                                 pl.DataFrame(left), pl.DataFrame(build))
                self.assertEqual(list(expected["v"]), out["v"].to_list())


class TestFilteringJoinImplSelection(unittest.TestCase):
    """The two pandas impls split on key arity, not on a cost guess."""

    def test_a_single_key_goes_through_isin(self):
        op = FilteringJoinOp(left_on="c")
        bind_op(op, _ctx())
        self.assertIsInstance(op, PandasIsInSemiJoinOp)

    def test_a_composite_key_goes_through_the_merge(self):
        # `isin` tests one column against one sequence, so it cannot express a
        # composite key at all.
        op = FilteringJoinOp(left_on=["a", "b"], right_on=["p", "q"])
        bind_op(op, _ctx())
        self.assertIsInstance(op, PandasMergeSemiJoinOp)

    def test_polars_runs_either_arity_natively(self):
        for left_on, right_on in (("c", None), (["a", "b"], ["p", "q"])):
            with self.subTest(left_on=left_on):
                op = FilteringJoinOp(anti=True, left_on=left_on, right_on=right_on)
                bind_op(op, _ctx(),
                        selector=FlagBasedSelector(backend="polars"))
                self.assertIsInstance(op, PolarsFilteringJoinOp)

    def test_the_ordinary_join_impls_are_not_candidates(self):
        # `merge` has no semi/anti spelling, so only the filtering impls register.
        from stratum.optimizer.physical._registry import get_default_physical_registry
        impls = {c.impl_class for c in
                 get_default_physical_registry().candidates_for(FilteringJoinOp)}
        self.assertNotIn(PandasJoinOp, impls)
        self.assertIn(PandasIsInSemiJoinOp, impls)


class TestSemiJoinPromotion(unittest.TestCase):
    """Only `isin` against a graph-fed relation is promoted."""

    def setUp(self):
        self.df = pd.DataFrame({"c": ["a", "b", "c", "a", "d"],
                                "v": [1, 2, 3, 4, 5]})
        self.other = pd.DataFrame({"c": ["a", "c", "a"], "z": [9, 9, 9]})

    def _plan(self, build, **config):
        return optimize(build(), OptConfig(dataframe_ops=True, **config))

    def _joins(self, ops):
        return [o for o in ops if isinstance(o, (JoinOp, FilteringJoinOp))]

    def test_a_literal_collection_stays_a_selection(self):
        # A membership test against a constant is a predicate, not a join.
        data = st.as_data_op(self.df)
        ops = self._plan(lambda: data[data["c"].isin(["a", "b"])])
        self.assertEqual([], self._joins(ops))
        self.assertEqual(1, sum(isinstance(o, SelectionOp) for o in ops))
        pd.testing.assert_frame_equal(_run(ops),
                                      self.df[self.df["c"].isin(["a", "b"])])

    def test_a_relation_becomes_a_semi_join(self):
        data, other = st.as_data_op(self.df), st.as_data_op(self.other)
        ops = self._plan(lambda: data[data["c"].isin(other["c"])])
        joins = self._joins(ops)
        self.assertEqual(1, len(joins))
        self.assertIsInstance(joins[0], FilteringJoinOp)
        self.assertFalse(joins[0].anti)
        self.assertEqual("c", joins[0].left_on)
        # The build side is the key sequence itself, with no column to name.
        self.assertIsNone(joins[0].right_on)
        pd.testing.assert_frame_equal(
            _run(ops), self.df[self.df["c"].isin(self.other["c"])])

    def test_a_negated_relation_becomes_an_anti_join(self):
        data, other = st.as_data_op(self.df), st.as_data_op(self.other)
        ops = self._plan(lambda: data[~data["c"].isin(other["c"])])
        self.assertTrue(self._joins(ops)[0].anti)
        pd.testing.assert_frame_equal(
            _run(ops), self.df[~self.df["c"].isin(self.other["c"])])

    def test_replacing_a_selection_does_not_skip_downstream_rewrites(self):
        data, other = st.as_data_op(self.df), st.as_data_op(self.other)
        first = data[data["c"].isin(other["c"])]
        second = first[~first["c"].isin(other[other["c"] == "a"]["c"])]
        ops = self._plan(lambda: second)
        self.assertEqual([False, True], [op.anti for op in self._joins(ops)])
        expected = self.df[self.df["c"].isin(self.other["c"])]
        expected = expected[~expected["c"].isin(self.other[self.other["c"] == "a"]["c"])]
        pd.testing.assert_frame_equal(_run(ops), expected)

    def test_a_shared_key_column_is_still_promoted(self):
        # The column being filtered is also read by something else, so it stays
        # an operand instead of folding into a Col. It is still a column of the
        # left relation, which is the shape the driving pipeline produces.
        data, other = st.as_data_op(self.df), st.as_data_op(self.other)
        column = data["c"]
        ops = self._plan(lambda: data[column.isin(other["c"])].assign(
            k=column.str.upper()))
        joins = self._joins(ops)
        self.assertEqual(1, len(joins))
        self.assertEqual("c", joins[0].left_on)

    def test_the_rewrite_can_be_switched_off(self):
        data, other = st.as_data_op(self.df), st.as_data_op(self.other)
        ops = self._plan(lambda: data[data["c"].isin(other["c"])],
                         semi_join_rewrite=False)
        self.assertEqual([], self._joins(ops))
        masks = [o for o in ops if isinstance(o, MaskSelectionOp)]
        self.assertEqual(1, len(masks))

    def test_a_filtered_series_is_not_promoted(self):
        # There is no left relation to join, only a predicate over values.
        op = MaskSelectionOp()
        op.output_type = OutputType.SERIES
        self.assertIs(op, promote_isin_to_filtering_join(op))

    def test_a_non_membership_predicate_is_not_promoted(self):
        data = st.as_data_op(self.df)
        ops = self._plan(lambda: data[data["v"] > 2])
        self.assertEqual([], self._joins(ops))


class TestDrivingPipeline(unittest.TestCase):
    """The pipeline from #193, end to end."""

    def setUp(self):
        self.df = pd.DataFrame({
            "t": ["c"] * 2 + ["a"] * 5 + ["b"] * 4 + ["c"] + ["d"] + ["e"] * 3,
            "x": range(16),
        })

    def _build(self, frame):
        target = frame["t"]
        counts = target.value_counts()
        eligible = counts[counts >= 3].index
        return frame[target.isin(eligible)].reset_index(drop=True)

    def test_it_plans_as_a_semi_join_and_matches_pandas(self):
        ops = optimize(self._build(st.as_data_op(self.df)),
                       OptConfig(dataframe_ops=True))
        joins = [o for o in ops if isinstance(o, (JoinOp, FilteringJoinOp))]
        self.assertEqual(1, len(joins))
        self.assertIsInstance(joins[0], FilteringJoinOp)
        self.assertFalse(joins[0].anti)
        pd.testing.assert_frame_equal(_run(ops), self._build(self.df))

    def test_the_two_reads_of_the_source_share_one_scan(self):
        # The main optimization payoff: the plan counts and filters the same
        # relation, and reads it once.
        ops = optimize(self._build(st.as_data_op(self.df)),
                       OptConfig(dataframe_ops=True))
        sources = [o for o in ops if isinstance(o, InMemoryFrameOp)]
        self.assertEqual(1, len(sources))
        self.assertEqual(2, len(sources[0].outputs),
                         "the scan should feed both the count and the join")


if __name__ == "__main__":
    unittest.main()
