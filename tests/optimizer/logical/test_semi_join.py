"""`isin` against a relation becomes a semi/anti join (#196)."""
import unittest

import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.logical._relational_rewrites import promote_isin_to_filtering_join
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._ops import Op, OutputType
from stratum.optimizer.logical._selection_ops import SelectionKind, SelectionOp
from stratum.optimizer.physical._impl_selection import bind_op, FlagBasedSelector
from stratum.optimizer.physical._join_execs import (
    PandasIsInSemiJoinOp, PandasJoinOp, PandasMergeSemiJoinOp,
    PolarsFilteringJoinOp)
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._source_execs import InMemoryFrame
from .test_dataframe_ops import (
    force_polars, optimize, run_op)


def _ctx(backend="pandas"):
    return PlanContext(backend=backend, pandas_query=False, rechunk=False,
                       parallelism=1, rust_backend=False, allow_patch=False)


def _run(ops):
    pool = {}
    for op in ops:
        pool[id(op)] = op.process("fit_transform",
                                  [pool[id(i)] for i in op.inputs])
    return pool[id(ops[-1])]


class TestFilteringJoinConfig(unittest.TestCase):
    """`how="semi"` / `"anti"` carry no suffixes."""

    def test_suffixes_are_dropped(self):
        for how in ("semi", "anti"):
            with self.subTest(how=how):
                self.assertIsNone(JoinOp(how=how, left_on="c").suffixes)

    def test_an_explicit_suffix_pair_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            JoinOp(how="semi", left_on="c", suffixes=("_l", "_r"))
        self.assertIn("suffixes", str(cm.exception))

    def test_a_regular_join_keeps_its_suffixes(self):
        self.assertEqual(("_x", "_y"), JoinOp(how="inner").suffixes)

    def test_the_direction_is_visible_in_the_plan(self):
        # "Join" alone cannot tell a semi from an anti.
        self.assertIn("semi", str(JoinOp(how="semi", left_on="c")))
        self.assertIn("anti", str(JoinOp(how="anti", left_on="c")))

    def test_clone_round_trips(self):
        clone = JoinOp(how="anti", left_on="c").clone()
        self.assertEqual("anti", clone.how)
        self.assertIsNone(clone.suffixes)

    def test_the_schema_is_the_left_one(self):
        # A filtering join adds no columns, whatever the build side carries, and
        # it has no suffixes to apply to a shared name.
        left, build = Op(), Op()
        left.output_schema = pl.Schema({"c": pl.String, "v": pl.Int64})
        build.output_schema = pl.Schema({"c": pl.String, "z": pl.Int64})
        for how, left_on, right_on in (("semi", "c", None), ("anti", ["c"], ["c"])):
            with self.subTest(how=how):
                op = JoinOp(how=how, left_on=left_on, right_on=right_on,
                            inputs=[left, build])
                op.propagate_output_schema()
                self.assertEqual(left.output_schema, op.output_schema)

    def test_semi_and_anti_do_not_share_a_structure_key(self):
        self.assertNotEqual(JoinOp(how="semi", left_on="c").structure_key()[2],
                            JoinOp(how="anti", left_on="c").structure_key()[2])


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
                got = run_op(JoinOp(how=how, left_on="c"), self.left, self.build)
                pd.testing.assert_frame_equal(got, expected)
                with force_polars():
                    out = run_op(JoinOp(how=how, left_on="c"),
                                 pl.DataFrame(self.left),
                                 pl.Series("c", self.build.tolist()))
                self.assertEqual(list(expected["v"]), out["v"].to_list())

    def test_a_repeated_build_key_does_not_duplicate_left_rows(self):
        got = run_op(JoinOp(how="semi", left_on="c"), self.left, self.build)
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
                got = run_op(JoinOp(how=how, left_on=["a", "b"],
                                    right_on=["p", "q"]), left, build)
                pd.testing.assert_frame_equal(got, expected)
                with force_polars():
                    out = run_op(JoinOp(how=how, left_on=["a", "b"],
                                        right_on=["p", "q"]),
                                 pl.DataFrame(left), pl.DataFrame(build))
                self.assertEqual(list(expected["v"]), out["v"].to_list())


class TestFilteringJoinImplSelection(unittest.TestCase):
    """The two pandas impls split on key arity, not on a cost guess."""

    def test_a_single_key_goes_through_isin(self):
        op = JoinOp(how="semi", left_on="c")
        bind_op(op, _ctx())
        self.assertIsInstance(op, PandasIsInSemiJoinOp)

    def test_a_composite_key_goes_through_the_merge(self):
        # `isin` tests one column against one sequence, so it cannot express a
        # composite key at all.
        op = JoinOp(how="semi", left_on=["a", "b"], right_on=["p", "q"])
        bind_op(op, _ctx())
        self.assertIsInstance(op, PandasMergeSemiJoinOp)

    def test_polars_runs_either_arity_natively(self):
        for left_on, right_on in (("c", None), (["a", "b"], ["p", "q"])):
            with self.subTest(left_on=left_on):
                op = JoinOp(how="anti", left_on=left_on, right_on=right_on)
                # Pinned to the context backend; the production selectors rank
                # backends without consulting it (see #204).
                bind_op(op, _ctx(backend="polars"),
                        selector=FlagBasedSelector())
                self.assertIsInstance(op, PolarsFilteringJoinOp)

    def test_the_ordinary_join_impls_refuse_a_filtering_join(self):
        # `merge` has no semi/anti spelling.
        self.assertFalse(PandasJoinOp.supports(JoinOp(how="semi", left_on="c"),
                                               _ctx()))
        self.assertTrue(PandasJoinOp.supports(JoinOp(how="inner"), _ctx()))


class TestSemiJoinPromotion(unittest.TestCase):
    """Only `isin` against a graph-fed relation is promoted."""

    def setUp(self):
        self.df = pd.DataFrame({"c": ["a", "b", "c", "a", "d"],
                                "v": [1, 2, 3, 4, 5]})
        self.other = pd.DataFrame({"c": ["a", "c", "a"], "z": [9, 9, 9]})

    def _plan(self, build, **config):
        return optimize(build(), OptConfig(dataframe_ops=True, **config))

    def _joins(self, ops):
        return [o for o in ops if isinstance(o, JoinOp)]

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
        self.assertEqual("semi", joins[0].how)
        self.assertEqual("c", joins[0].left_on)
        # The build side is the key sequence itself, with no column to name.
        self.assertIsNone(joins[0].right_on)
        pd.testing.assert_frame_equal(
            _run(ops), self.df[self.df["c"].isin(self.other["c"])])

    def test_a_negated_relation_becomes_an_anti_join(self):
        data, other = st.as_data_op(self.df), st.as_data_op(self.other)
        ops = self._plan(lambda: data[~data["c"].isin(other["c"])])
        self.assertEqual("anti", self._joins(ops)[0].how)
        pd.testing.assert_frame_equal(
            _run(ops), self.df[~self.df["c"].isin(self.other["c"])])

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
        masks = [o for o in ops if isinstance(o, SelectionOp)
                 and o.kind is SelectionKind.MASK]
        self.assertEqual(1, len(masks))

    def test_a_filtered_series_is_not_promoted(self):
        # There is no left relation to join, only a predicate over values.
        op = SelectionOp(kind=SelectionKind.MASK)
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
        joins = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(joins))
        self.assertEqual("semi", joins[0].how)
        pd.testing.assert_frame_equal(_run(ops), self._build(self.df))

    def test_the_two_reads_of_the_source_share_one_scan(self):
        # The main optimization payoff: the plan counts and filters the same
        # relation, and reads it once.
        ops = optimize(self._build(st.as_data_op(self.df)),
                       OptConfig(dataframe_ops=True))
        sources = [o for o in ops if isinstance(o, InMemoryFrame)]
        self.assertEqual(1, len(sources))
        self.assertEqual(2, len(sources[0].outputs),
                         "the scan should feed both the count and the join")


if __name__ == "__main__":
    unittest.main()
