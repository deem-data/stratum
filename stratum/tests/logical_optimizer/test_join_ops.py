import unittest

import pandas as pd
import polars as pl
import pytest
import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.ir._join_ops import JoinOp
from stratum.runtime._buffer_pool import BufferPool
from stratum.tests.logical_optimizer.test_dataframe_ops import (
    force_polars, optimize, run_op)


def to_list(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.tolist()
    if isinstance(obj, (pl.Series, pl.DataFrame)):
        return obj.to_list()
    return obj


def make_frame(data, polars):
    return pl.DataFrame(data) if polars else pd.DataFrame(data)


@pytest.mark.parametrize("polars", [False, True])
def test_join_op_merge_on_key(polars):
    left = make_frame({"k": [1, 2, 3], "a": [10, 20, 30]}, polars)
    right = make_frame({"k": [2, 3, 4], "b": [200, 300, 400]}, polars)
    op = JoinOp(how="inner", left_on="k", right_on="k")
    with st.config(force_polars=polars):
        result = run_op(op, left, right)

    assert [2, 3] == to_list(result["k"])
    assert [20, 30] == to_list(result["a"])
    assert [200, 300] == to_list(result["b"])


@pytest.mark.parametrize("polars", [False, True])
def test_join_op_merge_left_on_right_on_distinct(polars):
    left = make_frame({"lk": [1, 2], "a": [10, 20]}, polars)
    right = make_frame({"rk": [2, 3], "b": [200, 300]}, polars)
    op = JoinOp(how="inner", left_on="lk", right_on="rk")
    with st.config(force_polars=polars):
        result = run_op(op, left, right)

    assert [2] == to_list(result["lk"]) == to_list(result["rk"])
    assert [200] == to_list(result["b"])


@pytest.mark.parametrize("polars", [False, True])
def test_join_op_merge_on_key_with_suffixes(polars):
    left = make_frame({"k": [1, 2, 3], "a": [10, 20, 30]}, polars)
    right = make_frame({"k": [2, 3, 4], "a": [200, 300, 400]}, polars)
    op = JoinOp(how="inner", left_on="k", right_on="k",
                suffixes=("_L", "_R"))
    with st.config(force_polars=polars):
        result = run_op(op, left, right)

    assert [2, 3] == to_list(result["k"])
    assert [20, 30] == to_list(result["a_L"])
    assert [200, 300] == to_list(result["a_R"])


@pytest.mark.parametrize("polars", [False, True])
def test_outer_join(polars):
    left = make_frame({"k": [1, 2, 3], "a": [10, 20, 30]}, polars)
    right = make_frame({"k": [2, 3, 4], "a": [200, 300, 400]}, polars)

    op = JoinOp(how="outer", left_on="k", right_on="k")
    with st.config(force_polars=polars):
        result = run_op(op, left, right)

    if polars:                  # this is needed since row order after an outer-join is non deterministic
        result = result.sort("k")
    else:
        result = result.sort_values("k").reset_index(drop=True)

    assert [1, 2, 3, 4] == to_list(result["k"])

    a_x = to_list(result["a_x"])
    a_y = to_list(result["a_y"])

    assert a_x[:3] == [10, 20, 30]
    assert a_y[1:] == [200, 300, 400]

    if polars:     # polars uses None to represent missing values compared to nan in pandas case
        assert a_x[3] is None
        assert a_y[0] is None
    else:
        assert pd.isna(a_x[3])
        assert pd.isna(a_y[0])
    
@pytest.mark.parametrize("polars", [False, True])
def test_multi_column_join_key(polars):
    left = make_frame({
        "k1": [1, 1, 2],
        "k2": ["a", "b", "a"],
        "a": [10, 11, 20],
        "b": ["L1", "L2", "L3"],
    }, polars)
    right = make_frame({
        "k1": [1, 2],
        "k2": ["a", "a"],
        "a": [100, 200],
        "c": ["R1", "R2"],
    }, polars)

    op = JoinOp(how="inner", left_on=["k1", "k2"], right_on=["k1", "k2"])
    with st.config(force_polars=polars):
        result = run_op(op, left, right)

    assert ["k1", "k2", "a_x", "b", "a_y", "c"] == list(result.columns)
    assert [1, 2] == to_list(result["k1"])
    assert ["a", "a"] == to_list(result["k2"])
    assert [10, 20] == to_list(result["a_x"])
    assert [100, 200] == to_list(result["a_y"])
    assert ["L1", "L3"] == to_list(result["b"])
    assert ["R1", "R2"] == to_list(result["c"])


@pytest.mark.parametrize("polars", [False, True])
def test_wo_key_join(polars):
    left = make_frame({
        "k": [1, 2, 3],
        "a": [10, 20, 30],
    }, polars)
    right = make_frame({
        "k": [2, 3, 4],
        "b": [200, 300, 400],
    }, polars)

    op = JoinOp(how="inner")
    with st.config(force_polars=polars):
        result = run_op(op, left, right)

    assert ["k", "a", "b"] == list(result.columns)
    assert [2, 3] == to_list(result["k"])
    assert [20, 30] == to_list(result["a"])
    assert [200, 300] == to_list(result["b"])


@pytest.mark.parametrize("polars", [False, True])
def test_join_op_wrong_input_count_raises(polars):
    op = JoinOp(how="inner", left_on="k", right_on="k")
    data = make_frame({"k": [1]}, polars)
    with st.config(force_polars=polars):
        with pytest.raises(ValueError, match="expects exactly 2 inputs"):
            run_op(op, data)


class TestJoinOpPandas(unittest.TestCase):
    """Pandas-only `JoinOp.process` index join coverage."""

    def test_join_index_based_with_suffixes(self):
        left = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])
        right = pd.DataFrame({"x": [10, 20, 30]}, index=["b", "c", "d"])
        op = JoinOp(how="left", left_index=True, right_index=True,
                    suffixes=("_L", "_R"))
        result = run_op(op, left, right)
        self.assertEqual(["a", "b", "c"], result.index.tolist())
        self.assertIn("x_L", result.columns)
        self.assertIn("x_R", result.columns)

    def test_outer_join(self):
        left = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        right = pd.DataFrame({"b": [10, 20]}, index=["y", "z"])
        op = JoinOp(how="outer", left_index=True, right_index=True)
        result = run_op(op, left, right)
        self.assertEqual({"x", "y", "z"}, set(result.index.tolist()))

    def test_polars_index_based_join_raises(self):
        with force_polars():
            op = JoinOp(how="left", left_index=True, right_index=True)
            with self.assertRaises(NotImplementedError):
                run_op(op, pl.DataFrame({"k": [1]}), pl.DataFrame({"k": [1]}))

    def test_polars_unsupported_how_raises(self):
        # Only inner/left/outer are known to match the pandas backend; other
        # `how` values (e.g. cross/right) are rejected rather than diverging.
        for how in ("cross", "right"):
            with force_polars():
                op = JoinOp(how=how, left_on="k", right_on="k")
                with self.assertRaises(NotImplementedError):
                    run_op(op, pl.DataFrame({"k": [1]}), pl.DataFrame({"k": [1]}))

    def test_polars_non_str_list_key_raises(self):
        with force_polars():
            op = JoinOp(how="inner", left_on=0, right_on=0)
            with self.assertRaises(NotImplementedError):
                run_op(op, pl.DataFrame({"k": [1]}), pl.DataFrame({"k": [1]}))


class TestJoinRewrites(unittest.TestCase):
    """End-to-end: skrub DataOp expressions get rewritten to JoinOp(s)."""

    def _run_plan(self, ops):
        """Execute a linearized DAG and return the last op's output."""
        pool = BufferPool()
        for op in ops:
            inputs = [pool.pin(key) for key in op.inputs]
            pool.put(op, op.process("fit_transform", inputs))
        return pool.pin(ops[-1])

    def test_merge_on_key_rewrites_and_executes(self):
        df1 = pd.DataFrame({"k": [1, 2, 3], "a": [10, 20, 30]})
        df2 = pd.DataFrame({"k": [2, 3, 4], "b": [200, 300, 400]})
        data = st.as_data_op(df1).merge(st.as_data_op(df2), on="k")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual("k", join_ops[0].left_on)
        self.assertEqual("k", join_ops[0].right_on)
        self.assertEqual("inner", join_ops[0].how)

        result = self._run_plan(ops)
        expected = df1.merge(df2, on="k")
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_merge_left_on_right_on_preserved(self):
        df1 = pd.DataFrame({"lk": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"rk": [2, 3], "b": [200, 300]})
        data = st.as_data_op(df1).merge(
            st.as_data_op(df2), left_on="lk", right_on="rk", how="outer")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual("lk", join_ops[0].left_on)
        self.assertEqual("rk", join_ops[0].right_on)
        self.assertEqual("outer", join_ops[0].how)

    def test_merge_sort_true_raises(self):
        df1 = pd.DataFrame({"k": [1, 2]})
        df2 = pd.DataFrame({"k": [1, 2]})
        data = st.as_data_op(df1).merge(
            st.as_data_op(df2), on="k", sort=True)
        with self.assertRaises(NotImplementedError):
            optimize(data, OptConfig(dataframe_ops=True))

    def test_merge_sort_false_is_accepted(self):
        df1 = pd.DataFrame({"k": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"k": [1, 2], "b": [100, 200]})
        data = st.as_data_op(df1).merge(
            st.as_data_op(df2), on="k", sort=False)
        ops = optimize(data, OptConfig(dataframe_ops=True))
        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))

    def test_join_no_args_defaults_to_index_based_left(self):
        df1 = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        df2 = pd.DataFrame({"b": [10, 20]}, index=["x", "y"])
        data = st.as_data_op(df1).join(st.as_data_op(df2))
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual("left", join_ops[0].how)
        self.assertTrue(join_ops[0].left_index)
        self.assertTrue(join_ops[0].right_index)

    def test_join_with_on_uses_left_on_and_right_index(self):
        df1 = pd.DataFrame({"k": ["x", "y"], "a": [1, 2]})
        df2 = pd.DataFrame({"b": [10, 20]}, index=["x", "y"])
        data = st.as_data_op(df1).join(st.as_data_op(df2), on="k")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual("k", join_ops[0].left_on)
        self.assertFalse(join_ops[0].left_index)
        self.assertTrue(join_ops[0].right_index)

    def test_join_with_suffixes_rewrites_and_executes(self):
        df1 = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])
        df2 = pd.DataFrame({"x": [10, 20, 30]}, index=["b", "c", "d"])
        data = st.as_data_op(df1).join(
            st.as_data_op(df2), lsuffix="_L", rsuffix="_R")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertTrue(join_ops[0].left_index)
        self.assertTrue(join_ops[0].right_index)
        self.assertEqual(("_L", "_R"), join_ops[0].suffixes)

        result = self._run_plan(ops)
        expected = df1.join(df2, lsuffix="_L", rsuffix="_R")
        pd.testing.assert_frame_equal(result, expected)

    def test_merge_overlapping_non_key_columns_uses_pandas_default_suffixes(self):
        df1 = pd.DataFrame({"k": [1, 2], "v": [10, 20]})
        df2 = pd.DataFrame({"k": [1, 2], "v": [100, 200]})
        data = st.as_data_op(df1).merge(st.as_data_op(df2), on="k")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual(("_x", "_y"), join_ops[0].suffixes)

        result = self._run_plan(ops)
        expected = df1.merge(df2, on="k")
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True))
        self.assertIn("v_x", result.columns)
        self.assertIn("v_y", result.columns)

    def test_join_overlapping_columns_without_suffixes_raises(self):
        # Pandas .join() defaults both lsuffix and rsuffix to "", so overlapping
        # columns raise ValueError. JoinOp must reproduce that — not silently
        # invent suffixes like "_left"/"_right".
        df1 = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
        df2 = pd.DataFrame({"x": [10, 20]}, index=["a", "b"])
        with self.assertRaisesRegex(Exception, "columns overlap"):
            data = st.as_data_op(df1).join(st.as_data_op(df2))
            optimize(data, OptConfig(dataframe_ops=True))

    def test_join_overlapping_columns_with_suffixes_succeeds(self):
        # Sibling to the above: with suffixes provided, the same join works.
        df1 = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
        df2 = pd.DataFrame({"x": [10, 20]}, index=["a", "b"])
        data = st.as_data_op(df1).join(
            st.as_data_op(df2), lsuffix="_L", rsuffix="_R")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(("_L", "_R"), join_ops[0].suffixes)

        result = self._run_plan(ops)
        expected = df1.join(df2, lsuffix="_L", rsuffix="_R")
        pd.testing.assert_frame_equal(result, expected)

    def test_chained_join_decomposes_into_binary_chain(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        df2 = pd.DataFrame({"b": [10, 20, 30]}, index=["x", "y", "z"])
        df3 = pd.DataFrame({"c": [100, 200, 300]}, index=["x", "y", "z"])
        data = st.as_data_op(df1).join(
            [st.as_data_op(df2), st.as_data_op(df3)])
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(2, len(join_ops))
        # Both chain links are index-based.
        for j in join_ops:
            self.assertTrue(j.left_index)
            self.assertTrue(j.right_index)
        # Second JoinOp's left input is the first JoinOp.
        self.assertIs(join_ops[0], join_ops[1].inputs[0])

        result = self._run_plan(ops)
        expected = df1.join([df2, df3])
        pd.testing.assert_frame_equal(result, expected)

    def test_chained_join_with_duplicate_inputs_raises_error(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        df2 = pd.DataFrame(index=["x", "y", "z"])

        df2_op = st.as_data_op(df2)
        data = st.as_data_op(df1).join([df2_op, df2_op])
        with self.assertRaisesRegex(ValueError, "Duplicate right-hand frames in chained joins are not supported"):
            optimize(data, OptConfig(dataframe_ops=True))

    def test_join_with_other_kwarg(self):
        df1 = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        df2 = pd.DataFrame({"b": [10, 20]}, index=["x", "y"])
        data = st.as_data_op(df1).join(other=st.as_data_op(df2))
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))

        result = self._run_plan(ops)
        expected = df1.join(df2)
        pd.testing.assert_frame_equal(result, expected)

    def test_merge_unsupported_arguments_raises(self):
        df1 = pd.DataFrame({"k": [1, 2]})
        df2 = pd.DataFrame({"k": [1, 2]})
        data = st.as_data_op(df1).merge(
            st.as_data_op(df2), on="k", indicator=True
        )
        with self.assertRaisesRegex(NotImplementedError, "Unsupported arguments for merge"):
            optimize(data, OptConfig(dataframe_ops=True))

    def test_join_unsupported_arguments_raises(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [1, 2]})
        data = st.as_data_op(df1).join(
            st.as_data_op(df2), validate="one_to_one"
        )
        with self.assertRaisesRegex(NotImplementedError, "Unsupported arguments for join"):
            optimize(data, OptConfig(dataframe_ops=True))


if __name__ == "__main__":
    unittest.main()
