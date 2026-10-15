import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal as assert_pandas_frame_equal
from polars.testing import assert_frame_equal as assert_polars_frame_equal

import stratum as st
from stratum._config import config
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._ops import OperandRef, OutputType
from stratum.optimizer.logical._selection_ops import (
    DropDuplicatesSelectionOp,
    DropLabelsSelectionOp,
    DropMissingSelectionOp,
    MaskSelectionOp,
    SampleSelectionOp,
    SliceSelectionOp,
)
from stratum.optimizer.physical._impl_selection import GreedyImplementationSelector, bind_op
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._registry import get_default_physical_registry
from stratum.optimizer.physical._selection_execs import (
    PandasDropDuplicatesSelectionOp,
    PandasDropLabelsSelectionOp,
    PandasDropMissingSelectionOp,
    PandasSampleSelectionOp,
    PandasSliceSelectionOp,
    PolarsDropDuplicatesSelectionOp,
    PolarsDropMissingSelectionOp,
    PolarsSampleSelectionOp,
    PolarsSeriesDropDuplicatesSelectionOp,
    PolarsSeriesDropMissingSelectionOp,
    PolarsSliceSelectionOp,
    get_polars_sample_kwargs,
)
from tests.optimizer.physical.test_source_execs import run_plan


def greedy_ctx():
    kwargs = {
        "pandas_query": False,
        "rechunk": True,
        "parallelism": 1,
        "rust_backend": False,
        "allow_patch": True,
        "implementation_selector": "greedy",
    }
    return PlanContext(**kwargs)


def greedy_bind(op):
    return bind_op(op, greedy_ctx(), selector=GreedyImplementationSelector())


def duplicate_frames():
    rows = {
        "id": range(7),
        "k": [2, 1, 2, None, 1, None, 3],
        "v": ["c", "a", "c", "n", "b", "n", "d"],
    }
    return pd.DataFrame(rows), pl.DataFrame(rows)


@pytest.mark.parametrize("variant,backends", [
    (MaskSelectionOp, {"pandas", "polars"}),
    (SliceSelectionOp, {"pandas", "polars"}),
    (SampleSelectionOp, {"pandas", "polars"}),
    (DropDuplicatesSelectionOp, {"pandas", "polars"}),
    (DropMissingSelectionOp, {"pandas", "polars"}),
    # polars rows carry no index labels to drop by.
    (DropLabelsSelectionOp, {"pandas"}),
])
def test_each_variant_registers_its_own_impls(variant, backends):
    assert set(get_default_physical_registry().backends_for(variant)) == backends


@pytest.mark.parametrize("field,value", [
    ("weights", "weight"),
    ("weights", [0.0, 1.0, 0.0, 1.0]),
    ("seed", np.random.RandomState(7)),
    ("seed", np.random.default_rng(7)),
    ("seed", np.random.PCG64(7)),
    ("seed", np.array([7], dtype=np.uint32)),
], ids=["weight-column", "weight-array", "random-state", "generator", "bit-generator",
        "array-seed"])
def test_sample_greedy_selector_falls_back_to_pandas(field, value):
    frame = pd.DataFrame({"x": [10, 20, 30, 40], "weight": [0.0, 1.0, 0.0, 1.0]})
    fields = {"n": 2, "seed": 7, field: value}
    # Keep the oracle's RNG independent of the implementation's RNG.
    oracle = deepcopy(fields)
    expected = frame.sample(n=2, weights=oracle.get("weights"), random_state=oracle["seed"])

    bound = greedy_bind(SampleSelectionOp(**fields))

    assert isinstance(bound, PandasSampleSelectionOp)
    assert_pandas_frame_equal(bound.process("fit_transform", [frame]), expected)


@pytest.mark.parametrize("method,kwargs", [
    ("head", {"n": 3}),
    ("tail", {"n": -2}),
    ("drop_duplicates", {"subset": "x", "keep": "last"}),
    ("dropna", {"subset": ["x"], "thresh": 1}),
    ("sample", {"n": 3, "random_state": 7}),
])
def test_selection_executes_in_greedy_plan(method, kwargs):
    frame = pd.DataFrame({"id": range(5), "x": [2.0, 1.0, 2.0, np.nan, 3.0]})
    with config(implementation_selector="greedy"):
        ops, *_ = optimize(getattr(st.as_data_op(frame), method)(**kwargs))

    result = run_plan(ops)

    assert isinstance(result, pl.DataFrame)
    expected = getattr(frame, method)(**kwargs)
    if method == "sample":
        assert result.height == len(expected)
        assert result["id"].n_unique() == result.height
        assert set(result["id"]) <= set(frame["id"])
        expected = frame.iloc[result["id"].to_list()]
    assert_polars_frame_equal(result, pl.from_pandas(expected))


@pytest.mark.parametrize("method", ["dropna", "drop_duplicates"])
def test_series_selection_executes_in_greedy_plan(method):
    frame = pd.DataFrame({"x": [1.0, 1.0, None]})
    data = st.as_data_op(frame)
    dag = getattr(data["x"], method)()

    with config(implementation_selector="greedy"):
        ops, *_ = optimize(dag)
        result = st._api.evaluate(dag)

    assert isinstance(result, pl.Series)
    expected_impl = (PolarsSeriesDropMissingSelectionOp if method == "dropna"
                     else PolarsSeriesDropDuplicatesSelectionOp)
    assert isinstance(ops[-1], expected_impl)
    expected = getattr(frame["x"], method)().reset_index(drop=True)
    pd.testing.assert_series_equal(result.to_pandas().reset_index(drop=True), expected,
                                   check_dtype=False)


@pytest.mark.parametrize("keep", ["first", "last", False])
def test_series_drop_duplicates_keep_in_greedy_plan(keep):
    frame = pd.DataFrame({"x": [1, 2, 1, None, None]})
    data = st.as_data_op(frame)
    dag = data["x"].drop_duplicates(keep=keep)

    with config(implementation_selector="greedy"):
        result = st._api.evaluate(dag)

    expected = frame["x"].drop_duplicates(keep=keep).reset_index(drop=True)
    pd.testing.assert_series_equal(result.to_pandas().reset_index(drop=True), expected,
                                   check_dtype=False)


def test_polars_series_dropna_removes_nan_and_null():
    series = pl.Series("x", [1.0, np.nan, None])
    op = PolarsSeriesDropMissingSelectionOp()
    assert op.process("fit_transform", [series]).to_list() == [1.0]


class TestPandasSampleSelectionOp(unittest.TestCase):
    def test_process_resolves_and_translates_all_fields(self):
        frame = pd.DataFrame({"x": range(5), "weight": [1.0, 2.0, 3.0, 4.0, 5.0]})
        op = PandasSampleSelectionOp(n=OperandRef(1), with_replacement=OperandRef(2),
                                     weights="weight", seed=OperandRef(3), ignore_index=True)

        result = op.process("fit_transform", [frame, 3, False, 7])
        expected = frame.sample(n=3, replace=False, weights="weight", random_state=7,
                                ignore_index=True)

        assert_pandas_frame_equal(result, expected)


class TestPolarsSampleSelectionOp(unittest.TestCase):
    def test_supported_fields(self):
        cases = [
            {"n": 2}, {"n": OperandRef(1)},
            {"frac": 0.5}, {"frac": OperandRef(1)}, {"with_replacement": OperandRef(1)},
            {"seed": 7}, {"seed": np.int64(7)},
            {"weights": None, "seed": None},
            {"ignore_index": True}, {"ignore_index": OperandRef(1)},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                self.assertTrue(PolarsSampleSelectionOp.supports(SampleSelectionOp(**fields), None))

    def test_unsupported_fields(self):
        cases = [
            {"weights": "weight"}, {"weights": OperandRef(1)},
            {"seed": np.random.default_rng(7)}, {"seed": OperandRef(1)},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                self.assertFalse(PolarsSampleSelectionOp.supports(SampleSelectionOp(**fields), None))

    def test_supported_call_is_selected_by_greedy_planner(self):
        op = SampleSelectionOp(n=2, with_replacement=True, seed=7)

        self.assertIsInstance(greedy_bind(op), PolarsSampleSelectionOp)

    def test_weighted_call_falls_back_and_executes_with_pandas(self):
        frame = pd.DataFrame({"x": [10, 20, 30, 40]})
        weights = [0.0, 1.0, 0.0, 1.0]
        op = SampleSelectionOp(n=2, weights=OperandRef(1), seed=7)

        bound = greedy_bind(op)
        result = bound.process("fit_transform", [frame, weights])
        expected = frame.sample(n=2, weights=weights, random_state=7)

        self.assertIsInstance(bound, PandasSampleSelectionOp)
        assert_pandas_frame_equal(result, expected)

    def test_translation_resolves_runtime_operands(self):
        frame = pl.DataFrame({"x": range(5)})
        op = PolarsSampleSelectionOp(frac=OperandRef(1), with_replacement=OperandRef(2),
                                     seed=OperandRef(3), ignore_index=OperandRef(4))

        translated = get_polars_sample_kwargs(op, [frame, 0.5, True, 7, True])

        self.assertEqual({"n": 2, "with_replacement": True, "seed": 7, "shuffle": True},
                         translated)

    def test_process_defaults_to_one_row(self):
        frame = pl.DataFrame({"x": range(5)})

        result = PolarsSampleSelectionOp().process("fit_transform", [frame])

        self.assertEqual(1, result.height)

    def test_process_uses_repeatable_integer_seed(self):
        frame = pl.DataFrame({"x": range(5)})
        op = PolarsSampleSelectionOp(frac=0.5, seed=7)

        first = op.process("fit_transform", [frame])
        second = op.process("fit_transform", [frame])

        self.assertEqual(2, first.height)
        assert_polars_frame_equal(first, second)

    def test_process_maps_replacement_for_upsampling(self):
        frame = pl.DataFrame({"x": range(3)})
        op = PolarsSampleSelectionOp(n=8, with_replacement=True, seed=7)

        result = op.process("fit_transform", [frame])

        self.assertEqual(8, result.height)

    def test_empty_frame_with_zero_fraction(self):
        frame = pl.DataFrame(schema={"x": pl.Int64})
        op = PolarsSampleSelectionOp(frac=0.0, seed=7)

        result = op.process("fit_transform", [frame])

        assert_polars_frame_equal(result, frame)

    def test_single_row_frame_with_default_n(self):
        frame = pl.DataFrame({"x": [42]})
        op = PolarsSampleSelectionOp(seed=7)

        result = op.process("fit_transform", [frame])

        assert_polars_frame_equal(result, frame)


@pytest.mark.parametrize("n,fraction", [(None, None), (None, 0.5), (3, None), (0, None), (None, 0.0)])
def test_sample_resolves_optional_operands_before_defaults(n, fraction):
    frame = pd.DataFrame({"x": range(5)})
    op = PolarsSampleSelectionOp(n=OperandRef(1), frac=OperandRef(2))

    result = op.process("fit_transform", [pl.from_pandas(frame), n, fraction])

    assert result.height == len(frame.sample(n=n, frac=fraction))
    assert result.columns == list(frame.columns)
    assert result["x"].n_unique() == result.height
    assert set(result["x"]) <= set(frame["x"])


@pytest.mark.parametrize("fields,extra", [
    ({"n": 1, "frac": 0.5}, []),
    ({"n": OperandRef(1), "frac": OperandRef(2)}, [1, 0.5]),
])
def test_sample_rejects_resolved_n_and_frac(fields, extra):
    op = PolarsSampleSelectionOp(**fields)
    with pytest.raises(ValueError, match="provide n or frac exclusively"):
        op.process("fit_transform", [pl.DataFrame({"x": range(5)}), *extra])


SAMPLE_ROUNDING_CASES = [(5, 0.0, 0), (5, 0.1, 0), (5, 0.3, 2), (5, 0.5, 2), (3, 0.5, 2), (5, 1.0, 5)]
SAMPLE_ROUNDING_IDS = ["zero", "below-half", "half-to-even-up", "half-to-even-down", "three-row-half", "whole-frame"]


@pytest.mark.parametrize("row_count,fraction,expected_n", SAMPLE_ROUNDING_CASES, ids=SAMPLE_ROUNDING_IDS)
def test_fraction_count_uses_pandas_rounding(row_count, fraction, expected_n):
    frame = pl.DataFrame({"x": range(row_count)})
    op = PolarsSampleSelectionOp(frac=fraction)

    translated = get_polars_sample_kwargs(op, [frame])
    pandas_count = len(pd.DataFrame({"x": range(row_count)}).sample(frac=fraction))

    assert translated["n"] == expected_n
    assert translated["n"] == pandas_count


def test_polars_sample_shuffles_like_pandas():
    # pandas.sample returns a permutation; polars only does with shuffle=True.
    frame = pl.DataFrame({"x": range(20)})
    op = PolarsSampleSelectionOp(frac=1.0, seed=0)
    result = op.process("fit", [frame])["x"].to_list()
    assert sorted(result) == list(range(20))
    assert result != list(range(20))


@pytest.mark.parametrize("impl", [PandasSliceSelectionOp, PolarsSliceSelectionOp])
@pytest.mark.parametrize("from_end,method", [(False, "head"), (True, "tail")])
@pytest.mark.parametrize("args", [(), (-2,), (0,), (2,)], ids=["default", "negative", "zero", "positive"])
def test_slice_matches_pandas(impl, from_end, method, args):
    frame = pd.DataFrame({"x": range(8), "y": range(10, 18)})
    op = impl(*args, from_end=from_end)
    source = frame if impl is PandasSliceSelectionOp else pl.from_pandas(frame)

    result = op.process("fit_transform", [source])
    expected = getattr(frame, method)(*args)

    assert_polars_frame_equal(pl.DataFrame(result), pl.from_pandas(expected))


def test_pandas_slice_resolves_runtime_operand():
    frame = pd.DataFrame({"x": range(5)})
    op = PandasSliceSelectionOp(n=OperandRef(1))

    result = op.process("fit_transform", [frame, -2])

    assert_pandas_frame_equal(result, frame.head(-2))


def test_polars_slice_resolves_runtime_operand():
    frame = pl.DataFrame({"x": range(5)})
    op = PolarsSliceSelectionOp(n=OperandRef(1), from_end=True)

    result = op.process("fit_transform", [frame, -2])

    assert_polars_frame_equal(result, frame.tail(-2))


class TestPandasDropDuplicatesSelectionOp(unittest.TestCase):
    def test_process_resolves_subset_and_translates_keep(self):
        frame = pd.DataFrame({"k": [1, 1, 2], "v": ["a", "b", "c"]})
        op = PandasDropDuplicatesSelectionOp(subset=OperandRef(1), keep="none")

        result = op.process("fit_transform", [frame, "k"])
        expected = frame.drop_duplicates(subset="k", keep=False)

        assert_pandas_frame_equal(result, expected)


class TestPolarsDropDuplicatesSelectionOp(unittest.TestCase):
    def test_supports_frames_only(self):
        frame_op = DropDuplicatesSelectionOp(subset=OperandRef(1), ignore_index=OperandRef(2))
        series_op = DropDuplicatesSelectionOp()
        series_op.output_type = OutputType.SERIES

        self.assertTrue(PolarsDropDuplicatesSelectionOp.supports(frame_op, None))
        self.assertFalse(PolarsDropDuplicatesSelectionOp.supports(series_op, None))

    def test_supported_call_is_selected_by_greedy_planner(self):
        op = DropDuplicatesSelectionOp(subset=OperandRef(1), keep="none", ignore_index=True)

        self.assertIsInstance(greedy_bind(op), PolarsDropDuplicatesSelectionOp)

    def test_default_keeps_first_and_maintains_order(self):
        frame = pl.DataFrame({"k": [2, 1, 2]})

        result = PolarsDropDuplicatesSelectionOp().process("fit_transform", [frame])

        self.assertEqual([2, 1], result["k"].to_list())

    def test_keep_none_drops_every_copy(self):
        frame = pl.DataFrame({"k": [2, 1, 2]})

        result = PolarsDropDuplicatesSelectionOp(keep="none").process("fit_transform", [frame])

        self.assertEqual([1], result["k"].to_list())

    def test_process_resolves_subset_operand(self):
        frame = pl.DataFrame({"k": [2, 1, 2], "v": ["a", "b", "c"]})
        op = PolarsDropDuplicatesSelectionOp(subset=OperandRef(1))

        result = op.process("fit_transform", [frame, "k"])

        self.assertEqual(["a", "b"], result["v"].to_list())


DROP_DUPLICATE_CASES = [
    ("k", "first", [0, 1, 3, 6]),
    ("k", "last", [2, 4, 5, 6]),
    ("k", False, [6]),
    (["k", "v"], "first", [0, 1, 3, 4, 6]),
    (["k", "v"], "last", [1, 2, 4, 5, 6]),
    (["k", "v"], False, [1, 4, 6]),
]
DROP_DUPLICATE_IDS = ["subset-first", "subset-last", "subset-none", "multi-first", "multi-last", "multi-none"]
_KEEP_FIELD = {"first": "first", "last": "last", False: "none"}


@pytest.mark.parametrize("subset,keep,expected_ids", DROP_DUPLICATE_CASES, ids=DROP_DUPLICATE_IDS)
def test_drop_duplicate_rows_match_pandas(subset, keep, expected_ids):
    pandas_frame, polars_frame = duplicate_frames()
    fields = {"subset": subset, "keep": _KEEP_FIELD[keep]}

    pandas_result = PandasDropDuplicatesSelectionOp(**fields).process("fit_transform", [pandas_frame])
    polars_result = PolarsDropDuplicatesSelectionOp(**fields).process("fit_transform", [polars_frame])

    assert pandas_frame.drop_duplicates(subset=subset, keep=keep)["id"].tolist() == expected_ids
    assert pandas_result["id"].tolist() == expected_ids
    assert_polars_frame_equal(polars_result, polars_frame[expected_ids])


DEFAULT_DUPLICATE_CASES = [("first", ["c", "a", "n"]), ("last", ["a", "c", "n"]), (False, ["a"])]
DEFAULT_DUPLICATE_IDS = ["first", "last", "none"]


@pytest.mark.parametrize("keep,expected_values", DEFAULT_DUPLICATE_CASES, ids=DEFAULT_DUPLICATE_IDS)
def test_default_subset_matches_pandas(keep, expected_values):
    rows = {"k": [2.0, 1.0, 2.0, np.nan, np.nan], "v": ["c", "a", "c", "n", "n"]}
    pandas_frame = pd.DataFrame(rows)
    polars_frame = pl.DataFrame(rows)
    fields = {"keep": _KEEP_FIELD[keep]}

    pandas_result = PandasDropDuplicatesSelectionOp(**fields).process("fit_transform", [pandas_frame])
    polars_result = PolarsDropDuplicatesSelectionOp(**fields).process("fit_transform", [polars_frame])

    assert pandas_result["v"].tolist() == expected_values
    assert_polars_frame_equal(polars_result, polars_frame[pandas_result.index.to_list()])


def test_drop_duplicates_empty_input_matches_pandas():
    pandas_frame = pd.DataFrame({"k": pd.Series(dtype="int64"), "v": pd.Series(dtype="str")})
    polars_frame = pl.DataFrame(schema={"k": pl.Int64, "v": pl.String})

    pandas_result = PandasDropDuplicatesSelectionOp().process("fit_transform", [pandas_frame])
    polars_result = PolarsDropDuplicatesSelectionOp().process("fit_transform", [polars_frame])

    assert pandas_result.to_dict("records") == polars_result.to_dicts()


def test_drop_duplicates_single_row_matches_pandas():
    pandas_frame = pd.DataFrame({"k": [1], "v": ["a"]})
    polars_frame = pl.DataFrame({"k": [1], "v": ["a"]})

    pandas_result = PandasDropDuplicatesSelectionOp().process("fit_transform", [pandas_frame])
    polars_result = PolarsDropDuplicatesSelectionOp().process("fit_transform", [polars_frame])

    assert pandas_result.to_dict("records") == polars_result.to_dicts()


# (pandas dropna kwargs, equivalent fields, surviving "s" values)
DROPNA_CASES = [
    ({}, {}, ["a"]),
    ({"how": "all"}, {"min_present": 1}, ["a", "b", "c"]),
    ({"thresh": 1}, {"min_present": 1}, ["a", "b", "c"]),
    ({"thresh": 2}, {"min_present": 2}, ["a"]),
    ({"subset": ["s"]}, {"subset": ["s"]}, ["a", "b", "c"]),
    ({"subset": ["f"], "how": "all"}, {"subset": ["f"], "min_present": 1}, ["a"]),
]
DROPNA_IDS = ["default-any", "all", "thresh-one", "thresh-two", "non-float-subset", "float-subset"]


@pytest.mark.parametrize("kwargs,fields,expected_s", DROPNA_CASES, ids=DROPNA_IDS)
def test_dropna_rows_match_pandas(kwargs, fields, expected_s):
    rows = {"f": [1.0, np.nan, None, np.nan], "s": ["a", "b", "c", None]}
    pandas_frame = pd.DataFrame(rows)
    polars_frame = pl.DataFrame(rows)

    pandas_result = PandasDropMissingSelectionOp(**fields).process("fit_transform", [pandas_frame])
    polars_result = PolarsDropMissingSelectionOp(**fields).process("fit_transform", [polars_frame])

    assert pandas_frame.dropna(**kwargs)["s"].tolist() == expected_s
    assert pandas_result["s"].tolist() == expected_s
    assert_polars_frame_equal(polars_result, polars_frame[pandas_result.index.to_list()])


class TestPolarsDropMissingSelectionOp(unittest.TestCase):
    def test_supported_call_is_selected_by_greedy_planner(self):
        op = DropMissingSelectionOp(subset=["f"], min_present=1)

        self.assertIsInstance(greedy_bind(op), PolarsDropMissingSelectionOp)

    def test_process_resolves_subset_and_min_present_operands(self):
        frame = pl.DataFrame({"f": [1.0, np.nan, None], "s": ["a", "b", "c"]})
        op = PolarsDropMissingSelectionOp(subset=OperandRef(1), min_present=OperandRef(2))

        result = op.process("fit_transform", [frame, ["f", "s"], 2])

        self.assertEqual(["a"], result["s"].to_list())


class TestDropLabelsSelectionOp(unittest.TestCase):
    def test_binds_pandas_even_under_the_greedy_selector(self):
        op = DropLabelsSelectionOp(labels=[0])

        self.assertIsInstance(greedy_bind(op), PandasDropLabelsSelectionOp)

    def test_process_drops_rows_by_label(self):
        frame = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])
        op = PandasDropLabelsSelectionOp(labels=OperandRef(1), errors="ignore")

        result = op.process("fit_transform", [frame, ["b", "z"]])

        assert_pandas_frame_equal(result, frame.drop(index=["b", "z"], errors="ignore"))


if __name__ == "__main__":
    unittest.main()
