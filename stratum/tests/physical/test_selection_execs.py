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
from stratum.optimizer.ir._ops import OperandRef
from stratum.optimizer.ir._selection_ops import SelectionKind, SelectionOp
from stratum.optimizer.physical._impl_selection import GreedyImplementationSelector, bind_op
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._selection_execs import (
    PandasMethodBasedSelectionOp,
    PolarsDropDuplicatesSelectionOp,
    PolarsDropnaSelectionOp,
    PolarsMaskSelectionOp,
    PolarsSampleSelectionOp,
    PolarsSliceSelectionOp,
    get_param,
    get_polars_sample_kwargs,
    get_polars_unique_kwargs,
)
from stratum.tests.physical.test_source_execs import run_plan


def greedy_ctx():
    kwargs = {
        "backend": "polars",
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


@pytest.mark.parametrize("kind", list(SelectionKind))
@pytest.mark.parametrize("impl,supported", [
    (PolarsMaskSelectionOp, (SelectionKind.MASK,)),
    (PolarsSliceSelectionOp, (SelectionKind.HEAD, SelectionKind.TAIL)),
    (PandasMethodBasedSelectionOp, (SelectionKind.DROPNA, SelectionKind.DROP_DUPLICATES,
                              SelectionKind.HEAD, SelectionKind.TAIL, SelectionKind.SAMPLE)),
])
def test_supported_selection_kinds(impl, supported, kind):
    assert impl.supports(SelectionOp(kind=kind), None) == (kind in supported)


@pytest.mark.parametrize("parameter,value", [
    ("weights", "weight"),
    ("weights", [0.0, 1.0, 0.0, 1.0]),
    ("axis", 1),
    ("axis", "columns"),
    ("random_state", np.random.RandomState(7)),
    ("random_state", np.random.default_rng(7)),
    ("random_state", np.random.PCG64(7)),
    ("random_state", np.array([7], dtype=np.uint32)),
], ids=["weight-column", "weight-array", "axis-one", "axis-columns",
        "random-state", "generator", "bit-generator", "array-seed"])
def test_sample_greedy_selector_falls_back_to_pandas(parameter, value):
    frame = pd.DataFrame({"x": [10, 20, 30, 40], "weight": [0.0, 1.0, 0.0, 1.0]})
    kwargs = {"n": 2, "random_state": 7, parameter: value}
    # Keep the oracle's RNG independent of the implementation's RNG.
    expected = frame.sample(**deepcopy(kwargs))

    bound = greedy_bind(SelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs))

    assert isinstance(bound, PandasMethodBasedSelectionOp)
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


class TestPandasMethodBasedSelectionOp(unittest.TestCase):
    def test_process_resolves_and_delegates_all_parameters(self):
        frame = pd.DataFrame({"x": range(5), "weight": [1.0, 2.0, 3.0, 4.0, 5.0]})
        kwargs = {
            "n": OperandRef(1),
            "replace": OperandRef(2),
            "weights": "weight",
            "random_state": OperandRef(3),
            "axis": "index",
            "ignore_index": True,
        }
        op = PandasMethodBasedSelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs)

        result = op.process("fit_transform", [frame, 3, False, 7])
        expected = frame.sample(n=3, replace=False, weights="weight", random_state=7, axis="index", ignore_index=True)

        assert_pandas_frame_equal(result, expected)


class TestPolarsSampleSelectionOp(unittest.TestCase):
    def test_supported_parameters(self):
        cases = [
            {"n": 2}, {"n": OperandRef(1)},
            {"frac": 0.5}, {"frac": OperandRef(1)}, {"replace": OperandRef(1)},
            {"random_state": 7}, {"random_state": np.int64(7)},
            {"axis": "index"}, {"axis": "rows"}, {"axis": 0},
            {"weights": None, "random_state": None, "axis": None},
            {"ignore_index": True}, {"ignore_index": OperandRef(1)},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                op = SelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs)
                self.assertTrue(PolarsSampleSelectionOp.supports(op, None))

    def test_unsupported_parameters(self):
        cases = [
            {"weights": "weight"}, {"weights": OperandRef(1)},
            {"random_state": np.random.default_rng(7)}, {"random_state": OperandRef(1)},
            {"axis": "columns"}, {"axis": 1}, {"axis": OperandRef(1)},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                op = SelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs)
                self.assertFalse(PolarsSampleSelectionOp.supports(op, None))
        self.assertFalse(PolarsSampleSelectionOp.supports(SelectionOp(kind=SelectionKind.HEAD), None))

    def test_supported_call_is_selected_by_greedy_planner(self):
        kwargs = {"n": 2, "replace": True, "random_state": 7, "axis": "rows"}
        op = SelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs)

        self.assertIsInstance(greedy_bind(op), PolarsSampleSelectionOp)

    def test_weighted_call_falls_back_and_executes_with_pandas(self):
        frame = pd.DataFrame({"x": [10, 20, 30, 40]})
        weights = [0.0, 1.0, 0.0, 1.0]
        kwargs = {"n": 2, "weights": OperandRef(1), "random_state": 7}
        op = SelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs)

        bound = greedy_bind(op)
        result = bound.process("fit_transform", [frame, weights])
        expected = frame.sample(n=2, weights=weights, random_state=7)

        self.assertIsInstance(bound, PandasMethodBasedSelectionOp)
        assert_pandas_frame_equal(result, expected)

    def test_get_param_keeps_operand_ref_raw_at_plan_time(self):
        ref = OperandRef(1)
        op = SelectionOp(kind=SelectionKind.SAMPLE, kwargs={"frac": ref})

        self.assertIs(get_param(op, 1, "frac"), ref)

    def test_keyword_translation_resolves_runtime_operands(self):
        frame = pl.DataFrame({"x": range(5)})
        kwargs = {
            "frac": OperandRef(1),
            "replace": OperandRef(2),
            "random_state": OperandRef(3),
            "ignore_index": OperandRef(4),
        }
        op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs)

        translated = get_polars_sample_kwargs(op, [frame, 0.5, True, 7, True])

        self.assertEqual({"n": 2, "with_replacement": True, "seed": 7}, translated)

    def test_positional_frac_replace_and_seed_use_pandas_order(self):
        frame = pl.DataFrame({"x": range(5)})
        args = (None, OperandRef(1), OperandRef(2), None, OperandRef(3), 0, True)
        op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE, args=args)

        translated = get_polars_sample_kwargs(op, [frame, 0.5, True, 7])

        self.assertEqual({"n": 2, "with_replacement": True, "seed": 7}, translated)

    def test_process_defaults_to_one_row(self):
        frame = pl.DataFrame({"x": range(5)})
        op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE)

        result = op.process("fit_transform", [frame])

        self.assertEqual(1, result.height)

    def test_process_uses_repeatable_integer_seed(self):
        frame = pl.DataFrame({"x": range(5)})
        kwargs = {"frac": 0.5, "random_state": 7}
        op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs)

        first = op.process("fit_transform", [frame])
        second = op.process("fit_transform", [frame])

        self.assertEqual(2, first.height)
        assert_polars_frame_equal(first, second)

    def test_process_maps_replace_for_upsampling(self):
        frame = pl.DataFrame({"x": range(3)})
        kwargs = {"n": 8, "replace": True, "random_state": 7}
        op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs)

        result = op.process("fit_transform", [frame])

        self.assertEqual(8, result.height)

    def test_empty_frame_with_zero_fraction(self):
        frame = pl.DataFrame(schema={"x": pl.Int64})
        kwargs = {"frac": 0.0, "random_state": 7}
        op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE, kwargs=kwargs)

        result = op.process("fit_transform", [frame])

        assert_polars_frame_equal(result, frame)

    def test_single_row_frame_with_default_n(self):
        frame = pl.DataFrame({"x": [42]})
        op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE, kwargs={"random_state": 7})

        result = op.process("fit_transform", [frame])

        assert_polars_frame_equal(result, frame)


@pytest.mark.parametrize("n,fraction", [(None, None), (None, 0.5), (3, None), (0, None), (None, 0.0)])
@pytest.mark.parametrize("positional", [False, True])
def test_sample_resolves_optional_operands_before_defaults(n, fraction, positional):
    frame = pd.DataFrame({"x": range(5)})
    args = (OperandRef(1), OperandRef(2)) if positional else ()
    kwargs = {} if positional else {"n": OperandRef(1), "frac": OperandRef(2)}
    op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE, args=args, kwargs=kwargs)

    result = op.process("fit_transform", [pl.from_pandas(frame), n, fraction])

    assert result.height == len(frame.sample(n=n, frac=fraction))
    assert result.columns == list(frame.columns)
    assert result["x"].n_unique() == result.height
    assert set(result["x"]) <= set(frame["x"])


@pytest.mark.parametrize("args,kwargs,extra", [
    ((), {"n": 1, "frac": 0.5}, []),
    ((1,), {"frac": 0.5}, []),
    ((), {"n": OperandRef(1), "frac": OperandRef(2)}, [1, 0.5]),
])
def test_sample_rejects_resolved_n_and_frac(args, kwargs, extra):
    op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE, args=args, kwargs=kwargs)
    with pytest.raises(ValueError, match="provide n or frac exclusively"):
        op.process("fit_transform", [pl.DataFrame({"x": range(5)}), *extra])


SAMPLE_ROUNDING_CASES = [(5, 0.0, 0), (5, 0.1, 0), (5, 0.3, 2), (5, 0.5, 2), (3, 0.5, 2), (5, 1.0, 5)]
SAMPLE_ROUNDING_IDS = ["zero", "below-half", "half-to-even-up", "half-to-even-down", "three-row-half", "whole-frame"]


@pytest.mark.parametrize("row_count,fraction,expected_n", SAMPLE_ROUNDING_CASES, ids=SAMPLE_ROUNDING_IDS)
def test_fraction_count_uses_pandas_rounding(row_count, fraction, expected_n):
    frame = pl.DataFrame({"x": range(row_count)})
    op = PolarsSampleSelectionOp(kind=SelectionKind.SAMPLE, kwargs={"frac": fraction})

    translated = get_polars_sample_kwargs(op, [frame])
    pandas_count = len(pd.DataFrame({"x": range(row_count)}).sample(frac=fraction))

    assert translated["n"] == expected_n
    assert translated["n"] == pandas_count


@pytest.mark.parametrize("impl", [PandasMethodBasedSelectionOp, PolarsSliceSelectionOp])
@pytest.mark.parametrize("kind,method", [(SelectionKind.HEAD, "head"), (SelectionKind.TAIL, "tail")])
@pytest.mark.parametrize("args", [(), (-2,), (0,), (2,)], ids=["default", "negative", "zero", "positive"])
def test_slice_matches_pandas(impl, kind, method, args):
    frame = pd.DataFrame({"x": range(8), "y": range(10, 18)})
    op = impl(kind=kind, args=args)
    source = frame if impl is PandasMethodBasedSelectionOp else pl.from_pandas(frame)

    result = op.process("fit_transform", [source])
    expected = getattr(frame, method)(*args)

    assert_polars_frame_equal(pl.DataFrame(result), pl.from_pandas(expected))


def test_pandas_slice_resolves_runtime_operand():
    frame = pd.DataFrame({"x": range(5)})
    op = PandasMethodBasedSelectionOp(kind=SelectionKind.HEAD, args=(OperandRef(1),))

    result = op.process("fit_transform", [frame, -2])

    assert_pandas_frame_equal(result, frame.head(-2))


def test_polars_slice_resolves_runtime_operand():
    frame = pl.DataFrame({"x": range(5)})
    op = PolarsSliceSelectionOp(kind=SelectionKind.TAIL, args=(OperandRef(1),))

    result = op.process("fit_transform", [frame, -2])

    assert_polars_frame_equal(result, frame.tail(-2))


class TestPandasMethodDropDuplicates(unittest.TestCase):
    def test_process_resolves_subset_and_keep(self):
        frame = pd.DataFrame({"k": [1, 1, 2], "v": ["a", "b", "c"]})
        kwargs = {"subset": OperandRef(1), "keep": OperandRef(2)}
        op = PandasMethodBasedSelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs=kwargs)

        result = op.process("fit_transform", [frame, "k", False])
        expected = frame.drop_duplicates(subset="k", keep=False)

        assert_pandas_frame_equal(result, expected)


class TestPolarsDropDuplicatesSelectionOp(unittest.TestCase):
    def test_supports_subset_keep_and_ignored_index(self):
        subset = SelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs={"subset": OperandRef(1)})
        keep = SelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs={"keep": OperandRef(1)})
        inplace_false = SelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs={"inplace": False})
        ignore_index = SelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs={"ignore_index": OperandRef(1)})

        self.assertTrue(PolarsDropDuplicatesSelectionOp.supports(subset, None))
        self.assertTrue(PolarsDropDuplicatesSelectionOp.supports(keep, None))
        self.assertTrue(PolarsDropDuplicatesSelectionOp.supports(inplace_false, None))
        self.assertTrue(PolarsDropDuplicatesSelectionOp.supports(ignore_index, None))

    def test_rejects_unresolved_inplace(self):
        kwargs = {"inplace": OperandRef(1)}
        op = SelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs=kwargs)

        self.assertFalse(PolarsDropDuplicatesSelectionOp.supports(op, None))

    def test_supported_call_is_selected_by_greedy_planner(self):
        kwargs = {"subset": OperandRef(1), "keep": OperandRef(2), "ignore_index": True}
        op = SelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs=kwargs)

        self.assertIsInstance(greedy_bind(op), PolarsDropDuplicatesSelectionOp)

    def test_inplace_true_falls_back_and_executes_with_pandas(self):
        frame = pd.DataFrame({"k": [2, 1, 2, 3], "v": ["a", "b", "c", "d"]})
        expected = frame.copy()
        expected.drop_duplicates(subset="k", keep="last", inplace=True, ignore_index=True)
        kwargs = {"subset": "k", "keep": "last", "inplace": True, "ignore_index": True}
        op = SelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs=kwargs)

        bound = greedy_bind(op)
        result = bound.process("fit_transform", [frame])

        self.assertIsInstance(bound, PandasMethodBasedSelectionOp)
        self.assertIsNone(result)
        assert_pandas_frame_equal(frame, expected)

    def test_default_translation_keeps_first_and_maintains_order(self):
        frame = pl.DataFrame({"k": [2, 1, 2]})
        op = PolarsDropDuplicatesSelectionOp(kind=SelectionKind.DROP_DUPLICATES)

        translated = get_polars_unique_kwargs(op, [frame])

        self.assertEqual({"keep": "first", "maintain_order": True}, translated)

    def test_keep_false_maps_to_none(self):
        frame = pl.DataFrame({"k": [2, 1, 2]})
        op = PolarsDropDuplicatesSelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs={"keep": False})

        translated = get_polars_unique_kwargs(op, [frame])

        self.assertEqual("none", translated["keep"])
        self.assertTrue(translated["maintain_order"])

    def test_runtime_translation_resolves_subset_and_keep(self):
        frame = pl.DataFrame({"k": [2, 1, 2]})
        kwargs = {"subset": OperandRef(1), "keep": OperandRef(2), "ignore_index": OperandRef(3)}
        op = PolarsDropDuplicatesSelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs=kwargs)

        translated = get_polars_unique_kwargs(op, [frame, "k", False, True])

        expected = {"subset": "k", "keep": "none", "maintain_order": True}
        self.assertEqual(expected, translated)

    def test_positional_subset_translation(self):
        frame = pl.DataFrame({"k": [2, 1, 2], "v": ["a", "b", "c"]})
        op = PolarsDropDuplicatesSelectionOp(kind=SelectionKind.DROP_DUPLICATES, args=("k",))

        translated = get_polars_unique_kwargs(op, [frame])

        expected = {"subset": "k", "keep": "first", "maintain_order": True}
        self.assertEqual(expected, translated)


DROP_DUPLICATE_CASES = [
    ("k", "first", [0, 1, 3, 6]),
    ("k", "last", [2, 4, 5, 6]),
    ("k", False, [6]),
    (["k", "v"], "first", [0, 1, 3, 4, 6]),
    (["k", "v"], "last", [1, 2, 4, 5, 6]),
    (["k", "v"], False, [1, 4, 6]),
]
DROP_DUPLICATE_IDS = ["subset-first", "subset-last", "subset-none", "multi-first", "multi-last", "multi-none"]


@pytest.mark.parametrize("subset,keep,expected_ids", DROP_DUPLICATE_CASES, ids=DROP_DUPLICATE_IDS)
def test_drop_duplicate_rows_match_pandas(subset, keep, expected_ids):
    pandas_frame, polars_frame = duplicate_frames()
    kwargs = {"subset": subset, "keep": keep}
    pandas_op = PandasMethodBasedSelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs=kwargs)
    polars_op = PolarsDropDuplicatesSelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs=kwargs)

    pandas_result = pandas_op.process("fit_transform", [pandas_frame])
    polars_result = polars_op.process("fit_transform", [polars_frame])

    assert pandas_result["id"].tolist() == expected_ids
    assert_polars_frame_equal(polars_result, polars_frame[expected_ids])


DEFAULT_DUPLICATE_CASES = [("first", ["c", "a", "n"]), ("last", ["a", "c", "n"]), (False, ["a"])]
DEFAULT_DUPLICATE_IDS = ["first", "last", "none"]


@pytest.mark.parametrize("keep,expected_values", DEFAULT_DUPLICATE_CASES, ids=DEFAULT_DUPLICATE_IDS)
def test_default_subset_matches_pandas(keep, expected_values):
    rows = {"k": [2.0, 1.0, 2.0, np.nan, np.nan], "v": ["c", "a", "c", "n", "n"]}
    pandas_frame = pd.DataFrame(rows)
    polars_frame = pl.DataFrame(rows)
    kwargs = {"keep": keep}
    pandas_op = PandasMethodBasedSelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs=kwargs)
    polars_op = PolarsDropDuplicatesSelectionOp(kind=SelectionKind.DROP_DUPLICATES, kwargs=kwargs)

    pandas_result = pandas_op.process("fit_transform", [pandas_frame])
    polars_result = polars_op.process("fit_transform", [polars_frame])

    assert pandas_result["v"].tolist() == expected_values
    assert_polars_frame_equal(polars_result, polars_frame[pandas_result.index.to_list()])


def test_drop_duplicates_empty_input_matches_pandas():
    pandas_frame = pd.DataFrame({"k": pd.Series(dtype="int64"), "v": pd.Series(dtype="str")})
    polars_frame = pl.DataFrame(schema={"k": pl.Int64, "v": pl.String})
    pandas_op = PandasMethodBasedSelectionOp(kind=SelectionKind.DROP_DUPLICATES)
    polars_op = PolarsDropDuplicatesSelectionOp(kind=SelectionKind.DROP_DUPLICATES)

    pandas_result = pandas_op.process("fit_transform", [pandas_frame])
    polars_result = polars_op.process("fit_transform", [polars_frame])

    assert pandas_result.to_dict("records") == polars_result.to_dicts()


def test_drop_duplicates_single_row_matches_pandas():
    pandas_frame = pd.DataFrame({"k": [1], "v": ["a"]})
    polars_frame = pl.DataFrame({"k": [1], "v": ["a"]})
    pandas_op = PandasMethodBasedSelectionOp(kind=SelectionKind.DROP_DUPLICATES)
    polars_op = PolarsDropDuplicatesSelectionOp(kind=SelectionKind.DROP_DUPLICATES)

    pandas_result = pandas_op.process("fit_transform", [pandas_frame])
    polars_result = polars_op.process("fit_transform", [polars_frame])

    assert pandas_result.to_dict("records") == polars_result.to_dicts()


DROPNA_CASES = [
    ({}, ["a"]),
    ({"how": "all"}, ["a", "b", "c"]),
    ({"thresh": 1}, ["a", "b", "c"]),
    ({"thresh": 2}, ["a"]),
    ({"subset": ["s"]}, ["a", "b", "c"]),
    ({"subset": ["f"], "how": "all"}, ["a"]),
]
DROPNA_IDS = ["default-any", "all", "thresh-one", "thresh-two", "non-float-subset", "float-subset"]


@pytest.mark.parametrize("kwargs,expected_s", DROPNA_CASES, ids=DROPNA_IDS)
def test_dropna_rows_match_pandas(kwargs, expected_s):
    rows = {"f": [1.0, np.nan, None, np.nan], "s": ["a", "b", "c", None]}
    pandas_frame = pd.DataFrame(rows)
    polars_frame = pl.DataFrame(rows)
    pandas_op = PandasMethodBasedSelectionOp(kind=SelectionKind.DROPNA, kwargs=kwargs)
    polars_op = PolarsDropnaSelectionOp(kind=SelectionKind.DROPNA, kwargs=kwargs)

    pandas_result = pandas_op.process("fit_transform", [pandas_frame])
    polars_result = polars_op.process("fit_transform", [polars_frame])

    assert pandas_result["s"].tolist() == expected_s
    assert_polars_frame_equal(polars_result, polars_frame[pandas_result.index.to_list()])


class TestPolarsDropnaSelectionOp(unittest.TestCase):
    def test_supports_row_axes_and_non_inplace_calls(self):
        ops = [
            SelectionOp(kind=SelectionKind.DROPNA),
            SelectionOp(kind=SelectionKind.DROPNA, kwargs={"axis": 0}),
            SelectionOp(kind=SelectionKind.DROPNA, kwargs={"axis": "index"}),
            SelectionOp(kind=SelectionKind.DROPNA, kwargs={"axis": "rows", "inplace": False}),
        ]

        for op in ops:
            with self.subTest(kwargs=op.kwargs):
                self.assertTrue(PolarsDropnaSelectionOp.supports(op, None))

    def test_rejects_other_kinds_and_unsupported_calls(self):
        ops = [
            SelectionOp(kind=SelectionKind.HEAD),
            SelectionOp(kind=SelectionKind.DROPNA, args=(0,)),
            SelectionOp(kind=SelectionKind.DROPNA, kwargs={"axis": 1}),
            SelectionOp(kind=SelectionKind.DROPNA, kwargs={"axis": "columns"}),
            SelectionOp(kind=SelectionKind.DROPNA, kwargs={"axis": OperandRef(1)}),
            SelectionOp(kind=SelectionKind.DROPNA, kwargs={"inplace": True}),
            SelectionOp(kind=SelectionKind.DROPNA, kwargs={"inplace": OperandRef(1)}),
        ]

        for op in ops:
            with self.subTest(args=op.args, kwargs=op.kwargs):
                self.assertFalse(PolarsDropnaSelectionOp.supports(op, None))

    def test_supported_call_is_selected_by_greedy_planner(self):
        op = SelectionOp(kind=SelectionKind.DROPNA, kwargs={"subset": ["f"], "thresh": 1})

        self.assertIsInstance(greedy_bind(op), PolarsDropnaSelectionOp)

    def test_inplace_call_falls_back_and_executes_with_pandas(self):
        frame = pd.DataFrame({"f": [1.0, np.nan], "s": ["a", "b"]})
        expected = frame.dropna()
        op = SelectionOp(kind=SelectionKind.DROPNA, kwargs={"inplace": True})

        bound = greedy_bind(op)
        result = bound.process("fit_transform", [frame])

        self.assertIsInstance(bound, PandasMethodBasedSelectionOp)
        self.assertIsNone(result)
        assert_pandas_frame_equal(frame, expected)

    def test_process_resolves_subset_and_thresh_operands(self):
        frame = pl.DataFrame({"f": [1.0, np.nan, None], "s": ["a", "b", "c"]})
        kwargs = {"subset": OperandRef(1), "thresh": OperandRef(2)}
        op = PolarsDropnaSelectionOp(kind=SelectionKind.DROPNA, kwargs=kwargs)

        result = op.process("fit_transform", [frame, ["f", "s"], 2])

        self.assertEqual(["a"], result["s"].to_list())


if __name__ == "__main__":
    unittest.main()
