import unittest

import pandas as pd
import polars as pl
import pytest
import skrub

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize as optimize_plan
from stratum.optimizer.logical._metadata_ops import (
    ColumnNamesOp, MetadataOp, RenameOp, ResetIndexOp, ShapeOp, make_rename_op,
    make_reset_index_op, make_structure_read_op)
from stratum.optimizer.logical._ops import (
    GetAttrOp, MethodCallOp, OperandRef, OutputType)
from stratum.optimizer.physical import FlagBasedSelector
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._impl_selection import get_implementation_selector
from stratum.runtime._scheduler import SequentialScheduler
from .test_dataframe_ops import _inp, force_polars, optimize, run_op


def _call(method, *args, output_type=OutputType.FRAME, **kwargs) -> MethodCallOp:
    op = MethodCallOp(method, args=args, kwargs=kwargs)
    op.inputs = [_inp(None)]
    op.inputs[0].output_type = output_type
    op.inputs[0].outputs = [op]
    return op


class TestRenameExtraction(unittest.TestCase):
    """`rename` parses into which labels it remaps, per axis."""

    def test_columns_and_index_keywords(self):
        op = make_rename_op(_call("rename", columns={"a": "x"}, index={0: 9}))
        self.assertEqual(({"a": "x"}, {0: 9}, "ignore"), (op.columns, op.index, op.errors))

    def test_a_positional_mapper_renames_the_index_by_default(self):
        # pandas' default axis is the index, not the columns.
        op = make_rename_op(_call("rename", {"a": "x"}))
        self.assertEqual((None, {"a": "x"}), (op.columns, op.index))

    def test_a_mapper_with_a_column_axis_renames_columns(self):
        for axis in (1, "columns"):
            with self.subTest(axis=axis):
                op = make_rename_op(_call("rename", {"a": "x"}, axis=axis))
                self.assertEqual(({"a": "x"}, None), (op.columns, op.index))

    def test_errors_carries_over(self):
        op = make_rename_op(_call("rename", columns={"a": "x"}, errors="raise"))
        self.assertEqual("raise", op.errors)

    def test_calls_it_cannot_model_stay_opaque(self):
        cases = {
            "inplace": _call("rename", columns={"a": "x"}, inplace=True),
            "level": _call("rename", columns={"a": "x"}, level=0),
            "graph-fed axis": _call("rename", {"a": "x"}, axis=OperandRef(1)),
            "mapper and columns": _call("rename", {"a": "x"}, columns={"b": "y"}),
            "axis and columns": _call("rename", columns={"a": "x"}, axis=1),
            "nothing to rename": _call("rename"),
            "series": _call("rename", "name", output_type=OutputType.SERIES),
            # a grouped size() is typed as a frame but is a Series at run time.
            "scalar name": _call("rename", "name"),
            "graph-fed mapping": _call("rename", columns=OperandRef(1)),
        }
        for name, call in cases.items():
            with self.subTest(name):
                self.assertIsNone(make_rename_op(call))

    def test_a_rename_is_a_metadata_op(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ops = optimize(st.as_data_op(df).rename(columns={"a": "x"}),
                       OptConfig(dataframe_ops=True))
        renames = [o for o in ops if isinstance(o, RenameOp)]
        self.assertEqual(1, len(renames))
        self.assertIsInstance(renames[0], MetadataOp)
        self.assertEqual("Metadata", MetadataOp.logical_family)


class TestRenameExecution(unittest.TestCase):
    def test_pandas_renames_both_axes(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = run_op(RenameOp(columns={"a": "x"}, index={0: 9}), df)
        pd.testing.assert_frame_equal(df.rename(columns={"a": "x"}, index={0: 9}), result)

    def test_polars_renames_columns(self):
        with force_polars():
            result = run_op(RenameOp(columns={"a": "x"}), pl.DataFrame({"a": [1], "b": [2]}))
        self.assertEqual(["x", "b"], result.columns)

    def test_polars_ignores_absent_labels_like_pandas(self):
        # pandas' default errors="ignore" skips a label that is not there.
        with force_polars():
            result = run_op(RenameOp(columns={"a": "x", "zz": "y"}),
                            pl.DataFrame({"a": [1]}))
        self.assertEqual(["x"], result.columns)

    def test_polars_raises_on_absent_labels_when_asked(self):
        with force_polars(), self.assertRaises(pl.exceptions.ColumnNotFoundError):
            run_op(RenameOp(columns={"zz": "y"}, errors="raise"), pl.DataFrame({"a": [1]}))

    def test_polars_refuses_an_index_rename(self):
        from stratum.optimizer.physical._metadata_execs import PolarsRenameOp
        self.assertFalse(PolarsRenameOp.supports(RenameOp(index={0: 9}), None))
        self.assertTrue(PolarsRenameOp.supports(RenameOp(columns={"a": "x"}), None))


class TestResetIndex(unittest.TestCase):
    """Only the dropping form of `reset_index` is representable."""

    def test_drop_true_becomes_a_reset_index_op(self):
        self.assertIsInstance(make_reset_index_op(_call("reset_index", drop=True)),
                              ResetIndexOp)

    def test_drop_false_and_the_default_are_refused(self):
        # drop=False promotes the index to a column, changing the schema, and it
        # is also pandas' default, so it is refused rather than guessed at.
        self.assertIsNone(make_reset_index_op(_call("reset_index", drop=False)))
        self.assertIsNone(make_reset_index_op(_call("reset_index")))

    def test_other_arguments_are_refused(self):
        self.assertIsNone(make_reset_index_op(_call("reset_index", drop=True, level=0)))
        self.assertIsNone(make_reset_index_op(_call("reset_index", 0, drop=True)))

    def test_it_keeps_the_container_kind(self):
        call = _call("reset_index", drop=True, output_type=OutputType.SERIES)
        self.assertIs(OutputType.SERIES, make_reset_index_op(call).output_type)

    def test_pandas_renumbers_the_rows(self):
        df = pd.DataFrame({"a": [1, 2, 3]}).iloc[[2, 0]]
        self.assertEqual([0, 1], list(run_op(ResetIndexOp(), df).index))

    def test_polars_has_no_index_to_drop(self):
        # polars rows carry no labels, so the dropping form is identity there.
        with force_polars():
            frame = pl.DataFrame({"a": [1, 2, 3]})
            self.assertTrue(run_op(ResetIndexOp(), frame).equals(frame))


class TestStructureReadExtraction(unittest.TestCase):
    """`.shape` and `.columns` read a frame's structure into a metadata op."""

    def _read(self, read):
        frame = st.as_data_op(pd.DataFrame({"a": [1, 2]}))
        return optimize(read(frame), OptConfig(dataframe_ops=True))[-1]

    def test_shape_of_a_frame_or_series_is_a_scalar_read(self):
        # A tuple, not a frame: before, the read copied the container's kind.
        for kind, read in (("frame", lambda x: x.shape), ("series", lambda x: x["a"].shape)):
            with self.subTest(kind=kind):
                op = self._read(read)
                self.assertIsInstance(op, ShapeOp)
                self.assertIsInstance(op, MetadataOp)
                self.assertIs(OutputType.SCALAR, op.output_type)

    def test_columns_of_a_frame_is_a_column_names_read(self):
        # Not a SERIES: `df[df.columns[:2]]` would then read as a row mask.
        op = self._read(lambda x: x.columns)
        self.assertIsInstance(op, ColumnNamesOp)
        self.assertIs(OutputType.UNKNOWN, op.output_type)

    def test_a_series_has_no_columns(self):
        series = _inp(None)
        series.output_type = OutputType.SERIES
        read = GetAttrOp(attr_name="columns")
        read.inputs = [series]
        series.outputs = [read]
        self.assertIsNone(make_structure_read_op(read))

    def test_structure_reads_clone(self):
        for cls, kind in ((ShapeOp, OutputType.SCALAR), (ColumnNamesOp, OutputType.UNKNOWN)):
            with self.subTest(cls=cls.__name__):
                clone = cls().clone()
                self.assertIs(cls, type(clone))
                self.assertIs(kind, clone.output_type)


_FRAME = pd.DataFrame({"c": ["a", "b", "a"], "s": ["xy", "zw", "uv"], "n": [1, 2, 3]})
_STRUCTURE_READS = {
    "frame.shape": lambda x: x.shape,
    "series.shape": lambda x: x["n"].shape,
    "shape[0]": lambda x: x.shape[0],
    "columns": lambda x: x.columns,
    "columns.tolist()": lambda x: x.columns.tolist(),
    "x[columns[:2]]": lambda x: x[x.columns[:2]],
    "drop(columns[0])": lambda x: x.drop(columns=x.columns[0]),
}
_SELECTORS = {
    "pandas": lambda: FlagBasedSelector(backend="pandas"),
    "polars": lambda: FlagBasedSelector(backend="polars"),
    "greedy": lambda: get_implementation_selector("greedy"),
}


def _evaluate(dag, source, selector):
    plan, split, flagged = optimize_plan(dag, OptConfig(selector=_SELECTORS[selector]()),
                                         env={"x": source})
    return plan, SequentialScheduler(plan, split, flagged).evaluate()


@pytest.mark.parametrize("selector", _SELECTORS)
@pytest.mark.parametrize("name", _STRUCTURE_READS)
def test_structure_read_matches_pandas(name, selector):
    read = _STRUCTURE_READS[name]
    expected = read(_FRAME)
    with skrub.config_context(eager_data_ops=False):
        dag = read(skrub.var("x", _FRAME))
    _, result = _evaluate(dag, _FRAME, selector)
    if isinstance(result, pl.DataFrame):
        result = result.to_pandas()
    assert type(expected) is type(result)
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(expected, result, check_dtype=False)
    elif isinstance(expected, pd.Index):
        pd.testing.assert_index_equal(expected, result)
    else:
        assert expected == result


def test_column_names_of_a_polars_source_need_no_conversion():
    # `.tolist()` is a pandas-only call; the names it reads must not pin the
    # polars source to pandas and convert the whole frame.
    source = pl.from_pandas(_FRAME)
    with skrub.config_context(eager_data_ops=False):
        dag = skrub.var("x", source).columns.tolist()
    plan, result = _evaluate(dag, source, "greedy")
    assert ["c", "s", "n"] == result
    assert [] == [op for op in plan if isinstance(op, ConversionOp)]
    names = [op for op in plan if isinstance(op, ColumnNamesOp)]
    assert ["PolarsColumnNamesOp"] == [type(op).__name__ for op in names]
    assert "polars" == str(names[0].inputs[0]._frame_lib)


if __name__ == "__main__":
    unittest.main()
