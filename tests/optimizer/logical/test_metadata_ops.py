import unittest

import pandas as pd
import polars as pl
import pytest
import skrub

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize as optimize_plan
from stratum.optimizer.logical._metadata_ops import (
    ColumnNamesOp, MetadataOp, RenameOp, RowLabelsOp, ShapeOp, make_rename_op,
    make_reset_index_op, make_set_axis_op, make_set_index_op, make_structure_read_op,
    make_to_frame_op)
from stratum.optimizer.logical._ops import (
    GetAttrOp, MethodCallOp, OperandRef, OutputType)
from stratum.optimizer.physical import FlagBasedSelector
from stratum.optimizer.physical._conversion import ConversionOp, index_origin
from stratum.optimizer.physical._impl_selection import get_implementation_selector
from stratum.runtime._scheduler import SequentialScheduler
from .test_dataframe_ops import _inp, force_polars, optimize, run_op


def _call(method, *args, output_type=OutputType.FRAME, operands=0, **kwargs) -> MethodCallOp:
    """A recorded call on a receiver of kind ``output_type``, with ``operands``
    more inputs for its ``OperandRef`` arguments."""
    op = MethodCallOp(method, args=args, kwargs=kwargs)
    op.inputs = [_inp(None) for _ in range(1 + operands)]
    op.inputs[0].output_type = output_type
    for inp in op.inputs:
        inp.outputs = [op]
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
            "series index mapping": _call("rename", {0: 9}, output_type=OutputType.SERIES),
            "series name and more": _call("rename", "n", output_type=OutputType.SERIES,
                                          inplace=True),
            "series tuple name": _call("rename", ("a", "b"), output_type=OutputType.SERIES),
            # a grouped size() is typed as a frame but is a Series at run time.
            "scalar name": _call("rename", "name"),
            "graph-fed mapping": _call("rename", columns=OperandRef(1)),
        }
        for name, call in cases.items():
            with self.subTest(name):
                self.assertIsNone(make_rename_op(call))

    def test_a_series_name_relabels_its_column(self):
        for label in ("x", 3):
            with self.subTest(label=label):
                op = make_rename_op(_call("rename", label, output_type=OutputType.SERIES))
                self.assertEqual(((label,), False), (op.label, op.as_frame))
                self.assertIs(OutputType.SERIES, op.output_type)

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


class TestToFrame(unittest.TestCase):
    """`to_frame` turns a series into a frame: a rename, which may relabel its column."""

    def test_it_becomes_a_rename_to_a_frame(self):
        # No name keeps the series' own; a name, None included, relabels it.
        for kwargs, label in (({}, None), ({"name": "x"}, ("x",)), ({"name": None}, (None,)),
                              ({"name": 0}, (0,))):
            with self.subTest(kwargs=repr(kwargs)):
                op = make_to_frame_op(_call("to_frame", output_type=OutputType.SERIES,
                                            **kwargs))
                self.assertIsInstance(op, RenameOp)
                self.assertEqual((label, True), (op.label, op.as_frame))
                self.assertIs(OutputType.FRAME, op.output_type)

    def test_other_forms_stay_opaque(self):
        for call in (_call("to_frame", output_type=OutputType.FRAME),
                     _call("to_frame", ("a", "b"), output_type=OutputType.SERIES),
                     _call("to_frame", OperandRef(1), operands=1,
                           output_type=OutputType.SERIES)):
            with self.subTest(call=repr((call.args, call.inputs[0].output_type))):
                self.assertIsNone(make_to_frame_op(call))

    def test_pandas_labels_the_column_as_pandas_does(self):
        # The series' name, 0 for an unnamed one, or the label given, None included.
        for series in (pd.Series([1, 2], name="s", index=[5, 6]), pd.Series([1, 2])):
            for label in (None, ("x",), (None,)):
                with self.subTest(name=series.name, label=repr(label)):
                    expected = series.to_frame(*(label or ()))
                    pd.testing.assert_frame_equal(
                        expected, run_op(RenameOp(label=label, as_frame=True), series))

    def test_a_series_rename_and_a_to_frame_differ(self):
        # Same label, same input: one is a series, the other a frame.
        from stratum.optimizer._lineage import lineage_key
        rename, to_frame = RenameOp(label=("x",)), RenameOp(label=("x",), as_frame=True)
        self.assertNotEqual(lineage_key(rename, ["in"]), lineage_key(to_frame, ["in"]))
        self.assertNotEqual(rename.structure_key(), to_frame.structure_key())

    def test_polars_needs_a_string_label(self):
        from stratum.optimizer.physical._metadata_execs import PolarsRenameOp
        self.assertTrue(PolarsRenameOp.supports(RenameOp(label=("x",), as_frame=True), None))
        self.assertTrue(PolarsRenameOp.supports(RenameOp(label=("x",)), None))
        # pandas' label is not known to be the polars name.
        self.assertFalse(PolarsRenameOp.supports(RenameOp(as_frame=True), None))
        self.assertFalse(PolarsRenameOp.supports(RenameOp(label=(None,), as_frame=True), None))
        self.assertFalse(PolarsRenameOp.supports(RenameOp(label=(0,)), None))


class TestResetIndex(unittest.TestCase):
    """Only the dropping form of `reset_index` is representable."""

    def test_drop_true_becomes_a_reset_index_op(self):
        self.assertIsInstance(make_reset_index_op(_call("reset_index", drop=True)),
                              RowLabelsOp)

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
        self.assertEqual([0, 1], list(run_op(RowLabelsOp(), df).index))

    def test_polars_has_no_index_to_drop(self):
        # polars rows carry no labels, so the dropping form is identity there.
        with force_polars():
            frame = pl.DataFrame({"a": [1, 2, 3]})
            self.assertTrue(run_op(RowLabelsOp(), frame).equals(frame))


class TestSetIndex(unittest.TestCase):
    """`set_index` moves named columns into the row labels."""

    def test_column_names_become_keys(self):
        for keys in ("k", ["k"], ["k", "j"]):
            with self.subTest(keys=keys):
                op = make_set_index_op(_call("set_index", keys))
                self.assertIsInstance(op, RowLabelsOp)
                self.assertEqual(tuple([keys] if isinstance(keys, str) else keys), op.keys)
                self.assertIsNone(op.labels)

    def test_other_forms_stay_opaque(self):
        # Keeping the columns, appending, checking uniqueness, in place, keys that
        # are values rather than names, and a series receiver.
        for call in (_call("set_index", "k", drop=False), _call("set_index", "k", append=True),
                     _call("set_index", "k", verify_integrity=True),
                     _call("set_index", "k", inplace=True),
                     _call("set_index", OperandRef(1), operands=1),
                     _call("set_index", ["k", "k"]), _call("set_index", []),
                     _call("set_index", "k", output_type=OutputType.SERIES)):
            with self.subTest(call=repr((call.args, call.kwargs, call.inputs[0].output_type))):
                self.assertIsNone(make_set_index_op(call))

    def test_pandas_moves_the_columns(self):
        df = pd.DataFrame({"k": ["a", "b"], "j": [1, 2], "v": [3.0, 4.0]})
        for keys in (("k",), ("k", "j")):
            with self.subTest(keys=keys):
                pd.testing.assert_frame_equal(df.set_index(list(keys)),
                                              run_op(RowLabelsOp(keys=keys), df))

    def test_each_form_has_its_own_lineage_key(self):
        from stratum.optimizer._lineage import lineage_key
        forms = (RowLabelsOp(), RowLabelsOp(keys=("k",)), RowLabelsOp(keys=("k", "j")),
                 RowLabelsOp(labels=OperandRef(1)))
        keys = {lineage_key(op, ["in"]) for op in forms}
        self.assertEqual(len(forms), len(keys))

    def test_polars_refuses_new_labels(self):
        from stratum.optimizer.physical._metadata_execs import PolarsRowLabelsOp
        self.assertFalse(PolarsRowLabelsOp.supports(RowLabelsOp(keys=("k",)), None))
        self.assertFalse(PolarsRowLabelsOp.supports(RowLabelsOp(labels=OperandRef(1)), None))
        self.assertTrue(PolarsRowLabelsOp.supports(RowLabelsOp(), None))


class TestSetAxis(unittest.TestCase):
    """`set_axis` along the rows takes another op's values as the labels."""

    def test_computed_labels_become_labels(self):
        for kind in (OutputType.FRAME, OutputType.SERIES):
            for kwargs in ({}, {"axis": 0}, {"axis": "index"}):
                with self.subTest(kind=kind.name, kwargs=repr(kwargs)):
                    op = make_set_axis_op(_call("set_axis", OperandRef(1), operands=1,
                                                output_type=kind, **kwargs))
                    self.assertEqual((None, OperandRef(1)), (op.keys, op.labels))
                    self.assertIs(kind, op.output_type)

    def test_other_forms_stay_opaque(self):
        # Literal labels, the column axis, copy, and the receiver's own values.
        for call in (_call("set_axis", [0, 1]),
                     _call("set_axis", OperandRef(1), operands=1, axis=1),
                     _call("set_axis", OperandRef(1), operands=1, axis="columns"),
                     _call("set_axis", OperandRef(1), operands=1, copy=False),
                     _call("set_axis", OperandRef(0))):
            with self.subTest(call=repr((call.args, call.kwargs))):
                self.assertIsNone(make_set_axis_op(call))

    def test_pandas_relabels_by_position(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        labels = pd.Index([7, 5, 9])
        pd.testing.assert_frame_equal(df.set_axis(labels),
                                      run_op(RowLabelsOp(labels=OperandRef(1)), df, labels))
        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            run_op(RowLabelsOp(labels=OperandRef(1)), df, labels[:2])

    def test_it_takes_the_index_origin_of_the_frame_it_reads(self):
        # `y.set_axis(x.index)` pairs with `x` by position, as `x`'s own rows do;
        # labels from any other op are new.
        from stratum.optimizer.logical._index_ops import IndexAccessOp
        x, y = _inp(None), _inp(None)
        x_index = IndexAccessOp(inputs=[x])
        relabelled = RowLabelsOp(labels=OperandRef(1), inputs=[y, x_index])
        self.assertEqual(index_origin(x), index_origin(relabelled))
        self.assertNotEqual(index_origin(y), index_origin(relabelled))
        column = RowLabelsOp(labels=OperandRef(1), inputs=[y, _inp(None)])
        self.assertEqual(id(column), index_origin(column))


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
    plan, split = optimize_plan(dag, OptConfig(selector=_SELECTORS[selector]()),
                                         env={"x": source})
    return plan, SequentialScheduler(plan, split).evaluate()


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


_RELABELLINGS = {
    "set_index": lambda x: x.set_index("c")["n"],
    "set_index(list)": lambda x: x.set_index(["c", "s"]),
    "set_axis(other.index)": lambda x: x.assign(m=x["n"].sort_values(ascending=False)
                                                .set_axis(x.index)),
    "reset_index": lambda x: x[x["n"] > 1].reset_index(drop=True),
    "to_frame(name)": lambda x: x["c"].to_frame("k").assign(m=x["n"]),
    "to_frame()": lambda x: x["c"].to_frame(),
    "to_frame(None)": lambda x: x["c"].to_frame(None),
    "series rename": lambda x: x["n"].rename("z"),
}


@pytest.mark.parametrize("selector", _SELECTORS)
@pytest.mark.parametrize("name", _RELABELLINGS)
def test_relabelling_matches_pandas(name, selector):
    if selector == "polars" and name.startswith(("set_", "to_frame()", "to_frame(None)")):
        pytest.skip("runs on pandas only, which the polars selector forbids")
    relabel = _RELABELLINGS[name]
    expected = relabel(_FRAME)
    with skrub.config_context(eager_data_ops=False):
        dag = relabel(skrub.var("x", _FRAME))
    plan, result = _evaluate(dag, _FRAME, selector)
    assert any(isinstance(op, (RowLabelsOp, RenameOp)) for op in plan)
    if isinstance(result, (pl.DataFrame, pl.Series)):
        # A polars result carries no labels (yet): its values and order must match.
        result, expected = result.to_pandas(), expected.reset_index(drop=True)
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(expected, result, check_dtype=False)
    else:
        pd.testing.assert_frame_equal(expected, result, check_dtype=False)


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
