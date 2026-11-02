"""Frame representations on plan edges (#226): checked at planning, not at run time."""
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum as st
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._base import FRAME_TYPES
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._registry import get_default_physical_registry
from stratum.optimizer.physical._representation import (
    FrameLib, _polars_safe_estimators, frame_lib)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "x.parquet"
    pd.DataFrame({"x": [1, -2, 3], "s": ["1.0", "2", "3.0"]}).to_parquet(path)
    with skrub.config_context(eager_data_ops=False):
        yield st.as_data_op(str(path)).skb.apply_func(pd.read_parquet)


def _optimize(plan, selector):
    with st.config(scheduler=True, implementation_selector=selector):
        return optimize(plan, env=get_data(plan))[0]


def test_an_opaque_call_fed_polars_gets_converted_input(source):
    # #226: `mask` stays the recorded pandas call; the greedy plan converts its
    # polars inputs explicitly (#227).
    plan = source.assign(c=source["x"].mask(source["x"] > 0))
    ops = _optimize(plan, "greedy")
    mask = next(op for op in ops if type(op).__name__ == "MethodCallExec")
    assert all(isinstance(i, ConversionOp) and i.target is FrameLib.PANDAS
               for i in mask.inputs)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    with st.config(scheduler=True):
        expected = evaluate(plan)
    assert isinstance(result, pl.DataFrame), "mask keeps the rows: the assign stays on polars"
    pd.testing.assert_frame_equal(result.to_pandas(), expected, check_dtype=False)


def test_a_translated_expression_runs_on_polars(source):
    # #226, second example: the regex replace now compiles exactly.
    plan = source.assign(c=source["s"].str.replace(r"\.0$", "", regex=True))
    ops = _optimize(plan, "greedy")
    assert not any(isinstance(op, ConversionOp) for op in ops)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    assert isinstance(result, pl.DataFrame)
    assert result["c"].to_list() == ["1", "2", "3"]


def test_an_untranslatable_expression_runs_on_pandas(source):
    plan = source.assign(c=source["s"].str.replace(r"(\d)", r"\1\1", regex=True))
    ops = _optimize(plan, "greedy")
    assign = next(op for op in ops if type(op).__name__.endswith("AssignMapOp"))
    assert type(assign).__name__ == "PandasAssignMapOp"
    assert "not portable" in dict(assign._refusals)["polars"]
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    with st.config(scheduler=True):
        expected = evaluate(plan)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_every_frame_edge_has_a_library(source, selector):
    plan = source.assign(c=source["x"] * 2)
    ops = _optimize(plan, selector)
    expected = FrameLib.PANDAS if selector == "default" else FrameLib.POLARS
    libs = {frame_lib(op) for op in ops if op.output_type in FRAME_TYPES}
    assert libs == {expected}


def test_registered_impls_declare_their_backends_library():
    registry = get_default_physical_registry()
    for op_type in registry.op_types():
        for impl in registry.candidates_for(op_type):
            if impl.impl_class is None:
                continue
            expected = {"pandas": FrameLib.PANDAS, "polars": FrameLib.POLARS}.get(
                impl.backend_name)
            assert impl.impl_class.frame_lib == expected, impl


# --- Estimators read what the pandas run gives them ----------------------------

_FEATURES = pd.DataFrame({"c": ["C1", "C2", "C1", "C3"] * 10,
                          "x": [float(i % 7) for i in range(40)],
                          "y": [i % 2 for i in range(40)]})


def _estimator_inputs(ops):
    return {type(op.original_estimator).__name__: [frame_lib(i) for i in op.inputs]
            for op in ops if hasattr(op, "original_estimator")}


def test_an_estimator_reads_pandas_under_greedy(tmp_path):
    # skrub's ToCategorical makes polars categoricals of a polars frame, which
    # HistGradientBoosting does not take as categorical: it fails on "C1".
    from sklearn.ensemble import HistGradientBoostingClassifier
    from skrub import TableVectorizer, ToCategorical
    path = tmp_path / "f.parquet"
    _FEATURES.to_parquet(path)
    with skrub.config_context(eager_data_ops=False):
        data = st.as_data_op(str(path)).skb.apply_func(pd.read_parquet)
        kept = data[data["x"] > 0]
        X, y = kept[["c", "x"]].skb.mark_as_X(), kept["y"].skb.mark_as_y()
        plan = (X.skb.apply(TableVectorizer(low_cardinality=ToCategorical()))
                .skb.apply(HistGradientBoostingClassifier(max_iter=5), y=y))
    ops = _optimize(plan, "greedy")
    assert _estimator_inputs(ops)["HistGradientBoostingClassifier"][0] is FrameLib.PANDAS
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    with st.config(scheduler=True):
        expected = evaluate(plan)
    assert list(result) == list(expected)


def test_an_allowed_estimator_reads_polars_given_dtypes_it_reads_alike(source):
    from sklearn.preprocessing import StandardScaler
    with skrub.config_context(eager_data_ops=False):
        known = skrub.var("frame", pd.DataFrame({"x": [1.0, -2.0, 3.0]}))
        plan = known[["x"]].skb.apply(StandardScaler())
    ops = _optimize(plan, "greedy")
    assert _estimator_inputs(ops)["StandardScaler"][0] is FrameLib.POLARS
    # A parquet read's dtypes are not known when planning.
    ops = _optimize(source[["x"]].skb.apply(StandardScaler()), "greedy")
    assert _estimator_inputs(ops)["StandardScaler"][0] is FrameLib.PANDAS


@pytest.mark.parametrize("estimator", sorted(_polars_safe_estimators(), key=lambda c: c.__name__),
                         ids=lambda c: c.__name__)
def test_an_allowed_estimator_answers_as_on_pandas(estimator):
    # Every estimator allowed to read polars, fitted through skrub on the same
    # data as a pandas and as a polars frame, missing values included where it
    # takes them.
    import numpy as np
    from sklearn.base import is_classifier, is_regressor
    rng = np.random.default_rng(0)
    # The dtypes scikit-learn reads alike from both libraries (see _read_alike).
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.integers(0, 9, 200).astype(float),
                      "c": rng.normal(size=200) * 1e3, "d": rng.integers(0, 9, 200),
                      "e": rng.normal(size=200) > 0})
    noise = rng.normal(size=200)
    y = pd.Series((noise > 0).astype(int) if is_classifier(estimator()) else noise, name="y")
    if not (is_classifier(estimator()) or is_regressor(estimator())):
        X.loc[::7, "a"] = np.nan  # the linear models reject missing values

    def fitted(X, y):
        Xv, yv = skrub.var("X", X).skb.mark_as_X(), skrub.var("y", y).skb.mark_as_y()
        out = Xv.skb.apply(estimator(), y=yv).skb.eval()
        return np.asarray(out.to_numpy() if hasattr(out, "to_numpy") else out)

    on_pandas, on_polars = fitted(X, y), fitted(pl.from_pandas(X), pl.from_pandas(y))
    assert on_pandas.dtype == on_polars.dtype
    assert on_pandas.shape == on_polars.shape
    assert np.array_equal(on_pandas, on_polars, equal_nan=True)


def test_an_allowed_estimator_reads_pandas_where_it_would_widen_floats(tmp_path):
    # scikit-learn keeps pandas' float32 but reads polars' as float64.
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    path = tmp_path / "f.parquet"
    pd.DataFrame({"x": np.arange(5, dtype="float32") / 3}).to_parquet(path)
    with skrub.config_context(eager_data_ops=False):
        plan = st.as_data_op(str(path)).skb.apply_func(pd.read_parquet).skb.apply(StandardScaler())
    ops = _optimize(plan, "greedy")
    assert _estimator_inputs(ops)["StandardScaler"][0] is FrameLib.PANDAS
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    with st.config(scheduler=True):
        expected = evaluate(plan)
    pd.testing.assert_frame_equal(result, expected)


def test_a_frame_that_may_repeat_a_column_name_stays_on_pandas():
    # pandas keeps two columns of one name, which polars cannot hold; the drop
    # after the rename removes both.
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "k": [5, 6]})
    with skrub.config_context(eager_data_ops=False):
        data = skrub.var("frame", frame)
        name = data.skb.apply_func(lambda df: "pad")
        plan = data.rename(columns={"a": name, "b": name}).drop(columns=["pad"]).assign(m=1)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    with st.config(scheduler=True):
        expected = evaluate(plan)
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
