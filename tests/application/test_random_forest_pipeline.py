"""DataOps coverage for exact and histogram random-forest physical operators."""

import io
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import stratum as st
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._ops import PredictorOp
from stratum.optimizer.physical._predictor_execs import (
    RustHistogramRandomForestClassifier,
    SklearnRandomForest,
)
from stratum.optimizer.physical._source_execs import PolarsInMemoryFrame


def capture_std_out(capfd):
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    return (captured.out or "") + (captured.err or "")


def _classification_frame(n_samples=360):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=0,
        n_classes=3,
        class_sep=1.2,
        random_state=17,
    )
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame["target"] = np.asarray([f"class-{label}" for label in y])
    return frame


def _pipeline(*, criterion="gini"):
    data = st.as_data_op(_classification_frame())
    y = data["target"].skb.mark_as_y()
    X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=11),
        split_kwargs={},
    )
    model = RandomForestClassifier(
        n_estimators=11,
        max_depth=7,
        criterion=criterion,
        random_state=5,
    )
    return X.skb.apply(model, y=y)


def _predictor_for(selector, *, criterion="gini"):
    with st.config(implementation_selector=selector,
                   num_threads=2,
                   scheduler=True):
        ops, *_ = optimize(_pipeline(criterion=criterion))
    (predictor,) = [op for op in ops if isinstance(op, PredictorOp)]
    return predictor, ops


def test_default_keeps_sklearn_and_greedy_binds_histogram():
    default, _ = _predictor_for("default")
    hist, greedy_ops = _predictor_for("greedy")

    assert type(default) is SklearnRandomForest
    assert type(hist) is RustHistogramRandomForestClassifier
    assert hist.original_estimator._stratum_forest_binding[0] == "histogram"
    assert hist.original_estimator._stratum_forest_fit_args == (128,)
    # Greedy may produce a Polars frame; adapter validation normalizes it before Rust.
    assert any(isinstance(op, PolarsInMemoryFrame) for op in greedy_ops)


def test_unsupported_random_forest_parameters_keep_sklearn():
    predictor, _ = _predictor_for("greedy", criterion="entropy")

    assert type(predictor) is SklearnRandomForest


def _score_pipeline(selector, capfd):
    with st.config(
        implementation_selector=selector,
        num_threads=2,
        scheduler=True,
        rust_backend=True,
        allow_patch=True,
        explain=False, # set it True to print physical plan
        debug_timing=True,
    ):
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            search = _pipeline().skb.make_grid_search(
                n_jobs=1,
                fitted=True,
                refit=False,
                scoring="accuracy",
            )

    output_str = captured_output.getvalue()
    if output_str:
        with capfd.disabled():
            print(output_str)

    return np.asarray(search.results_["scores"], dtype=float).mean()


def test_both_random_forest_policies_score_end_to_end(capfd):
    scores = {}
    for selector in ("default", "greedy"):
        scores[selector] = _score_pipeline(selector, capfd)

    combined_output = capture_std_out(capfd)
    assert "[rust]" in combined_output
    assert np.isfinite(list(scores.values())).all()
    assert scores["default"] - scores["greedy"] <= 0.03
