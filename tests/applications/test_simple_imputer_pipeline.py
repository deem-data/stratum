import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import stratum as st
from stratum import _rust_backend as rb
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._transform_execs import (
    SimpleImputerOp, RustSimpleImputer, SklearnSimpleImputer,
)


def _pipeline(workload, keep):
    rng = np.random.default_rng(7)
    label = np.tile([0, 1], 90)
    if workload == "spaceship":
        df = pd.DataFrame({
            "Age": 20 + label * 40 + rng.normal(size=len(label)),
            "RoomService": rng.integers(0, 100, len(label)).astype(float),
            "Spa": np.nan,
        })
    else:
        df = pd.DataFrame({
            "violations": label * 20. + rng.integers(0, 4, len(label)),
            "days_since": rng.integers(0, 365, len(label)).astype(float),
            "unused_event_stream": np.nan,
        })
    df.iloc[::7, 0] = np.nan
    df.iloc[::5, 1] = np.nan
    raw = st.var("data", df.assign(target=label))
    y = raw["target"].skb.mark_as_y()
    X = raw.drop(columns="target").skb.mark_as_X(
        cv=StratifiedKFold(3, shuffle=True, random_state=42), split_kwargs={},
    )
    X = X.skb.apply(SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=keep,
    ))
    return X.skb.apply(LogisticRegression(max_iter=500), y=y)


@pytest.mark.parametrize("workload", ["spaceship", "housing"])
@pytest.mark.parametrize("keep", [False, True])
def test_fold_local_imputation_selector_parity(monkeypatch, capfd, workload, keep):
    fits, transforms = [], []
    fit, transform = rb.simple_imputer_fit, rb.simple_imputer_transform

    def record_fit(X, **kwargs):
        assert kwargs["worker_budget"] == 2
        fits.append(X.shape)
        return fit(X, **kwargs)

    def record_transform(*args, **kwargs):
        assert kwargs["worker_budget"] == 2
        transforms.append(True)
        return transform(*args, **kwargs)

    monkeypatch.setattr(rb, "simple_imputer_fit", record_fit)
    monkeypatch.setattr(rb, "simple_imputer_transform", record_transform)
    scores = {}
    for selector, expected in (("default", SklearnSimpleImputer), ("greedy", RustSimpleImputer)):
        # Set explain=True to print the physical plan.
        with st.config(implementation_selector=selector, scheduler=True,
                       rust_backend=False, allow_patch=False, num_threads=2,
                       explain=False), st.config_context(eager_data_ops=False):
            pred = _pipeline(workload, keep)
            ops, *_ = optimize(pred, env=pred.skb.get_data())
            imputers = [op for op in ops if isinstance(op, SimpleImputerOp)]
            assert sorted(op.mode for op in imputers) == ["fit_transform", "transform"]
            assert all(type(op) is expected for op in imputers)
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                search = pred.skb.make_grid_search(n_jobs=1, fitted=True, refit=False, scoring="accuracy")
            scores[selector] = np.asarray(search.results_["scores"], dtype=float)
        output_str = captured_output.getvalue()
        if output_str:
            with capfd.disabled():
                print(output_str)
        if selector == "default":
            assert not fits and not transforms
    assert len(fits) == 3
    assert all(shape == (120, 3) for shape in fits)
    assert len(transforms) == 6
    np.testing.assert_array_equal(scores["default"], scores["greedy"])
    assert np.min(scores["greedy"]) > 0.85
