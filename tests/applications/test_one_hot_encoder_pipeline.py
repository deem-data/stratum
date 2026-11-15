import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import stratum as st
from stratum import _rust_backend as rb
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._transform_execs import (
    OneHotEncoderOp, RustOneHotEncoder, SklearnOneHotEncoder,
)


def _pipeline(workload, dtype):
    positive = np.tile([0, 1], 45)
    if workload == "spaceship":
        data = pd.DataFrame({
            "Cabin": np.where(positive, "B/20/S", "A/10/P"),
            "HomePlanet": np.where(positive, "Europa", "Earth"),
            "Transported": positive,
        })
    else:
        data = pd.DataFrame({
            "hostname": [f"domain-{i}.{'org' if label else 'com'}" for i, label in enumerate(positive)],
            "tracked": positive,
        })
    raw = st.var("data", data)
    label = "Transported" if workload == "spaceship" else "tracked"
    y = raw[label].skb.mark_as_y()
    X = raw.drop(columns=label).skb.mark_as_X(
        cv=StratifiedKFold(3, shuffle=True, random_state=42), split_kwargs={},
    )
    if workload == "spaceship":
        parts = X["Cabin"].str.split("/")
        features = X.assign(Deck=parts.str[0], Side=parts.str[2]).drop(columns="Cabin")
    else:
        # Unique hostnames guarantee unseen categories in every validation fold;
        # the TLD signal must survive their all-zero encoding.
        features = X.assign(tld=X["hostname"].str.split(".").str[-1])
    encoded = features.skb.apply(OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=dtype,
    ))
    return encoded.skb.apply(LogisticRegression(C=10), y=y)


@pytest.mark.skipif(not rb.HAVE_RUST, reason="native OHE unavailable")
@pytest.mark.parametrize("workload", ["spaceship", "tracker"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_trajectory_pipeline_selector_and_cv_parity(monkeypatch, workload, dtype):
    calls = []
    native = rb.ohe_transform

    def emit(*args, **kwargs):
        calls.append(kwargs["dtype"])
        return native(*args, **kwargs)

    monkeypatch.setattr(rb, "ohe_transform", emit)
    scores = {}
    for selector, expected in (("default", SklearnOneHotEncoder), ("greedy", RustOneHotEncoder)):
        with st.config(implementation_selector=selector, scheduler=True,
                       rust_backend=False, allow_patch=False, num_threads=2,
                       explain=False), st.config_context(eager_data_ops=False):
            pred = _pipeline(workload, dtype)
            ops, *_ = optimize(pred, env=pred.skb.get_data())
            encoders = [op for op in ops if isinstance(op, OneHotEncoderOp)]
            assert sorted(op.mode for op in encoders) == ["fit_transform", "transform"]
            assert all(type(op) is expected for op in encoders)
            result = pred.skb.make_grid_search(
                n_jobs=1, fitted=True, refit=False, scoring="accuracy",
            )
            scores[selector] = np.asarray(result.results_["scores"], dtype=float)
        if selector == "default":
            assert not calls
    assert len(calls) >= 6
    assert set(calls) == {np.dtype(dtype).name}
    np.testing.assert_array_equal(scores["default"], scores["greedy"])
    np.testing.assert_array_equal(scores["greedy"], 1.0)
