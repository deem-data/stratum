"""DataOps standardization parity through cross-validation."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import stratum as st
from sklearn.preprocessing import StandardScaler
from stratum import _rust_backend as rb
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._transform_execs import (
    RustStandardScaler, SklearnStandardScaler, StandardScalerOp,
)


def _pipeline():
    X = np.random.default_rng(30).normal(size=(90, 4))
    X[:, 0] *= 100
    y = X @ np.array([0.02, 2, -1, 3])
    features = st.var("X", X).skb.mark_as_X(
        cv=KFold(n_splits=3, shuffle=True, random_state=4), split_kwargs={},
    )
    target = st.var("y", y).skb.mark_as_y()
    return features.skb.apply(StandardScaler()).skb.apply(Ridge(alpha=0.1), y=target)


@pytest.mark.skipif(rb.standard_scale_fit is None, reason="native scaler unavailable")
def test_standard_scaler_pipeline_selects_and_scores_both_backends(monkeypatch):
    calls = []
    native_fit = rb.standard_scale_fit

    def fit(*args, **kwargs):
        assert kwargs["worker_budget"] == 2
        calls.append(True)
        return native_fit(*args, **kwargs)

    monkeypatch.setattr(rb, "standard_scale_fit", fit)
    scores = {}
    for selector, expected in (("default", SklearnStandardScaler),
                               ("greedy", RustStandardScaler)):
        with st.config(implementation_selector=selector, scheduler=True,
                       rust_backend=False, allow_patch=False, num_threads=2):
            pipeline = _pipeline()
            ops, *_ = optimize(pipeline, env=pipeline.skb.get_data())
            (scaler,) = [op for op in ops if isinstance(op, StandardScalerOp)]
            assert type(scaler) is expected
            if selector == "greedy":
                assert scaler.original_estimator._stratum_worker_budget == 2
            search = pipeline.skb.make_grid_search(
                n_jobs=1, fitted=True, refit=False, scoring="r2",
            )
            scores[selector] = np.asarray(search.results_["scores"], dtype=float)
        if selector == "default":
            assert not calls
    assert len(calls) >= 3
    np.testing.assert_allclose(scores["default"], scores["greedy"], atol=1e-12)
    assert np.min(scores["greedy"]) > 0.99
