"""DataOps standardization parity through cross-validation."""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import stratum as st
from sklearn.preprocessing import StandardScaler
from stratum import _rust_backend as rb
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._transform_execs import (
    RustStandardScaler,
    SklearnStandardScaler,
    StandardScalerOp,
)


def _pipeline(wrapped):
    X = np.random.default_rng(30).normal(size=(90, 4))
    X[:, 0] *= 100
    y = X @ np.array([0.02, 2, -1, 3])
    features = st.var("X", X).skb.mark_as_X(
        cv=KFold(n_splits=3, shuffle=True, random_state=4),
        split_kwargs={},
    )
    target = st.var("y", y).skb.mark_as_y()
    mean = st.as_data_op(True) if wrapped else True
    return features.skb.apply(StandardScaler(with_mean=mean)).skb.apply(Ridge(alpha=0.1), y=target)


@pytest.mark.skipif(rb.standard_scale_fit is None, reason="native scaler unavailable")
@pytest.mark.parametrize("wrapped", [False, True])
def test_standard_scaler_pipeline_selects_and_scores_both_backends(monkeypatch, capfd, wrapped):
    calls = []
    native_fit = rb.standard_scale_fit

    def fit(*args, **kwargs):
        assert kwargs["worker_budget"] == 2
        calls.append(True)
        return native_fit(*args, **kwargs)

    monkeypatch.setattr(rb, "standard_scale_fit", fit)
    scores = {}
    for selector, expected in (
        ("default", SklearnStandardScaler),
        ("greedy", RustStandardScaler),
    ):
        with st.config(
            implementation_selector=selector,
            scheduler=True,
            rust_backend=False,
            allow_patch=False,
            num_threads=2,
            explain=False,
        ):  # Set True to print the physical plan.
            pipeline = _pipeline(wrapped)
            ops, *_ = optimize(pipeline, env=pipeline.skb.get_data())
            scalers = [op for op in ops if isinstance(op, StandardScalerOp)]
            assert sorted(op.mode for op in scalers) == ["fit_transform", "transform"]
            for scaler in scalers:
                assert type(scaler) is expected
                if selector == "greedy":
                    assert scaler.original_estimator._stratum_worker_budget == 2
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                search = pipeline.skb.make_grid_search(
                    n_jobs=1,
                    fitted=True,
                    refit=False,
                    scoring="r2",
                )
            scores[selector] = np.asarray(search.results_["scores"], dtype=float)
        output_str = captured_output.getvalue()
        if output_str:
            with capfd.disabled():
                print(output_str)
        if selector == "default":
            assert not calls
    assert len(calls) >= 3
    np.testing.assert_allclose(scores["default"], scores["greedy"], atol=1e-12)
    assert np.min(scores["greedy"]) > 0.99
