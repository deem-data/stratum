"""DataOps text counting parity through cross-validation."""
import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import stratum as st
from stratum import _rust_backend as rb
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._transform_execs import (
    CountVectorizerOp, RustCountVectorizer, SklearnCountVectorizer,
)


def _pipeline():
    documents = np.array(["bright sunny warm", "dark rainy cold"] * 30)
    features = st.var("documents", documents).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=4), split_kwargs={},
    )
    target = st.var("target", np.tile([0, 1], 30)).skb.mark_as_y()
    counts = features.skb.apply(CountVectorizer(), no_wrap=True)
    return counts.skb.apply(LogisticRegression(), y=target)


@pytest.mark.skipif(rb.count_vectorize_fit_transform is None, reason="native counts unavailable")
def test_count_vectorizer_pipeline_selects_and_scores_both_backends(monkeypatch):
    calls = []
    models = []
    transforms = []
    native_fit = rb.count_vectorize_fit_transform
    native_transform = rb.count_vectorize_transform

    def fit(*args, **kwargs):
        assert kwargs["worker_budget"] == 2
        calls.append(True)
        result = native_fit(*args, **kwargs)
        models.append(result[0])
        return result

    def transform(model, *args, **kwargs):
        assert any(model is fitted for fitted in models)
        assert type(model).__name__ == "_CountVectorizerModelHandle"
        assert kwargs["worker_budget"] == 2
        transforms.append(model)
        return native_transform(model, *args, **kwargs)

    monkeypatch.setattr(rb, "count_vectorize_fit_transform", fit)
    monkeypatch.setattr(rb, "count_vectorize_transform", transform)
    scores = {}
    for selector, expected in (("default", SklearnCountVectorizer), ("greedy", RustCountVectorizer)):
        with st.config(implementation_selector=selector, scheduler=True,
                       rust_backend=False, allow_patch=False, num_threads=2):
            pipeline = _pipeline()
            ops, *_ = optimize(pipeline, env=pipeline.skb.get_data())
            (vectorizer,) = [op for op in ops if isinstance(op, CountVectorizerOp)]
            assert type(vectorizer) is expected
            search = pipeline.skb.make_grid_search(
                n_jobs=1, fitted=True, refit=False, scoring="accuracy",
            )
            scores[selector] = np.asarray(search.results_["scores"], dtype=float)
        if selector == "default":
            assert not calls
    assert len(calls) >= 3
    assert len(transforms) >= 3
    np.testing.assert_array_equal(scores["default"], scores["greedy"])
    np.testing.assert_array_equal(scores["greedy"], 1.0)
