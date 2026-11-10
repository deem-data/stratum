"""CountVectorizer lowering, conservative eligibility and selection."""
from dataclasses import replace

import pytest
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer

import stratum as st
from stratum import _rust_backend as rb
from stratum.adapters.count_vectorizer import supports_rust_count_vectorizer
from stratum.optimizer.logical._ops import TransformerOp
from stratum.optimizer.physical._impl_selection import bind_op
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._transform_execs import (
    CountVectorizerOp, RustCountVectorizer, SklearnCountVectorizer, lower_transformer,
)


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_lower_and_select(selector):
    ctx = replace(PlanContext.from_flags(), implementation_selector=selector, parallelism=2)
    op = lower_transformer(TransformerOp(estimator=CountVectorizer()), ctx)
    assert type(op) is CountVectorizerOp
    bind_op(op, ctx)
    expected = RustCountVectorizer if selector == "greedy" and rb.count_vectorize_fit else SklearnCountVectorizer
    assert type(op) is expected
    if expected is RustCountVectorizer:
        cloned = clone(op.estimator)
        assert cloned._stratum_force_rust
        assert cloned._stratum_worker_budget == 2
        assert cloned.get_params()["token_pattern"] == CountVectorizer().token_pattern


def test_missing_runtime_uses_sklearn(monkeypatch):
    monkeypatch.setattr(rb, "count_vectorize_fit", None)
    op = CountVectorizerOp(estimator=CountVectorizer())
    bind_op(op, replace(PlanContext.from_flags(), implementation_selector="greedy"))
    assert type(op) is SklearnCountVectorizer


@pytest.mark.parametrize("parameter,value", [
    ("lowercase", True), ("max_df", 1.0), ("analyzer", "word"),
    ("dtype", None), ("stop_words", None), ("ngram_range", (1, 1)),
])
def test_deferred_parameters_are_not_evaluated(monkeypatch, parameter, value):
    monkeypatch.setattr(rb, "HAVE_RUST", True)
    for name in ("count_vectorize_fit", "count_vectorize_transform", "count_vectorize_fit_transform"):
        monkeypatch.setattr(rb, name, object())
    estimator = CountVectorizer(**{parameter: st.as_data_op(value)})
    assert not supports_rust_count_vectorizer(estimator)[0]


def test_custom_subclass_keeps_its_behavior():
    class CustomVectorizer(CountVectorizer):
        pass

    op = lower_transformer(TransformerOp(estimator=CustomVectorizer()), PlanContext.from_flags())
    assert not isinstance(op, CountVectorizerOp)
