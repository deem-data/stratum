import sys

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skrub import StringEncoder, TableVectorizer, selectors

import stratum as st
from stratum.adapters.one_hot_encoder import (
    RustyOneHotEncoder,
    supports_rust_one_hot_encoder,
)
from stratum.adapters.string_encoder import (
    RustyStringEncoder,
    supports_rust_string_encoder,
)
from stratum.adapters.table_vectorizer import (
    StratumFusedTableVectorizer as FusedTableVectorizerAdapter,
)
from stratum.optimizer.ir._ops import TransformerOp
from stratum.optimizer.physical._impl_selection import (
    DefaultImplementationSelector,
    GreedyImplementationSelector,
    select_implementations,
)
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._transform_execs import (
    SkrubTableVectorizer,
    StratumTableVectorizer,
    StratumFusedTableVectorizer,
    StringEncoderOp,
    TableVectorizerOp,
    lower_transformer,
)


def capture_std_out(capfd):
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    return (captured.out or "") + (captured.err or "")


def _ctx():
    return PlanContext(
        backend="pandas",
        pandas_query=False,
        rechunk=True,
        parallelism=1,
        rust_backend=False,
        allow_patch=True,
    )


def _op(vectorizer):
    # `no_wrap` is the mode used when TableVectorizer receives the complete
    # dataframe. `selectors.all()` satisfies Skrub's no-wrap validation.
    return TableVectorizerOp(
        estimator=vectorizer,
        cols=selectors.all(),
        how="no_wrap",
    )


def _rust_table_vectorizer_supported():
    vectorizer = TableVectorizer()
    return (
        supports_rust_one_hot_encoder(vectorizer.low_cardinality)[0]
        or supports_rust_string_encoder(vectorizer.high_cardinality)[0]
    )


def test_table_vectorizer_lowering_does_not_change_string_encoder_lowering():
    table_op = TransformerOp(estimator=TableVectorizer())
    lowered = lower_transformer(table_op, _ctx())
    assert isinstance(lowered, TableVectorizerOp)
    assert not isinstance(lowered, StringEncoderOp)

    string_op = TransformerOp(estimator=StringEncoder())
    lowered_string = lower_transformer(string_op, _ctx())
    assert isinstance(lowered_string, StringEncoderOp)


def test_default_selector_binds_the_unchanged_skrub_table_vectorizer():
    op = _op(TableVectorizer())
    select_implementations(
        op, _ctx(), selector=DefaultImplementationSelector()
    )

    assert isinstance(op, SkrubTableVectorizer)
    assert type(op.estimator.low_cardinality) is OneHotEncoder
    assert type(op.estimator.high_cardinality) is StringEncoder


@pytest.mark.skipif(
    not _rust_table_vectorizer_supported(),
    reason="Rust TableVectorizer leaf runtime is unavailable",
)
def test_greedy_selector_binds_fused_stratum_table_vectorizer():
    unsupported_high = StringEncoder(analyzer="word", n_components=2)
    op = _op(TableVectorizer(high_cardinality=unsupported_high))
    select_implementations(
        op, _ctx(), selector=GreedyImplementationSelector()
    )

    assert isinstance(op, StratumFusedTableVectorizer)
    assert isinstance(op.estimator, FusedTableVectorizerAdapter)
    assert isinstance(op.original_estimator, FusedTableVectorizerAdapter)
    assert type(op.estimator.low_cardinality) is OneHotEncoder
    assert type(op.estimator.high_cardinality) is StringEncoder

    # The selected estimator is still an ordinary cloneable sklearn estimator.
    assert clone(op.estimator) is not op.estimator
    assert clone(op.original_estimator) is not op.original_estimator


@pytest.mark.skipif(
    not _rust_table_vectorizer_supported(),
    reason="Rust TableVectorizer leaf runtime is unavailable",
)
def test_greedy_selector_binds_partial_stratum_table_vectorizer():
    high = StringEncoder(n_components=2)
    op = _op(
        TableVectorizer(
            low_cardinality=StandardScaler(),
            high_cardinality=high,
        )
    )

    select_implementations(
        op, _ctx(), selector=GreedyImplementationSelector()
    )

    assert isinstance(op, StratumTableVectorizer)
    assert isinstance(op.estimator.high_cardinality, RustyStringEncoder)
    assert isinstance(op.estimator.low_cardinality, StandardScaler)
    assert op.estimator.high_cardinality._stratum_force_rust


def test_unsupported_table_vectorizer_configuration_keeps_reference_candidate():
    op = _op(
        TableVectorizer(
            specific_transformers=[("passthrough", ["value"])]
        )
    )
    select_implementations(
        op, _ctx(), selector=GreedyImplementationSelector()
    )

    assert isinstance(op, SkrubTableVectorizer)


@pytest.mark.skipif(
    not _rust_table_vectorizer_supported(),
    reason="Rust TableVectorizer leaf runtime is unavailable",
)
def test_fused_stratum_table_vectorizer_fits_and_transforms_with_rust_leaves(capfd):
    vectorizer = TableVectorizer(
        cardinality_threshold=3,
        high_cardinality=StringEncoder(n_components=2),
    )
    op = _op(vectorizer)
    select_implementations(
        op, _ctx(), selector=GreedyImplementationSelector()
    )
    assert isinstance(op, StratumFusedTableVectorizer)
    assert isinstance(op.estimator, FusedTableVectorizerAdapter)

    train = pd.DataFrame(
        {
            "low": ["a", "b", "a", "b"],
            "high": ["alpha", "bravo", "charlie", "delta"],
            "number": [1, 2, 3, 4],
        }
    )
    test = pd.DataFrame(
        {
            "low": ["b", "a"],
            "high": ["echo", "foxtrot"],
            "number": [5, 6],
        }
    )

    # The selected fused estimator binds Rust leaves independently of the
    # public runtime flags used during execution.
    with st.config(
        rust_backend=False,
        allow_patch=False,
        debug_timing=True,
    ):
        fitted = op.process("fit_transform", [train])
        transformed = op.process("predict", [test])

    assert "[rust]" in capture_std_out(capfd)
    assert fitted.shape[1] == transformed.shape[1]
    assert list(fitted.columns) == list(transformed.columns)
    assert isinstance(op.estimator.transformers_["low"], RustyOneHotEncoder)
    assert isinstance(op.estimator.transformers_["high"], RustyStringEncoder)
    assert op.estimator.transformers_["low"]._supported_params
    assert op.estimator.transformers_["low"]._stratum_force_rust
    assert op.estimator.transformers_["high"]._stratum_force_rust
