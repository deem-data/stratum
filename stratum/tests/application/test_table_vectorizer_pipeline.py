import io
import sys
import numpy as np
import pandas as pd
import pytest
from contextlib import redirect_stdout
from skrub import StringEncoder, TableVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import stratum as st
from stratum.adapters.one_hot_encoder import (
    RustyOneHotEncoder,
    supports_rust_one_hot_encoder,
)
from stratum.adapters.string_encoder import (
    RustyStringEncoder,
    supports_rust_string_encoder,
)
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._map_execs import (
    PandasAssignMapOp,
    PolarsAssignMapOp,
)
from stratum.optimizer.physical._projection_execs import (
    PandasColumnProjectionOp,
    PolarsColumnProjectionOp,
)
from stratum.optimizer.physical._source_execs import (
    PandasReadCSV,
    PolarsReadCSV,
)
from stratum.optimizer.physical._transform_execs import (
    SkrubTableVectorizer,
    StratumTableVectorizer,
    TableVectorizerOp,
)
from stratum.tests._helpers import csv_file


def capture_std_out(capfd):
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    return (captured.out or "") + (captured.err or "")


def make_orders(n=36):
    """Create a small, mixed-type order table for the recorded pipeline."""
    rng = np.random.RandomState(17)
    quantity = rng.randint(1, 16, size=n)
    unit_price = rng.uniform(8, 80, size=n).round(2)
    category = rng.choice(["books", "games", "home", "garden"], size=n)
    country = rng.choice(["DE", "FR", "US"], size=n)
    status = rng.choice(["completed", "cancelled"], size=n, p=[0.85, 0.15])
    description = [
        f"customer note {i} about {category[i]} order {i % 5}"
        for i in range(n)
    ]
    target = (
        quantity * unit_price
        + (category == "games") * 7
        + (country == "DE") * 2
        + rng.normal(0, 0.2, size=n)
    ).round(3)
    return pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=n).astype(str),
            "quantity": quantity,
            "unit_price": unit_price,
            "category": category,
            "country": country,
            "description": description,
            "status": status,
            "target": target,
        }
    )


def build_pipeline(file_path, table_vectorizer):
    """Read, filter, engineer, vectorize, and fit a regression model."""
    data = st.as_data_op(file_path).skb.apply_func(pd.read_csv)

    active = data[(data["status"] == "completed") & (data["quantity"] > 0)]
    y = active["target"].skb.mark_as_y()
    X = active.drop(columns=["target"]).skb.mark_as_X(
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        split_kwargs={},
    )

    order_date = X["order_date"].skb.apply_func(pd.to_datetime)
    featured = X.assign(
        revenue=X["quantity"] * X["unit_price"],
        order_month=order_date.dt.month,
        is_bulk=X["quantity"] >= 8,
    )
    X = featured[
        [
            "quantity",
            "unit_price",
            "revenue",
            "order_month",
            "is_bulk",
            "category",
            "country",
            "description",
        ]
    ]

    X_vectorized = X.skb.apply(table_vectorizer)
    return X_vectorized.skb.apply(Ridge(alpha=1.0), y=y)


def _hashing_table_vectorizer():
    return TableVectorizer(
        cardinality_threshold=5,
        high_cardinality=StringEncoder(
            vectorizer="hashing",
            analyzer="char",
            ngram_range=(2, 3),
            n_components=2,
            random_state=0,
        ),
    )


def _unsupported_string_table_vectorizer():
    return TableVectorizer(
        cardinality_threshold=5,
        high_cardinality=StringEncoder(
            vectorizer="hashing",
            analyzer="word",  # word not supported by Rust
            ngram_range=(1, 2),
            n_components=2,
            random_state=0,
        ),
    )


@pytest.mark.parametrize(
    "table_vectorizer",
    [
        pytest.param(TableVectorizer(), id="default-table-vectorizer"),
        pytest.param(_hashing_table_vectorizer(), id="hashing-char-high-cardinality"),
        pytest.param(_unsupported_string_table_vectorizer(), id="unsupported-string-encoder",),
    ],
)
def test_table_vectorizer_pipeline_scores_end_to_end_for_each_selector(
    capfd,
    table_vectorizer,
):
    """Run a realistic DataOps pipeline through both physical selector policies."""
    with csv_file(make_orders()) as path:
        for selector in ("default", "greedy"):
            pipeline = build_pipeline(path, table_vectorizer)
            with st.config(
                implementation_selector=selector,
                scheduler=True,
                rust_backend=True,
                allow_patch=True,
                explain=("physical_impl"),
                debug_timing=True,
            ):
                ops, *_ = optimize(pipeline, env=pipeline.skb.get_data())

                table_ops = [op for op in ops if isinstance(op, TableVectorizerOp)]
                assert len(table_ops) == 1
                table_op = table_ops[0]
                supports_low = supports_rust_one_hot_encoder(
                    table_vectorizer.low_cardinality
                )[0]
                supports_high = supports_rust_string_encoder(
                    table_vectorizer.high_cardinality
                )[0]

                if selector == "default":
                    assert any(isinstance(op, PandasReadCSV) for op in ops)
                    assert any(isinstance(op, PandasAssignMapOp) for op in ops)
                    assert any(isinstance(op, PandasColumnProjectionOp) for op in ops)
                    assert isinstance(table_op, SkrubTableVectorizer)
                    uses_rust = False
                else:
                    assert any(isinstance(op, PolarsReadCSV) for op in ops)
                    assert any(isinstance(op, PolarsAssignMapOp) for op in ops)
                    assert any(isinstance(op, PolarsColumnProjectionOp) for op in ops)
                    expected_table_op = (
                        StratumTableVectorizer if (supports_low or supports_high)
                        else SkrubTableVectorizer
                    )
                    assert isinstance(table_op, expected_table_op)
                    uses_rust = expected_table_op is StratumTableVectorizer
                    if expected_table_op is StratumTableVectorizer:
                        if supports_low:
                            assert isinstance(
                                table_op.estimator.low_cardinality,
                                RustyOneHotEncoder,
                            )
                        else:
                            assert not isinstance(
                                table_op.estimator.low_cardinality,
                                RustyOneHotEncoder,
                            )

                        if supports_high:
                            assert isinstance(
                                table_op.estimator.high_cardinality,
                                RustyStringEncoder,
                            )
                        else:
                            assert not isinstance(
                                table_op.estimator.high_cardinality,
                                RustyStringEncoder,
                            )

                captured_output = io.StringIO()
                with redirect_stdout(captured_output):
                    search = pipeline.skb.make_grid_search(
                        fitted=True,
                        refit=False,
                        scoring="r2",
                    )

            output_str = captured_output.getvalue()
            if output_str:
                with capfd.disabled():
                    print(output_str)

            assert search.results_ is not None
            assert len(search.results_) > 0
            assert "scores" in search.results_.columns or (
                "mean_test_score" in search.results_.columns
            )

            combined_output = capture_std_out(capfd)
            if uses_rust:
                assert "[rust]" in combined_output
            else:
                assert "[rust]" not in combined_output

