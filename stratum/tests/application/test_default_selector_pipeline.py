import io
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from skrub import StringEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.physical._concat_execs import PandasConcatOp
from stratum.optimizer.physical._map_execs import PandasAssignMapOp
from stratum.optimizer.physical._projection_execs import (
    PandasColumnProjectionOp,
    PandasColumnSelectorOp,
)
from stratum.optimizer.physical._source_execs import PandasReadCSV
from stratum.optimizer.physical._transform_execs import SkrubStringEncoder
from stratum.tests._helpers import csv_file

# TODO: Add tests to match results with vanilla skrub

def make_small_table(n: int = 24) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    amount = rng.uniform(5.0, 50.0, size=n).round(2)
    quantity = rng.randint(1, 6, size=n)
    category = rng.choice(["books", "games", "home"], size=n)
    target = (amount * quantity + (category == "games") * 4.0).round(2)
    return pd.DataFrame({
        "when": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "amount": amount,
        "quantity": quantity,
        "category": category,
        "target": target,
    })


def build_pipeline(file_path):
    """Build a small recorded pipeline using several registered physical ops."""
    data = st.as_data_op(file_path).skb.apply_func(pd.read_csv)

    y = data["target"].skb.mark_as_y()
    X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=KFold(n_splits=2, shuffle=True, random_state=0),
        split_kwargs={})

    date = X["when"].skb.apply_func(pd.to_datetime)
    X = X.assign(
        month=date.dt.month,
        total=X["amount"] * X["quantity"],
    )
    X = X[["amount", "quantity", "total", "category"]]

    numeric = X.skb.select(st.selectors.numeric())
    categorical = X.skb.select(st.selectors.string())
    encoded = categorical.skb.apply(StringEncoder())
    features = numeric.skb.concat([encoded], axis=1)
    return features.skb.apply(Ridge(alpha=1.0), y=y)


def test_default_selector_compiles_registered_pipeline():
    with csv_file(make_small_table()) as path:
        with st.config(implementation_selector="default", rust_backend=True):
            ops, *_ = optimize(
                build_pipeline(path),
                OptConfig(dataframe_ops=True),
            )

    # The source and every migrated frame/transformer family is bound to the
    # default pandas/skrub implementations, even when the legacy Rust flag is on.
    assert any(isinstance(op, PandasReadCSV) for op in ops)
    assert any(isinstance(op, PandasAssignMapOp) for op in ops)
    assert any(isinstance(op, PandasColumnProjectionOp) for op in ops)
    assert any(isinstance(op, PandasColumnSelectorOp) for op in ops)
    assert any(isinstance(op, PandasConcatOp) for op in ops)
    assert any(isinstance(op, SkrubStringEncoder) for op in ops)


def test_default_selector_pipeline_scores_end_to_end():
    with csv_file(make_small_table()) as path:
        predictions = build_pipeline(path)
        
        # Capture explain output using redirect_stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            # Enable scheduler mode to trigger optimizer and get explain output
            with st.config(implementation_selector="default", scheduler=True, explain=("physical_impl")):
                search = predictions.skb.make_grid_search(
                    n_jobs=1,
                    fitted=True,
                    refit=False,
                    scoring="r2",
                )

    # Print the captured explain output
    output_str = captured_output.getvalue()
    if output_str:
        print(output_str)

    assert search.results_ is not None
    # SequentialScheduler returns a Polars DataFrame with columns ['id', 'scores']
    # instead of pandas DataFrame with 'mean_test_score'
    assert "scores" in search.results_.columns or "mean_test_score" in search.results_.columns
