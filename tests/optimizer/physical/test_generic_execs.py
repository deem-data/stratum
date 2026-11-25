"""Generic logical operators become executable physical nodes at lowering."""

import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler
from skrub import selectors

from stratum.optimizer.logical._base import OperandRef, OutputType
from stratum.optimizer.logical._ops import CallOp, MethodCallOp, TransformerOp, ValueOp
from stratum.optimizer.physical._generic_execs import CallExec, MethodCallExec, ValueExec
from stratum.optimizer.physical._lowering import lower_to_physical
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._transform_execs import PassthroughTransformer


def test_generic_ops_lower_in_place_and_execute():
    cases = [
        (ValueOp(42), ValueExec, [], 42),
        (MethodCallOp("upper", args=(), kwargs={}), MethodCallExec, ["hello"], "HELLO"),
        (CallOp(func=lambda x: x + 1, args=(OperandRef(0),), kwargs={}), CallExec, [2], 3),
    ]
    for op, expected_type, inputs, expected in cases:
        op.output_type = OutputType.FRAME
        original_id = id(op)
        lowered = lower_to_physical(op, PlanContext.from_flags())
        assert id(lowered) == original_id
        assert isinstance(lowered, expected_type)
        assert isinstance(lowered, PhysicalOp)
        assert " [df]" not in str(lowered)
        assert lowered.process(inputs) == expected


def test_generic_transformer_lowers_in_place():
    op = TransformerOp(estimator=MinMaxScaler(), cols=selectors.all(), no_wrap=True)
    with pytest.raises(NotImplementedError, match="must be lowered"):
        op.process([pd.DataFrame({"x": [1.0, 2.0]})])
    original_id = id(op)
    lowered = lower_to_physical(op, PlanContext.from_flags())
    assert id(lowered) == original_id
    assert isinstance(lowered, PassthroughTransformer)
    assert isinstance(lowered, PhysicalOp)
    lowered.output_type = OutputType.FRAME
    assert " [df]" not in str(lowered)
    result = lowered.process([pd.DataFrame({"x": [1.0, 2.0]})])
    assert result.output.shape == (2, 1)


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("no_wrap", [False, True])
def test_column_transformer_selects_named_columns(tmp_path, selector, no_wrap):
    import numpy as np
    import skrub
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import stratum as st
    from stratum.frontend._api import evaluate

    path = tmp_path / "columns.csv"
    pd.DataFrame({"a": np.arange(30, dtype=float),
                  "b": np.arange(30, dtype=float) % 3,
                  "y": np.arange(30, dtype=float) * 2}).to_csv(path, index=False)
    with skrub.config_context(eager_data_ops=False):
        data = st.as_data_op(str(path)).skb.apply_func(pd.read_csv)
        X = data.drop(columns="y").skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        transformer = ColumnTransformer([
            ("scaled", StandardScaler(), ["a", "b"]),
        ]).set_output(transform="pandas")
        features = X.skb.apply(transformer, no_wrap=no_wrap)
        pred = features.skb.apply(Ridge(), y=y)
    with st.config(scheduler=True, implementation_selector=selector, reuse_cache=False):
        actual = evaluate(pred, seed=42, test_size=0.2)
    with st.config(scheduler=False):
        splits = pred.skb.train_test_split(random_state=42, test_size=0.2)
        learner = pred.skb.make_learner()
        learner.fit(splits["train"])
        expected = learner.predict(splits["test"])
    np.testing.assert_allclose(actual, expected)
