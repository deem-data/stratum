"""Generic logical operators become executable physical nodes at lowering."""

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
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
        assert lowered.process("fit_transform", inputs) == expected


def test_generic_transformer_lowers_in_place():
    op = TransformerOp(estimator=StandardScaler(), cols=selectors.all(), how="no_wrap")
    with pytest.raises(NotImplementedError, match="must be lowered"):
        op.process("fit_transform", [pd.DataFrame({"x": [1.0, 2.0]})])
    original_id = id(op)
    lowered = lower_to_physical(op, PlanContext.from_flags())
    assert id(lowered) == original_id
    assert isinstance(lowered, PassthroughTransformer)
    assert isinstance(lowered, PhysicalOp)
    lowered.output_type = OutputType.FRAME
    assert " [df]" not in str(lowered)
    result = lowered.process("fit_transform", [pd.DataFrame({"x": [1.0, 2.0]})])
    assert result.shape == (2, 1)
