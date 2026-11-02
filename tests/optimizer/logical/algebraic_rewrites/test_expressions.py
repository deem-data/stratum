"""Exact algebraic simplification of column expressions."""
import operator

import numpy as np
import pandas as pd
import pytest
import skrub

from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._column_expr import BinOpExpr, Col, Const, UfuncExpr
from stratum.optimizer.logical._map_ops import AssignMapOp, SeriesMapOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp
from stratum.optimizer.logical.rewrites._algebraic import AlgebraicRewritesConfig
from stratum.runtime._scheduler import SequentialScheduler

_FRAME = pd.DataFrame({"i": [3, -1, 0, 7], "f": [-0.0, 1.5, np.nan, -2.25],
                       "b": [True, False, True, False]})

_abs = lambda e: e.skb.apply_func(np.abs) if hasattr(e, "skb") else np.abs(e)
# A function stratum cannot see into, so its column's dtype is unknown.
_opaque = lambda e: e.skb.apply_func(abs) if hasattr(e, "skb") else abs(e)

#: Spelling -> the expression it simplifies to.
_SIMPLIFIED = {
    "int + 0": (lambda x: x["i"] + 0, Col("i")),
    "0 + int": (lambda x: 0 + x["i"], Col("i")),
    "int - 0": (lambda x: x["i"] - 0, Col("i")),
    "int * 1": (lambda x: x["i"] * 1, Col("i")),
    "1 * int": (lambda x: 1 * x["i"], Col("i")),
    "int ** 1": (lambda x: x["i"] ** 1, Col("i")),
    "float - 0": (lambda x: x["f"] - 0, Col("f")),
    "float - 0.0": (lambda x: x["f"] - 0.0, Col("f")),
    "float * 1.0": (lambda x: x["f"] * 1.0, Col("f")),
    "float / 1": (lambda x: x["f"] / 1, Col("f")),
    "float ** 1.0": (lambda x: x["f"] ** 1.0, Col("f")),
    "a chain": (lambda x: (x["i"] * 1 - 0) * 2, BinOpExpr(operator.mul, Col("i"), Const(2))),
    "abs(abs(x))": (lambda x: _abs(_abs(x["f"])), UfuncExpr(Col("f"), "abs")),
}

#: Spellings an identity would change: in value, sign of zero or dtype.
_KEPT = {
    "float + 0": lambda x: x["f"] + 0,  # -0.0 + 0 is +0.0
    "bool + 0": lambda x: x["b"] + 0,  # an int
    "int / 1": lambda x: x["i"] / 1,  # a float
    "int * 1.0": lambda x: x["i"] * 1.0,  # a float
    "int ** 1.0": lambda x: x["i"] ** 1.0,  # a float
    "int * True": lambda x: x["i"] * True,
    "int * np.int8(1)": lambda x: x["i"] * np.int8(1),
    "0 - int": lambda x: 0 - x["i"],
    "1 / float": lambda x: 1 / x["f"],
    "a column of unknown dtype": lambda x: x.assign(u=_opaque(x["i"]))["u"] * 1,
}


def _run(build, config=None):
    with skrub.config_context(eager_data_ops=False):
        dag = build(skrub.var("x", _FRAME))
    plan, split = optimize(dag, config or OptConfig(), env={"x": _FRAME})
    return plan, SequentialScheduler(plan, split).evaluate()


def _assert_exactly(expected, result):
    pd.testing.assert_series_equal(expected, result, check_exact=True)
    assert np.signbit(expected.to_numpy(dtype=float)).tolist() == \
        np.signbit(result.to_numpy(dtype=float)).tolist()


@pytest.mark.parametrize("name", _SIMPLIFIED)
def test_an_identity_is_dropped(name):
    build, simplified = _SIMPLIFIED[name]
    plan, result = _run(build)
    [series_map] = [op for op in plan if isinstance(op, SeriesMapOp)]
    assert series_map.expr == simplified
    _assert_exactly(build(_FRAME), result)


@pytest.mark.parametrize("name", _KEPT)
def test_an_identity_that_would_change_the_result_is_kept(name):
    build = _KEPT[name]
    plan, result = _run(build)
    [series_map] = [op for op in plan if isinstance(op, SeriesMapOp)]
    assert isinstance(series_map.expr, BinOpExpr)
    _assert_exactly(build(_FRAME), result)


def test_assigns_and_masks_are_simplified_too():
    plan, _ = _run(lambda x: (lambda kept: kept.assign(z=kept["i"] - 0))(x[x["i"] * 1 > 0]))
    [mask] = [op for op in plan if isinstance(op, MaskSelectionOp)]
    [assign] = [op for op in plan if isinstance(op, AssignMapOp)]
    assert mask.predicate == BinOpExpr(operator.gt, Col("i"), Const(0))
    assert assign.entries["z"] == Col("i")


def test_each_rule_has_its_flag():
    config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(identity_op=False))
    plan, _ = _run(lambda x: x["i"] * 1 + (x["i"] - 0), config)
    [series_map] = [op for op in plan if isinstance(op, SeriesMapOp)]
    assert series_map.expr == BinOpExpr(operator.add, BinOpExpr(operator.mul, Col("i"), Const(1)),
                                         Col("i"))
    plan, _ = _run(lambda x: x["i"] * 1, OptConfig(algebraic_rewrites=False))
    [series_map] = [op for op in plan if isinstance(op, SeriesMapOp)]
    assert series_map.expr == BinOpExpr(operator.mul, Col("i"), Const(1))
