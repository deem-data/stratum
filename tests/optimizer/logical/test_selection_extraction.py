"""Frame extraction precedes mask folding and scalar numeric rewrites."""
import operator

import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum as st
from stratum.frontend._conversion import convert_to_ops
from stratum.optimizer._op_utils import topological_iterator, validate_dag
from stratum.optimizer._optimize import OptConfig, extract_frame_operators, optimize
from stratum.optimizer.logical._column_expr import (
    BinOpExpr, Col, Const, FillnaExpr, UfuncExpr)
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._ops import BinOp, GetItemOp, UnaryOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp
from stratum.optimizer.logical.rewrites._literal_inlining import literal_inlining
from stratum.optimizer.logical.rewrites._selection_extraction import extract_mask_selections
from stratum.optimizer.physical._impl_selection import get_implementation_selector
from stratum.optimizer.physical._selection_execs import PandasIndexSelectionOp
from stratum.runtime._scheduler import SequentialScheduler


FRAME = pd.DataFrame({"x": [0., 2., np.nan, 4.], "y": [1, 2, 3, 4]})


def build_query(build):
    with skrub.config_context(eager_data_ops=False):
        return build(skrub.var("frame", FRAME))


def execute(query, selector, **options):
    config = OptConfig(selector=get_implementation_selector(selector), **options)
    ops, split = optimize(query, config, env={"frame": FRAME})
    validate_dag(ops[-1])
    result = SequentialScheduler(ops, split).evaluate()
    if isinstance(result, (pl.Series, pl.DataFrame)):
        result = result.to_pandas()
    return ops, result


def test_frame_extraction_leaves_mask_folding_to_its_own_pass():
    query = build_query(lambda df: df[(df["x"].fillna(0.) + 1) > 2])
    root = extract_frame_operators(literal_inlining(convert_to_ops(query, {"frame": FRAME})))
    validate_dag(root)
    ops = list(topological_iterator(root))
    assert isinstance(root, GetItemOp)
    assert not any(isinstance(op, MaskSelectionOp) for op in ops)
    assert not any(isinstance(op, (BinOp, UnaryOp)) for op in ops)
    assert len([op for op in ops if isinstance(op, ExpressionMapOp)]) == 3

    root = extract_mask_selections(root)
    validate_dag(root)
    assert isinstance(root, MaskSelectionOp)
    assert root.predicate == BinOpExpr(
        operator.gt, BinOpExpr(operator.add, FillnaExpr(Col("x"), 0.), Const(1)), Const(2))
    assert len(list(topological_iterator(root))) == 2


@pytest.mark.parametrize("make_maps", [False, True])
@pytest.mark.parametrize("fold_masks", [False, True])
def test_map_and_mask_flags_remain_independent(make_maps, fold_masks):
    query = build_query(lambda df: df[(df["x"] + 1).gt(2)])
    with st.config(make_map_op=make_maps, make_selection_op=fold_masks):
        ops, result = execute(query, "greedy")
    if fold_masks:
        assert isinstance(ops[-1], MaskSelectionOp)
    else:
        assert isinstance(ops[-1], GetItemOp)
        assert ops[-1].is_filter
    expected = FRAME[(FRAME["x"] + 1).gt(2)]
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_nested_masks_use_the_rewritten_source(selector):
    def build(df):
        first = df[df["x"].notna()]
        return first[first["y"].isin([2, 4])]
    ops, result = execute(build_query(build), selector)
    masks = [op for op in ops if isinstance(op, MaskSelectionOp)]
    assert len(masks) == 2
    assert masks[-1].inputs[0] is masks[0]
    pd.testing.assert_frame_equal(result.reset_index(drop=True), build(FRAME).reset_index(drop=True))


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_numpy_maps_fold_into_mask_predicates(selector):
    query = build_query(lambda df: df[df["x"].skb.apply_func(np.log).fillna(0.) > 1])
    ops, result = execute(query, selector, pandas_query=True)
    assert isinstance(ops[-1], MaskSelectionOp)
    assert ops[-1].predicate == BinOpExpr(
        operator.gt, FillnaExpr(UfuncExpr(Col("x"), "log"), 0.), Const(1))
    assert not any(isinstance(op, ExpressionMapOp) for op in ops)
    if selector == "default":
        assert isinstance(ops[-1], PandasIndexSelectionOp)
    expected = FRAME[np.log(FRAME["x"]).fillna(0.) > 1]
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
