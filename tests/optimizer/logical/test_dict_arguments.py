"""Dict subclasses in operator arguments keep their type through the optimizer."""
from collections import Counter, OrderedDict, defaultdict

import pandas as pd
import pytest
import skrub

import stratum
from stratum._api import evaluate
from stratum.optimizer.logical._base import (
    OperandRef, _resolve_operand, clone_value, config_key, rebuild_dict,
    remap_operand_refs)


def _fallback():
    return "?"


@pytest.mark.parametrize("value", [
    defaultdict(_fallback, {"a": OperandRef(0)}),
    OrderedDict([("b", OperandRef(0)), ("a", 1)]),
    Counter({"a": 2}),
])
def test_operand_helpers_keep_the_dict_type(value):
    for rebuilt in (_resolve_operand(value, ["x"]), clone_value(value),
                    remap_operand_refs(value, {0: 0}),
                    rebuild_dict(value, value.items())):
        assert type(rebuilt) is type(value)
        assert list(rebuilt) == list(value)
    if isinstance(value, defaultdict):
        assert _resolve_operand(value, ["x"])["missing"] == "?"


def test_a_dict_subclass_has_its_own_cse_key():
    items = {"a": "1"}
    assert config_key(defaultdict(_fallback, items)) != config_key(dict(items))


def test_map_with_a_defaultdict_keeps_its_default():
    frame = pd.DataFrame({"k": ["MN", "BX"]})
    with_default = defaultdict(_fallback, {"MN": "1"})
    plain = {"MN": "1"}
    with skrub.config_context(eager_data_ops=False):
        source = skrub.var("frame", frame)
        plan = source.assign(a=source["k"].map(with_default), b=source["k"].map(plain))
    with stratum.config(scheduler=True):
        result = evaluate(plan)
    pd.testing.assert_frame_equal(
        result, frame.assign(a=frame["k"].map(with_default), b=frame["k"].map(plain)))
