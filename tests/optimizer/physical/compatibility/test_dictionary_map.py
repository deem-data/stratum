"""``Series.map`` with a plain dict runs natively under greedy and matches pandas."""
from collections import defaultdict

import pandas as pd
import pytest
import skrub

import stratum
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._column_methods import ColumnMethodOp

STRINGS = pd.DataFrame({"k": ["MN", "BX", "unknown", None, "MN"]})
INTS = pd.DataFrame({"k": [1, 2, 3, 1, 5]})


@pytest.mark.parametrize("frame,mapping", [
    (STRINGS, {"MN": "1", "BX": "2"}),
    (STRINGS, {"MN": 1, "BX": 2}),
    (STRINGS, {"MN": 1, "BX": 2.5}),
    (STRINGS, {"MN": True, "BX": False}),
    (STRINGS, {"MN": "1", "BX": None}),
    (STRINGS, {}),
    (INTS, {1: "a", 2: "b"}),
    (INTS, {1: 10, 5: 50}),
])
@pytest.mark.parametrize("call", ["positional", "func", "ignore"])
@pytest.mark.parametrize("path", ["standalone", "assign", "selection"])
def test_dictionary_map_matches_pandas(frame, mapping, call, path):
    kwargs = {"func": {"func": mapping}, "positional": {},
              "ignore": {"na_action": "ignore"}}[call]
    args = () if call == "func" else (mapping,)
    expected = frame["k"].map(*args, **kwargs)
    with skrub.config_context(eager_data_ops=False):
        source = skrub.var("frame", frame)
        mapped = source["k"].map(*args, **kwargs)
        plan = (mapped if path == "standalone" else
                source.assign(mapped=mapped) if path == "assign" else source[mapped.notna()])
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        ops, *_ = optimize(plan, env=get_data(plan))
        assert any(isinstance(op, ColumnMethodOp) and op.method == "map" for op in ops) \
            or path != "standalone"
        result = evaluate(plan).to_pandas()
    if path == "selection":
        pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                      frame[expected.notna()].reset_index(drop=True),
                                      check_dtype=False)
        return
    if path == "assign":
        result = result["mapped"].rename("k")
    assert result.isna().tolist() == expected.isna().tolist()
    pd.testing.assert_series_equal(result.dropna().reset_index(drop=True),
                                   expected.dropna().reset_index(drop=True),
                                   check_dtype=False)


@pytest.mark.parametrize("mapping", [
    lambda v: v,                              # arbitrary code
    pd.Series({"MN": "1"}),                   # index alignment
    defaultdict(lambda: "?", {"MN": "1"}),    # __missing__ makes it a function
    {"MN": "1", 2: "2"},                      # mixed key types
    {"MN": "1", "BX": 2},                     # mixed value types
    {float("nan"): "1"},                      # missing key
])
def test_other_mappings_are_not_folded(mapping):
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("frame", STRINGS)["k"].map(mapping)
    with stratum.config(scheduler=True):
        ops, *_ = optimize(plan, env=get_data(plan))
    assert not any(isinstance(op, ColumnMethodOp) for op in ops)


@pytest.mark.parametrize("mapping", [{"MN": "1", "BX": "2"}, lambda v: v])
def test_default_selector_is_unchanged(mapping):
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("frame", STRINGS)["k"].map(mapping)
    with stratum.config(scheduler=True):
        result = evaluate(plan)
    pd.testing.assert_series_equal(result, STRINGS["k"].map(mapping))
