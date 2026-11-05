"""Stable lineage keys without source readers, reuse planning or execution."""
import decimal
import json
import operator
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from stratum.optimizer._lineage import (
    KEY_BYTES, MAX_ENCODED_VALUES, LineageEncoder, Uncacheable,
    compute_lineage_keys, lineage_key,
)
from stratum.optimizer.logical._base import IRNode, OperandRef
from stratum.optimizer.logical._ops import CallOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._representation import FrameLib

REPO = Path(__file__).resolve().parents[2]


class ConfiguredOp(PhysicalOp):
    cacheable = True
    fields = ["config"]

    def __init__(self, config, inputs=None, lib=None):
        super().__init__(inputs=inputs)
        self.config = config
        self._frame_lib = lib


class OtherImpl(ConfiguredOp):
    pass


def digest(value):
    encoder = LineageEncoder()
    encoder.encode(value)
    return encoder.hexdigest()


def pipeline(offset=1):
    """Two branches sharing a source, joined at the sink."""
    source = ConfiguredOp({"source": "example"})
    edited = ConfiguredOp({"offset": offset}, [source])
    other = ConfiguredOp({"columns": {"a", "b", "c"}}, [source])
    sink = ConfiguredOp({"operands": [OperandRef(0), OperandRef(1)]}, [edited, other])
    return [source, edited, other, sink]


def test_distinct_values_encode_distinctly():
    values = [1, "1", 1.0, True, False, None, b"1", np.int64(1), (1,), [1], {1},
              {1: None}, float("nan"), float("inf"), float("-inf"), 0.0, -0.0,
              slice(1), slice(0, 1, 2), OperandRef(0), OperandRef(1), "", (), [],
              decimal.Decimal("1"), pd.Timestamp("2020-01-01"),
              pd.Timestamp("2020-01-01", tz="UTC"), pd.Timedelta("1 day"),
              pd.NaT, pd.NA, np.dtype("int64"), pd.StringDtype(), str, "str"]
    keys = [digest(value) for value in values]
    assert len(set(keys)) == len(values)
    assert all(len(bytes.fromhex(key)) == KEY_BYTES for key in keys)


def test_container_order_and_boundaries():
    assert digest({"a": 1, "b": 2}) != digest({"b": 2, "a": 1})
    assert digest({"a", "b", "c"}) == digest(set(["c", "b", "a"]))
    assert digest(frozenset(["c", "a"])) == digest(frozenset(["a", "c"]))
    assert digest(("a", ("b",))) != digest(("a", "b"))
    assert digest(["ab", "c"]) != digest(["a", "bc"])
    assert digest([OperandRef(0), OperandRef(1)]) != digest([OperandRef(1), OperandRef(0)])


@pytest.mark.parametrize("value", [
    pd.DataFrame({"a": [1]}), pd.Series([1]), np.arange(3), object(),
    pd.to_datetime, lambda x: x, operator.add, StandardScaler(),
    np.random.RandomState(0), np.random.default_rng(0), {"f": len},
])
def test_data_and_unknown_objects_are_rejected(value):
    with pytest.raises(Uncacheable, match="no lineage encoding"):
        digest(value)


@pytest.mark.parametrize("container", [list, set])
def test_configuration_size_is_bounded(container):
    with pytest.raises(Uncacheable, match=f"exceeds {MAX_ENCODED_VALUES} values"):
        digest(container(range(MAX_ENCODED_VALUES + 1)))


def test_objects_can_supply_plain_lineage_values():
    class Expression:
        def __init__(self, value):
            self.value = value

        def lineage(self):
            return ("expression", self.value)

    assert digest(Expression(1)) == digest(Expression(1))
    assert digest(Expression(1)) != digest(Expression(2))
    assert digest(Expression(1)) != digest(("expression", 1))
    with pytest.raises(Uncacheable):
        digest(Expression(object()))


def test_dtypes_include_configuration_missing_from_their_strings():
    # These categorical dtypes all stringify to "category"; string dtypes also
    # omit storage. Neither may collapse to the same lineage key.
    values = [pd.CategoricalDtype(["a", "b"]), pd.CategoricalDtype(["a"]),
              pd.CategoricalDtype(["a"], ordered=True), pd.CategoricalDtype([1, 2]),
              pd.CategoricalDtype(), pd.StringDtype("python"),
              pd.StringDtype("pyarrow"), pd.DatetimeTZDtype("ns", "UTC"),
              pd.DatetimeTZDtype("ns", "Europe/Berlin"), pd.PeriodDtype("D"),
              pd.PeriodDtype("M"), pd.Int64Dtype()]
    keys = [digest(value) for value in values]
    assert len(set(keys)) == len(values)
    assert digest(pd.CategoricalDtype(["a", "b"])) == keys[0]


def test_arrow_dtype_metadata_is_encoded():
    import pyarrow as pa

    assert digest(pd.ArrowDtype(pa.int64())) == digest(pd.ArrowDtype(pa.int64()))
    assert digest(pd.ArrowDtype(pa.int64())) != digest(pd.ArrowDtype(pa.float64()))


def test_operators_must_opt_in():
    for op in (IRNode(), PhysicalOp(), CallOp(func=len)):
        with pytest.raises(Uncacheable, match="not a cacheable operator"):
            lineage_key(op, [])
    assert ConfiguredOp({"offset": 1}).lineage_config() == {"config": {"offset": 1}}


def test_keys_include_implementation_library_configuration_and_ordered_inputs():
    op = ConfiguredOp({"offset": 1}, lib=FrameLib.PANDAS)
    base = lineage_key(op, ["left", "right"])
    assert base == lineage_key(ConfiguredOp({"offset": 1}, lib=FrameLib.PANDAS),
                               ["left", "right"])
    variants = [
        lineage_key(OtherImpl({"offset": 1}, lib=FrameLib.PANDAS), ["left", "right"]),
        lineage_key(ConfiguredOp({"offset": 1}, lib=FrameLib.POLARS), ["left", "right"]),
        lineage_key(ConfiguredOp({"offset": 1}), ["left", "right"]),
        lineage_key(ConfiguredOp({"offset": 2}, lib=FrameLib.PANDAS), ["left", "right"]),
        lineage_key(op, ["right", "left"]),
        lineage_key(op, ["left"]),
        lineage_key(op, ["left", "changed"]),
    ]
    assert len({base, *variants}) == len(variants) + 1


def test_display_and_execution_metadata_do_not_change_keys():
    op = ConfiguredOp({"offset": 1})
    key = lineage_key(op, [])
    op.name = "display label"
    op.remove_after = [IRNode()]
    op.dead_in_fit = True
    assert lineage_key(op, []) == key


def test_a_change_rekeys_only_the_op_and_its_descendants():
    original = list(compute_lineage_keys(pipeline()).values())
    edited = list(compute_lineage_keys(pipeline(offset=2)).values())
    assert all(key is not None for key in original)
    assert [a != b for a, b in zip(original, edited)] == [False, True, False, True]
    assert list(compute_lineage_keys(pipeline()).values()) == original


@pytest.mark.parametrize("opaque", [IRNode, lambda: ConfiguredOp(object())])
def test_uncacheable_inputs_block_only_their_descendants(opaque):
    source = opaque()
    blocked = ConfiguredOp("blocked", [source])
    independent = ConfiguredOp("independent")
    sink = ConfiguredOp("sink", [blocked, independent])
    reasons = {}
    keys = compute_lineage_keys([source, blocked, independent, sink], reasons=reasons)
    assert keys[source] is None and keys[blocked] is None and keys[sink] is None
    assert keys[independent] is not None
    assert "an input has no key" == reasons[blocked] == reasons[sink]
    assert reasons[source]
    assert independent not in reasons


def test_region_limits_keys_and_missing_inputs_block_descendants():
    source, edited, other, sink = pipeline()
    plan = [source, edited, other, sink]
    reasons = {}
    keys = compute_lineage_keys(iter(plan), region={source, edited, sink}, reasons=reasons)
    assert keys[source] is not None and keys[edited] is not None
    assert keys[other] is None and keys[sink] is None
    assert reasons == {sink: "an input has no key"}
    assert all(key is None for key in compute_lineage_keys(plan, region=set()).values())
    assert all(key is None for key in compute_lineage_keys(plan[1:]).values())


def test_keys_are_stable_across_processes():
    script = textwrap.dedent("""
        import json
        from tests.optimizer.test_lineage import pipeline
        from stratum.optimizer._lineage import compute_lineage_keys
        print(json.dumps(list(compute_lineage_keys(pipeline()).values())))
    """)
    expected = list(compute_lineage_keys(pipeline()).values())
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run([sys.executable, "-c", script], env=env, cwd=REPO,
                                capture_output=True, text=True, check=True, timeout=60)
        assert json.loads(result.stdout.strip().splitlines()[-1]) == expected
