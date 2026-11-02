"""Lineage keys: persistent, exact names for what an operator computes."""
import decimal
import operator
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._lineage import (
    MAX_ENCODED_VALUES, LineageEncoder, Uncacheable, compute_lineage_keys)
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._column_expr import BinOpExpr, Col, Const
from stratum.optimizer.logical._source_ops import (
    FileReadOp, InMemoryFrameOp, file_fingerprint)

REPO = Path(__file__).resolve().parents[2]


def digest(value) -> str:
    enc = LineageEncoder()
    enc.encode(value)
    return enc.hexdigest()


def plan_keys(dag, reasons=None) -> list[tuple[str, str | None]]:
    """``(op, key)`` for each op of the optimized plan, in plan order."""
    plan, _ = optimize(dag, env=get_data(dag))
    keys = compute_lineage_keys(plan, reasons=reasons)
    return [(str(op), keys[op]) for op in plan]


def read_csv(path):
    return st.as_data_op(str(path)).skb.apply_func(pd.read_csv)


def csv_pipeline(path, offset=1):
    data = read_csv(path)
    return data.assign(z=data["x"] + offset)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_csv(path, index=False)
    return path


# --- the encoder -------------------------------------------------------------

def test_distinct_values_encode_distinctly():
    values = [1, "1", 1.0, True, None, b"1", np.int64(1), (1,), [1], {1}, {1: None},
              float("nan"), 0.0, -0.0, slice(1), "", (), [], decimal.Decimal("1"),
              pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01", tz="UTC"),
              np.dtype("int64"), pd.StringDtype(), str, "str"]
    digests = [digest(v) for v in values]
    assert len(set(digests)) == len(values)


def test_dtypes_are_named_by_all_they_configure():
    # Each pair stringifies alike: "category", "string".
    values = [pd.CategoricalDtype(["a", "b"]), pd.CategoricalDtype(["a"]),
              pd.CategoricalDtype(["a"], ordered=True), pd.CategoricalDtype([1, 2]),
              pd.CategoricalDtype(), pd.StringDtype("python"), pd.StringDtype("pyarrow"),
              pd.DatetimeTZDtype("ns", "UTC"), pd.DatetimeTZDtype("ns", "Europe/Berlin"),
              pd.PeriodDtype("D"), pd.Int64Dtype()]
    digests = [digest(v) for v in values]
    assert len(set(digests)) == len(values)
    assert digest(pd.CategoricalDtype(["a", "b"])) == digests[0]


def test_calendar_offsets_are_named_by_all_they_configure():
    # normalize is not in the repr; months=12 is another offset than years=1.
    values = [pd.DateOffset(years=1), pd.DateOffset(months=12), pd.DateOffset(years=2),
              pd.DateOffset(2, years=1), pd.DateOffset(years=1, normalize=True),
              pd.DateOffset(years=1, months=2), pd.offsets.MonthEnd(2), pd.offsets.Day(3),
              pd.offsets.Week(weekday=2), pd.offsets.Week(weekday=3), pd.DateOffset()]
    digests = [digest(v) for v in values]
    assert len(set(digests)) == len(values)
    # The same offset spelled with its keywords in another order.
    assert digest(pd.DateOffset(years=1, months=2)) == digest(pd.DateOffset(months=2, years=1))


def test_offsets_of_unknown_code_are_never_encoded():
    class Shifted(pd.DateOffset):
        pass

    # A user subclass, and a business calendar object inside an offset.
    for value in (Shifted(years=1), pd.offsets.CustomBusinessDay(holidays=["2020-01-01"])):
        with pytest.raises(Uncacheable):
            digest(value)


def test_containers():
    assert digest({"a": 1, "b": 2}) != digest({"b": 2, "a": 1})  # order is semantic
    assert digest({3, 1, 2}) == digest({2, 3, 1})
    assert digest(("a", ("b",))) != digest(("a", "b"))


def test_data_is_never_encoded():
    for value in (pd.DataFrame({"a": [1]}), np.arange(3), pd.Series([1])):
        with pytest.raises(Uncacheable):
            digest(value)
    with pytest.raises(Uncacheable):
        digest(list(range(MAX_ENCODED_VALUES + 1)))


def test_unknown_objects_are_never_encoded():
    """Only plain values: no callable, estimator or arbitrary object is guessed at."""
    class Custom:
        pass

    for value in (pd.to_datetime, lambda x: x, operator.add, StandardScaler(),
                  Custom(), np.random.RandomState(0), {"f": len}):
        with pytest.raises(Uncacheable):
            digest(value)


def test_expressions_name_themselves():
    def expr(op, value):
        return BinOpExpr(op, Col("a"), Const(value))

    assert digest(expr(operator.add, 1)) == digest(expr(operator.add, 1))
    assert digest(expr(operator.add, 1)) != digest(expr(operator.sub, 1))
    assert digest(expr(operator.add, 1)) != digest(expr(operator.add, 1.0))
    with pytest.raises(Uncacheable, match="not a known operator"):
        digest(BinOpExpr(np.add, Col("a"), Const(1)))


# --- file sources --------------------------------------------------------------

def test_file_fingerprint_reads_metadata_only(csv_path, tmp_path):
    before = file_fingerprint(csv_path)
    assert before == file_fingerprint(str(csv_path))
    os.utime(csv_path, ns=(1, 1))
    assert file_fingerprint(csv_path) != before
    for bad in (tmp_path / "missing.csv", "s3://bucket/x.csv", str(tmp_path / "*.csv")):
        with pytest.raises(Uncacheable):
            file_fingerprint(bad)
    part = tmp_path / "dir"
    part.mkdir()
    (part / "a.parquet").write_bytes(b"1")
    listing = file_fingerprint(part)
    (part / "b.parquet").write_bytes(b"2")
    assert file_fingerprint(part) != listing


class FakeGCSFileSystem:
    """Stands in for gcsfs: a bucket of ``name -> (size, generation)``."""
    objects = {}
    calls = []

    def __init__(self, **storage_options):
        self.storage_options = storage_options

    def find(self, path, detail=False):
        FakeGCSFileSystem.calls.append((path, self.storage_options))
        prefix = path.split("://", 1)[1].rstrip("/")
        return {name: {"name": name, "size": size, "generation": generation}
                for name, (size, generation) in self.objects.items()
                if name == prefix or name.startswith(prefix + "/")}


@pytest.fixture
def fake_gcs(monkeypatch):
    module = type(sys)("gcsfs")
    module.GCSFileSystem = FakeGCSFileSystem
    monkeypatch.setitem(sys.modules, "gcsfs", module)
    FakeGCSFileSystem.objects = {"lake/t/year=1/part-0.parquet": (10, "1"),
                                 "lake/t/year=2/part-0.parquet": (20, "2")}
    FakeGCSFileSystem.calls = []
    return FakeGCSFileSystem


def test_gcs_fingerprint_lists_objects(fake_gcs):
    before = file_fingerprint("gs://lake/t", {"token": "key.json"})
    assert before == ("gs://lake/t", (("year=1/part-0.parquet", 10, "1"),
                                      ("year=2/part-0.parquet", 20, "2")))
    # credentials reach the listing; no listing is served from a cache
    assert fake_gcs.calls == [("gs://lake/t", {"token": "key.json",
                                               "use_listings_cache": False})]
    fake_gcs.objects["lake/t/year=2/part-0.parquet"] = (20, "3")  # overwritten
    assert file_fingerprint("gs://lake/t") != before
    assert file_fingerprint("gs://lake/t/year=1/part-0.parquet")[1] == (
        ("", 10, "1"),)
    with pytest.raises(Uncacheable, match="no objects"):
        file_fingerprint("gs://lake/missing")


def test_gcs_fingerprint_needs_gcsfs(monkeypatch):
    monkeypatch.setitem(sys.modules, "gcsfs", None)
    with pytest.raises(Uncacheable, match="gcsfs is not installed"):
        file_fingerprint("gs://lake/t")


# --- keys over plans ------------------------------------------------------------

def test_every_op_of_a_file_pipeline_is_keyed(csv_path):
    keys = plan_keys(csv_pipeline(csv_path))
    assert keys and all(k is not None for _, k in keys)
    assert len({k for _, k in keys}) == len(keys)
    assert plan_keys(csv_pipeline(csv_path)) == keys


def test_a_change_rekeys_the_op_and_what_follows_only(csv_path):
    base = plan_keys(csv_pipeline(csv_path, offset=1))
    edited = plan_keys(csv_pipeline(csv_path, offset=2))
    assert len(base) == len(edited)
    changed = [a != b for (_, a), (_, b) in zip(base, edited)]
    first = changed.index(True)
    assert first > 0, "the source is upstream of the edit and keeps its key"
    assert all(changed[first:]), "everything downstream of the edit is rekeyed"


def test_touching_the_source_rekeys_everything(csv_path):
    base = plan_keys(csv_pipeline(csv_path))
    os.utime(csv_path, ns=(1, 1))
    assert all(a != b for (_, a), (_, b) in zip(base, plan_keys(csv_pipeline(csv_path))))


def test_a_window_by_calendar_offset_is_keyed(tmp_path):
    path = tmp_path / "events.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2020-03-01", "2021-06-01"]),
                  "cutoff": pd.to_datetime(["2020-01-01", "2020-01-01"])}).to_parquet(path)
    events = st.as_data_op(str(path)).skb.apply_func(pd.read_parquet)
    in_window = ((events["date"] >= events["cutoff"])
                 & (events["date"] < events["cutoff"] + pd.DateOffset(years=1)))
    reasons = {}
    keys = plan_keys(events[in_window], reasons)
    assert all(key is not None for _, key in keys), reasons


def test_standalone_arithmetic_is_keyed(tmp_path):
    # Arithmetic no assign or mask consumes: on columns a series map (a number, a
    # string concatenation, a calendar offset), else a NumericOp (a numpy ufunc).
    path = tmp_path / "lots.parquet"
    pd.DataFrame({"boro": ["1", "2"], "block": ["7", "88"], "x": [1.5, 2.0],
                  "d": pd.to_datetime(["2020-02-29", "2021-03-01"])}).to_parquet(path)
    lots = st.as_data_op(str(path)).skb.apply_func(pd.read_parquet)
    for dag in (lots["x"] * 2 + 1, lots["boro"] + lots["block"],
                lots["d"] + pd.DateOffset(years=1), lots["x"].skb.apply_func(np.log)):
        reasons = {}
        keys = plan_keys(dag, reasons)
        assert all(key is not None for _, key in keys), reasons


def test_arithmetic_is_keyed_by_operation_and_operands(csv_path):
    data = read_csv(csv_path)
    x, y, frame = data["x"], data["y"], data[["x", "y"]]
    variants = [x + 1, x - 1, 1 - x, x * 2, x / 2, x ** 3, x + y, y + x,
                x.skb.apply_func(np.exp), x.skb.apply_func(np.sqrt), frame + 1, frame * 2]
    final_keys = [plan_keys(v)[-1][1] for v in variants]
    assert None not in final_keys
    assert len(set(final_keys)) == len(variants)


def test_series_maps_are_keyed_by_their_expression(csv_path):
    data = read_csv(csv_path)
    x, y = data["x"], data["y"]
    # The same expression spelled twice, and two that differ only in a constant or
    # in a column.
    finals = [plan_keys(dag)[-1] for dag in (
        (x * 2).fillna(0), (data["x"] * 2).fillna(0), (x * 2).fillna(1), (y * 2).fillna(0))]
    assert all("series_map" in op for op, _ in finals), finals
    keys = [key for _, key in finals]
    assert None not in keys
    assert keys[0] == keys[1]
    assert len(set(keys)) == 3


def test_an_unknown_numpy_function_is_uncacheable(csv_path):
    reasons = {}
    keys = plan_keys(read_csv(csv_path)["x"].skb.apply_func(np.cumsum), reasons)
    assert keys[-1][1] is None
    assert any("cumsum" in str(why) for why in reasons.values())


def test_in_memory_inputs_are_uncacheable():
    data = st.as_data_op(pd.DataFrame({"x": [1, 2]}))
    dag = data.assign(z=data["x"] + 1)
    plan, _ = optimize(dag, env=get_data(dag))
    reasons = {}
    keys = compute_lineage_keys(plan, reasons=reasons)
    assert all(k is None for k in keys.values())
    source = next(op for op in plan if isinstance(op, InMemoryFrameOp))
    assert "in-memory" in reasons[source]


def test_generic_ops_are_uncacheable(csv_path):
    """A call stratum does not understand gets no key, nor does anything after it."""
    def double(df):
        return df * 2

    reasons = {}
    keys = plan_keys(read_csv(csv_path).skb.apply_func(double)[["x"]], reasons)
    assert keys[0][1] is not None
    assert all(k is None for _, k in keys[1:])
    assert "not a cacheable operator" in "".join(reasons.values())


def test_sample_needs_a_seed(csv_path):
    data = read_csv(csv_path)
    reasons = {}
    assert plan_keys(data.sample(2)[["x"]], reasons)[1][1] is None
    assert "random_state" in "".join(reasons.values())
    assert all(k is not None for _, k in plan_keys(data.sample(2, random_state=0)[["x"]]))


def test_region_limits_keys(csv_path):
    dag = csv_pipeline(csv_path)
    plan, _ = optimize(dag, env=get_data(dag))
    source = next(op for op in plan if isinstance(op, FileReadOp))
    keys = compute_lineage_keys(plan, region={source})
    assert keys[source] is not None
    assert all(k is None for op, k in keys.items() if op is not source)


def test_keys_are_stable_across_processes(csv_path):
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
        from tests.optimizer.test_lineage import csv_pipeline, plan_keys
        print([k for _, k in plan_keys(csv_pipeline({str(csv_path)!r}))])
    """)
    outputs = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run([sys.executable, "-c", script], env=env, cwd=REPO,
                                capture_output=True, text=True, check=True)
        outputs.add(result.stdout.strip().splitlines()[-1])
    assert len(outputs) == 1
    assert outputs == {str([k for _, k in plan_keys(csv_pipeline(csv_path))])}
