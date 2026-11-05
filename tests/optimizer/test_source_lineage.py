"""Source and constant lineage, including invalidation without reading data."""
import io
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

import stratum as st
from stratum.optimizer._lineage import Uncacheable, compute_lineage_keys, lineage_key
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical import _source_ops
from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.logical._source_ops import (
    InMemoryFrameOp, LoadNumpyOp, ReadCSVOp, ReadParquetOp, file_fingerprint,
)
from stratum.optimizer.physical._impl_selection import FlagBasedSelector
from stratum.optimizer.physical._source_execs import (
    PandasReadCSV, PolarsReadCSV,
)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,2\n3,4\n")
    return path


def test_local_fingerprint_uses_path_and_metadata_only(csv_path, monkeypatch):
    monkeypatch.chdir(csv_path.parent)
    with patch("builtins.open", side_effect=AssertionError("read file contents")):
        original = file_fingerprint(csv_path)
        assert original == file_fingerprint(str(csv_path)) == file_fingerprint("data.csv")
    metadata = csv_path.stat()
    os.utime(csv_path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000))
    assert file_fingerprint(csv_path) != original

    csv_path.write_text("x,y\n1,2\n3,4\n5,6\n")
    os.utime(csv_path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert file_fingerprint(csv_path) != original, "size changes also invalidate"
    renamed = csv_path.with_name("renamed.csv")
    before_rename = file_fingerprint(csv_path)
    csv_path.rename(renamed)
    assert file_fingerprint(renamed) != before_rename


def test_directory_fingerprint_tracks_nested_files(tmp_path):
    directory = tmp_path / "dataset"
    for part in ("year=2", "year=1"):
        (directory / part).mkdir(parents=True)
        (directory / part / "part.parquet").write_bytes(b"data")
    original = file_fingerprint(directory)
    assert original == file_fingerprint(directory)
    assert [name for name, *_ in original[1]] == [
        os.path.join("year=1", "part.parquet"), os.path.join("year=2", "part.parquet")]
    added = directory / "year=1" / "extra.parquet"
    added.write_bytes(b"data")
    assert file_fingerprint(directory) != original
    added.unlink()
    assert file_fingerprint(directory) == original
    (directory / "year=1" / "part.parquet").write_bytes(b"changed data")
    assert file_fingerprint(directory) != original


@pytest.mark.parametrize("path", [
    "missing.csv", "*.csv", "s3://bucket/data.csv", "https://example.com/data.csv",
    io.StringIO("x\n1\n"), 1,
])
def test_unsupported_sources_have_no_fingerprint(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Uncacheable):
        file_fingerprint(path)


def test_incomplete_directory_listings_are_uncacheable(tmp_path, monkeypatch):
    def denied_walk(path, onerror):
        onerror(PermissionError("cannot list directory"))

    monkeypatch.setattr(_source_ops.os, "walk", denied_walk)
    with pytest.raises(Uncacheable, match="cannot list directory"):
        file_fingerprint(tmp_path)

    monkeypatch.setattr(_source_ops.os, "walk", lambda path, onerror: [
        (str(path), [], ["vanished.parquet"])])
    with pytest.raises(Uncacheable, match="cannot stat source"):
        file_fingerprint(tmp_path)


def test_symlinked_subdirectories_are_uncacheable(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "part.parquet").write_bytes(b"data")
    directory = tmp_path / "dataset"
    directory.mkdir()
    try:
        (directory / "linked").symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(Uncacheable, match="symlinked directory"):
        file_fingerprint(directory)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_special_files_are_uncacheable(tmp_path):
    directory = tmp_path / "dataset"
    directory.mkdir()
    fifo = directory / "input.csv"
    os.mkfifo(fifo)
    for path in (fifo, directory):
        with pytest.raises(Uncacheable, match="not a regular file"):
            file_fingerprint(path)


class FakeGCSFileSystem:
    """Only implements object metadata listings; no data-read API."""
    objects = {}
    calls = []

    def __init__(self, **storage_options):
        self.storage_options = storage_options

    def find(self, path, detail=False):
        assert detail is True
        self.calls.append((path, self.storage_options))
        prefix = path.split("://", 1)[1].rstrip("/")
        return {name: {"size": size, "generation": generation}
                for name, (size, generation) in self.objects.items()
                if name == prefix or name.startswith(prefix + "/")}


@pytest.fixture
def fake_gcs(monkeypatch):
    monkeypatch.setitem(sys.modules, "gcsfs",
                        SimpleNamespace(GCSFileSystem=FakeGCSFileSystem))
    FakeGCSFileSystem.objects = {"lake/table/year=2/part.parquet": (20, "2"),
                                 "lake/table/year=1/part.parquet": (10, "1")}
    FakeGCSFileSystem.calls = []
    return FakeGCSFileSystem


@pytest.mark.parametrize("protocol", ["gs", "gcs"])
def test_gcs_fingerprint_tracks_generations_and_fresh_listings(fake_gcs, protocol):
    path = f"{protocol}://lake/table"
    options = {"token": "key.json", "use_listings_cache": True}
    original = file_fingerprint(path, options)
    assert original == (path, (("year=1/part.parquet", 10, "1"),
                              ("year=2/part.parquet", 20, "2")))
    assert fake_gcs.calls == [(path, {"token": "key.json", "use_listings_cache": False})]
    assert options["use_listings_cache"] is True, "do not mutate reader options"
    assert file_fingerprint(path) == original
    fake_gcs.objects["lake/table/year=2/part.parquet"] = (20, "3")
    assert file_fingerprint(path) != original
    assert file_fingerprint(path + "/year=1/part.parquet")[1] == (("", 10, "1"),)
    del fake_gcs.objects["lake/table/year=1/part.parquet"]
    assert len(file_fingerprint(path)[1]) == 1


def test_gcs_sources_without_reliable_metadata_are_uncacheable(fake_gcs, monkeypatch):
    with pytest.raises(Uncacheable, match="no objects"):
        file_fingerprint("gs://lake/missing")
    fake_gcs.objects["lake/table/year=1/part.parquet"] = (10, None)
    with pytest.raises(Uncacheable, match="no generation"):
        file_fingerprint("gs://lake/table")
    with pytest.raises(Uncacheable, match="storage_options"):
        file_fingerprint("gs://lake/table", OperandRef(0))

    def denied_listing(*args, **kwargs):
        raise PermissionError("listing denied")

    monkeypatch.setattr(fake_gcs, "find", denied_listing)
    with pytest.raises(Uncacheable, match="cannot list source"):
        file_fingerprint("gs://lake/table")


def test_gcs_without_gcsfs_is_uncacheable(monkeypatch):
    monkeypatch.setitem(sys.modules, "gcsfs", None)
    with pytest.raises(Uncacheable, match="gcsfs is not installed"):
        file_fingerprint("gs://lake/table")


def test_reader_options_are_part_of_the_key(csv_path):
    ops = [PandasReadCSV(csv_path), PandasReadCSV(str(csv_path)),
           PandasReadCSV(csv_path, sep=";"), PandasReadCSV(csv_path, usecols=["x"]),
           PandasReadCSV(csv_path, pandas_options={"index_col": 0}),
           PandasReadCSV(csv_path, pandas_options={"dtype": {"x": "str"}}),
           PolarsReadCSV(csv_path)]
    keys = [lineage_key(op, []) for op in ops]
    assert keys[0] == keys[1]
    assert len(set(keys[1:])) == len(keys) - 1
    assert lineage_key(ReadParquetOp(csv_path, columns=["x"]), []) != lineage_key(
        ReadParquetOp(csv_path, columns=["y"]), [])
    assert lineage_key(LoadNumpyOp(csv_path, options={"allow_pickle": True}), []) != lineage_key(
        LoadNumpyOp(csv_path, options={"allow_pickle": False}), [])


def test_gcs_reader_keys_include_source_generation_and_options(fake_gcs):
    reader = ReadParquetOp("gs://lake/table", columns=["x"],
                           storage_options={"token": "key.json"})
    before = lineage_key(reader, [])
    assert fake_gcs.calls[-1][1]["token"] == "key.json"
    fake_gcs.objects["lake/table/year=1/part.parquet"] = (10, "3")
    assert lineage_key(reader, []) != before
    assert lineage_key(reader, []) != lineage_key(ReadParquetOp(
        "gs://lake/table", columns=["y"], storage_options={"token": "key.json"}), [])


@pytest.mark.parametrize("reader", [ReadCSVOp, ReadParquetOp, LoadNumpyOp])
def test_graph_fed_paths_are_uncacheable(reader):
    with pytest.raises(Uncacheable, match="graph-fed source path"):
        lineage_key(reader(OperandRef(0)), ["path-key"])


def test_callable_reader_options_are_uncacheable(csv_path):
    reasons = {}
    reader = PandasReadCSV(csv_path, usecols=lambda name: name == "x")
    assert compute_lineage_keys([reader], reasons=reasons) == {reader: None}
    assert "no lineage encoding" in reasons[reader]


@pytest.mark.parametrize("value", [None, 1, "path.csv", [1, 2], {"x": 1}])
def test_plain_constants_have_stable_keys(value):
    assert lineage_key(ValueOp(value), []) == lineage_key(ValueOp(value), [])


def test_different_constants_have_different_keys():
    assert lineage_key(ValueOp(1), []) != lineage_key(ValueOp(2), [])
    assert lineage_key(ValueOp(1), []) != lineage_key(ValueOp("1"), [])


@pytest.mark.parametrize("value", [pd.DataFrame({"x": [1]}), pl.DataFrame({"x": [1]}),
                                   pd.Series([1]), np.arange(3)])
def test_in_memory_data_is_uncacheable(value):
    with pytest.raises(Uncacheable, match="in-memory input frame"):
        lineage_key(InMemoryFrameOp(value), [])
    with pytest.raises(Uncacheable, match="no lineage encoding"):
        lineage_key(ValueOp(value), [])


@pytest.mark.parametrize("kind", ["csv", "parquet", "npy"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_optimized_file_sources_have_keys_that_follow_file_changes(tmp_path, kind, backend):
    path = tmp_path / f"data.{kind}"
    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    if kind == "csv":
        frame.to_csv(path, index=False)
        reader = pd.read_csv
    elif kind == "parquet":
        frame.to_parquet(path)
        reader = pd.read_parquet
    else:
        np.save(path, frame.to_numpy())
        reader = np.load

    def keys():
        plan, *_ = optimize(st.deferred(reader)(str(path)),
                            OptConfig(selector=FlagBasedSelector(backend=backend)))
        return list(compute_lineage_keys(plan).values())

    before = keys()
    assert before and all(key is not None for key in before)
    assert keys() == before
    metadata = path.stat()
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000))
    assert all(old != new for old, new in zip(before, keys()))
