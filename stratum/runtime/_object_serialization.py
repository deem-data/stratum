from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from itertools import count
from logging import getLogger
from pathlib import Path
from typing import Any

from numpy import ndarray, load as np_load, save as np_save
from pandas import DataFrame, Series, read_parquet
from polars import DataFrame as PolarsDataFrame, Series as PolarsSeries, read_parquet as pl_read_parquet

logger = getLogger(__name__)

_FORMAT_EXT = {
    "pandas_dataframe": ".parquet",
    "pandas_series": ".parquet",
    "polars_dataframe": ".parquet",
    "polars_series": ".parquet",
    "numpy_ndarray": ".npy",
}

# Structured leaves are the only objects that get their own portable file
# (parquet / .npy), so a spilled intermediate can be read by other tools
# (polars, arrow, rust, gpu, ...) without a Python round-trip. Everything else
# — primitives and the container glue — is folded into a single skeleton file.
_STRUCTURED_FORMAT: list[tuple[type, str]] = [
    (DataFrame, "pandas_dataframe"),
    (Series, "pandas_series"),
    (PolarsDataFrame, "polars_dataframe"),
    (PolarsSeries, "polars_series"),
    (ndarray, "numpy_ndarray"),
]


# Non-structured leaves we know how to pickle. Mirrors the types `get_size`
# accepts, so anything the buffer pool admits can also be spilled (and we reject
# the same unsupported types here, rather than silently pickling them).
_PICKLABLE_PRIMITIVE = (str, int, float, bool, bytes)


def _native_format(obj: Any) -> str | None:
    """Return the portable file format for a structured leaf, else None."""
    for typ, fmt in _STRUCTURED_FORMAT:
        if isinstance(obj, typ):
            return fmt
    return None


@dataclass(frozen=True)
class _LeafRef:
    """Placeholder left in a skeleton where a structured leaf was extracted.

    Points at the leaf's own portable file. Kept minimal (just path + format)
    so the skeleton stays a plain tree of containers, primitives and refs.
    """
    path: str
    format: str


@dataclass(frozen=True)
class SpilledObject:
    """Reference to a single spilled object (the handle the buffer pool holds).

    Two shapes:
      * a *bare* structured object → `path` is its native file, `format` is a
        native format, `leaves` is empty;
      * anything else (a primitive, or any list/tuple/dict) → `path` is the
        skeleton pickle, `format` is ``"skeleton"``, and `leaves` lists the
        native files that the skeleton's `_LeafRef`s point at.
    """
    path: Path
    format: str
    leaves: tuple[_LeafRef, ...]
    size_on_disk: int


def serialize_object(obj: Any, stem: Path) -> SpilledObject:
    """Spill a single object. `stem` is the target path prefix without extension.

    A bare frame/array/series is written straight to its native file. Any
    container is walked once: each structured leaf is peeled out to its own
    native file and replaced by a `_LeafRef`; primitives stay inline; the
    resulting skeleton is pickled to a single file.
    """
    fmt = _native_format(obj)
    if fmt is not None:
        path = _write_native(obj, stem, fmt)
        return SpilledObject(path=path, format=fmt, leaves=(), size_on_disk=path.stat().st_size)

    leaves: list[_LeafRef] = []
    ids = count()

    # Note: a container of *many small* structured objects (e.g. thousands of
    # tiny frames) still fans out to one native file each. That is a pathological
    # shape we have no real use for — normal structured leaves are large and few
    # — so we don't special-case it. If it ever matters, fold sub-threshold
    # leaves into the skeleton pickle instead of giving them their own file.
    def build(node: Any) -> Any:
        leaf_fmt = _native_format(node)
        if leaf_fmt is not None:
            leaf_path = _write_native(node, Path(f"{stem}_{next(ids)}"), leaf_fmt)
            ref = _LeafRef(path=str(leaf_path), format=leaf_fmt)
            leaves.append(ref)
            return ref
        if isinstance(node, list):
            return [build(x) for x in node]
        if isinstance(node, tuple):
            return tuple(build(x) for x in node)
        if isinstance(node, dict):
            return {k: build(v) for k, v in node.items()}
        if isinstance(node, _PICKLABLE_PRIMITIVE) or node is None:
            return node  # primitive: inline in the skeleton
        raise ValueError(f"Unsupported type for serialization: {type(node)}")

    skeleton = build(obj)
    # The skeleton is pickled, so it is Python-only. That is a deliberate
    # trade-off: the heavy data (frames/arrays) lives in portable native files
    # above, which is what another language would actually consume. If a
    # non-Python reader ever needs the *structure* too, swap this for a tagged
    # JSON encoding (tuples/bytes/non-str dict keys need tagging) and record
    # `format="json"` — the buffer pool is agnostic to the value.
    skel_path = _write_pickle(skeleton, stem)
    size = skel_path.stat().st_size + sum(Path(r.path).stat().st_size for r in leaves)
    return SpilledObject(path=skel_path, format="skeleton", leaves=tuple(leaves), size_on_disk=size)


def deserialize_object(spilled: SpilledObject) -> Any:
    if spilled.format != "skeleton":
        return _read_native(spilled.path, spilled.format)
    # Only ever loads a pickle we wrote ourselves; never an untrusted one.
    with open(spilled.path, "rb") as f:
        return _rehydrate(pickle.load(f))


def delete_object(spilled: SpilledObject) -> None:
    for path in (spilled.path, *(Path(r.path) for r in spilled.leaves)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _rehydrate(node: Any) -> Any:
    if isinstance(node, _LeafRef):
        return _read_native(Path(node.path), node.format)
    if isinstance(node, list):
        return [_rehydrate(x) for x in node]
    if isinstance(node, tuple):
        return tuple(_rehydrate(x) for x in node)
    if isinstance(node, dict):
        return {k: _rehydrate(v) for k, v in node.items()}
    return node


def _write_native(obj: Any, stem: Path, fmt: str) -> Path:
    """Write one structured object to `stem` + its native extension, atomically."""
    path = Path(f"{stem}{_FORMAT_EXT[fmt]}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if fmt == "pandas_dataframe":
        obj.to_parquet(tmp)
    elif fmt == "pandas_series":
        obj.to_frame().to_parquet(tmp)
    elif fmt == "polars_dataframe":
        obj.write_parquet(tmp)
    elif fmt == "polars_series":
        obj.to_frame().write_parquet(tmp)
    elif fmt == "numpy_ndarray":
        with open(tmp, "wb") as f:
            np_save(f, obj, allow_pickle=False)
    else:
        raise ValueError(f"Unknown native format: {fmt}")
    os.replace(tmp, path)
    return path


def _write_pickle(obj: Any, stem: Path) -> Path:
    path = Path(f"{stem}.pkl")
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def _read_native(path: Path, fmt: str) -> Any:
    if fmt == "pandas_dataframe":
        return read_parquet(path)
    if fmt == "pandas_series":
        return read_parquet(path).iloc[:, 0]
    if fmt == "polars_dataframe":
        return pl_read_parquet(path)
    if fmt == "polars_series":
        return pl_read_parquet(path).to_series(0)
    if fmt == "numpy_ndarray":
        with open(path, "rb") as f:
            return np_load(f, allow_pickle=False)
    raise ValueError(f"Unknown native format: {fmt}")
