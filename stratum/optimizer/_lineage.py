"""Stable names for the values computed by operators.

A lineage key hashes an op's concrete implementation class, output frame library,
configuration and input keys in operand order. Unlike the identity-based structure
keys used by CSE, it is stable across processes with the same code and libraries.

An op opts in with ``cacheable = True`` and supplies plain configuration values
through ``IRNode.lineage_config()``. Unsupported operators or values raise
``Uncacheable``; they and their descendants receive no key. This module computes
keys only: source fingerprints, operator opt-ins and reuse planning are separate.

Key computation visits a topologically ordered plan once and encodes configuration,
never frame or array contents. Library versions are not encoded: keys are only
comparable while the implementation and its dependencies remain unchanged.
"""
from __future__ import annotations

import datetime
import decimal
import hashlib
import struct
from typing import Iterable

import numpy as np
import pandas as pd

from stratum.optimizer.logical._base import IRNode, OperandRef, Uncacheable

__all__ = ["Uncacheable", "LineageEncoder", "compute_lineage_keys", "lineage_key"]

#: Bytes of a lineage key (hex-encoded to twice as many characters).
KEY_BYTES = 16

#: The encoder gives up beyond this many encoded values: a constant that large is
#: data, and encoding it would make key computation scale with it.
MAX_ENCODED_VALUES = 100_000

#: Scalars named exactly by their type and repr: numpy scalars, decimals, dates and
#: times (pandas' Timestamp and Timedelta among them) and the missing-value markers.
_SCALAR_TYPES = (np.generic, decimal.Decimal, datetime.date, datetime.time,
                 datetime.timedelta, type(pd.NaT), type(pd.NA))


def _dtype_field(value):
    """A field of an extension dtype as plain values, or ``value`` itself if the
    encoder takes it as it is."""
    if isinstance(value, pd.Index):
        return ("Index", _dtype_field(value.dtype), value.tolist())
    if isinstance(value, datetime.tzinfo):
        return ("tz", str(value))
    if isinstance(value, pd.offsets.BaseOffset):
        return ("freq", value.freqstr)
    if type(value).__module__.startswith("pyarrow"):
        return ("pyarrow", str(value))
    return value


#: Python types that stand for a data type (``astype(float)``).
_BUILTIN_DTYPES = {str: "str", int: "int", float: "float", bool: "bool", object: "object"}

_LEN = struct.Struct("<Q")


class LineageEncoder:
    """Canonical byte encoding of plain values, digested with blake2b.

    Every value is written as a type tag followed by its payload; variable-length
    payloads are length-prefixed, so distinct values never encode alike. Plain
    values are ``None``, ``bool``, ``int``, ``float``, ``str``, ``bytes``, slices,
    tuples, lists, dicts and sets of plain values, ``OperandRef``s, the literals in
    ``_SCALAR_TYPES``, pandas' calendar offsets, and numpy and pandas data types. An object can name itself by
    a ``lineage()`` method returning plain values, as column expressions do.
    Anything else raises :class:`Uncacheable`.
    """

    def __init__(self):
        # One buffer, hashed once: thousands of small `update` calls cost more
        # than the hashing itself.
        self._buf = bytearray()
        self._count = 0

    def hexdigest(self) -> str:
        return hashlib.blake2b(self._buf, digest_size=KEY_BYTES).hexdigest()

    # --- primitives ---------------------------------------------------------

    def _tag(self, tag: bytes) -> None:
        self._count += 1
        if self._count > MAX_ENCODED_VALUES:
            raise Uncacheable(f"configuration exceeds {MAX_ENCODED_VALUES} values")
        self._buf += tag

    def _bytes(self, data: bytes) -> None:
        self._buf += _LEN.pack(len(data))
        self._buf += data

    def _str(self, s: str) -> None:
        self._bytes(s.encode("utf-8", "surrogatepass"))

    def raw(self, tag: bytes, data: bytes) -> None:
        """Write a tagged byte string (an op's class, an input key)."""
        self._tag(tag)
        self._bytes(data)

    # --- values -------------------------------------------------------------

    def encode(self, value) -> None:
        if value is None:
            self._tag(b"N")
        elif value is True or value is False:
            self._tag(b"T" if value else b"F")
        elif type(value) is int:
            self._tag(b"i")
            self._str(str(value))
        elif type(value) is float:
            self._tag(b"f")
            self._str(repr(value))  # distinguishes nan, inf and -0.0
        elif type(value) is str:
            self._tag(b"s")
            self._str(value)
        elif type(value) is bytes:
            self._tag(b"b")
            self._bytes(value)
        elif isinstance(value, OperandRef):
            self._tag(b"$")
            self._str(str(value.k))
        elif type(value) in (tuple, list):
            self._tag(b"(" if type(value) is tuple else b"[")
            self._sequence(value)
        elif type(value) is dict:
            # Insertion order is kept: it is semantic for some ops (the column
            # order of `assign`), and deterministic for a given script.
            self._tag(b"{")
            self._str(str(len(value)))
            for k, v in value.items():
                self.encode(k)
                self.encode(v)
        elif type(value) in (set, frozenset):
            self._tag(b"<")
            self._set(value)
        elif type(value) is slice:
            self._tag(b":")
            self._sequence((value.start, value.stop, value.step))
        elif isinstance(value, _SCALAR_TYPES):
            self._tag(b"~")
            self._str(type(value).__qualname__)
            self._str(repr(value))
        elif isinstance(value, pd.offsets.BaseOffset):
            # A calendar offset (``DateOffset(years=1)``) is what pandas compares it
            # by; its repr leaves out ``normalize``. Only pandas' own classes: a
            # subclass may move dates by code of its own.
            cls = type(value)
            if cls.__module__.partition(".")[0] != "pandas":
                raise Uncacheable(f"no lineage encoding for {cls.__qualname__}")
            self._tag(b"O")
            self._str(cls.__qualname__)
            self.encode(value.n)
            self.encode(value.normalize)
            self.encode(dict(sorted(value.kwds.items())))
        elif isinstance(value, np.dtype):
            self._tag(b"D")
            self._str(type(value).__qualname__)
            self._str(str(value))
        elif isinstance(value, pd.api.extensions.ExtensionDtype):
            # Named by the fields pandas compares dtypes by: its string leaves out
            # the categories of a categorical, and a string dtype's storage.
            self._tag(b"E")
            self._str(type(value).__qualname__)
            fields = type(value)._metadata
            self._str(str(len(fields)))
            for name in fields:
                self.encode(_dtype_field(getattr(value, name)))
        elif isinstance(value, type) and value in _BUILTIN_DTYPES:
            self._tag(b"t")
            self._str(_BUILTIN_DTYPES[value])
        elif callable(getattr(type(value), "lineage", None)):
            self._tag(b"L")
            self.encode(value.lineage())
        else:
            raise Uncacheable(f"no lineage encoding for {type(value).__qualname__}")

    def _sequence(self, values: Iterable) -> None:
        values = list(values)
        self._str(str(len(values)))
        for v in values:
            self.encode(v)

    def _set(self, values) -> None:
        """A set's members in the order of their encodings, so iteration order
        (which follows the per-process ``hash()``) does not matter."""
        parts = []
        for v in values:
            enc = LineageEncoder()
            enc._count = self._count
            enc.encode(v)
            self._count = enc._count
            parts.append(bytes(enc._buf))
        self._str(str(len(parts)))
        for part in sorted(parts):
            self._buf += part


def lineage_key(op: IRNode, input_keys: list[str]) -> str:
    """The lineage key of ``op`` given its inputs' keys. Raises :class:`Uncacheable`."""
    enc = LineageEncoder()
    cls = type(op)
    enc.raw(b"op", f"{cls.__module__}.{cls.__qualname__}".encode())
    lib = getattr(op, "_frame_lib", None)
    enc.encode(None if lib is None else str(lib))
    enc.encode(op.lineage_config())
    enc._str(str(len(input_keys)))
    for key in input_keys:
        enc.raw(b"in", key.encode())
    return enc.hexdigest()


def compute_lineage_keys(ops: Iterable[IRNode], region: set | None = None,
                         reasons: dict | None = None) -> dict[IRNode, str | None]:
    """Lineage keys for ``ops``, given in topological order (inputs first).

    An op outside ``region`` (when given), one that is :class:`Uncacheable`, or one
    with an input lacking a key gets ``None``. Pass a dict as ``reasons`` to learn
    why each keyless op in the region has no key.
    """
    keys: dict[IRNode, str | None] = {}
    for op in ops:
        key = None
        if region is None or op in region:
            input_keys = [keys.get(in_) for in_ in op.inputs]
            if any(k is None for k in input_keys):
                if reasons is not None:
                    reasons[op] = "an input has no key"
            else:
                try:
                    key = lineage_key(op, input_keys)
                except Uncacheable as exc:
                    if reasons is not None:
                        reasons[op] = str(exc)
        keys[op] = key
    return keys
