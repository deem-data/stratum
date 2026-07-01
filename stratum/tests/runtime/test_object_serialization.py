import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from polars.testing import assert_series_equal as pl_assert_series_equal

from stratum.runtime._object_serialization import (
    _LeafRef,
    delete_object,
    deserialize_object,
    serialize_object,
)


class TestBareObjects(unittest.TestCase):
    """A single structured object is written straight to its native file."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _stem(self, name: str = "x") -> Path:
        return self.root / name

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        h = serialize_object(df, self._stem())
        self.assertEqual(h.format, "pandas_dataframe")
        self.assertEqual(h.leaves, ())
        self.assertEqual(h.path.suffix, ".parquet")
        self.assertGreater(h.size_on_disk, 0)
        pd.testing.assert_frame_equal(deserialize_object(h), df)

    def test_pandas_series(self):
        ser = pd.Series([1, 2, 3], name="x")
        h = serialize_object(ser, self._stem())
        pd.testing.assert_series_equal(deserialize_object(h), ser)

    def test_polars_dataframe(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        h = serialize_object(df, self._stem())
        pl_assert_frame_equal(deserialize_object(h), df)

    def test_polars_series(self):
        ser = pl.Series("x", [1, 2, 3])
        h = serialize_object(ser, self._stem())
        pl_assert_series_equal(deserialize_object(h), ser)

    def test_numpy(self):
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        h = serialize_object(arr, self._stem())
        self.assertEqual(h.path.suffix, ".npy")
        np.testing.assert_array_equal(deserialize_object(h), arr)

    def test_bare_primitive(self):
        # A lone primitive has no native format, so it lands in a skeleton pickle.
        for i, val in enumerate(["hello", 42, 3.14, True, b"raw", None]):
            h = serialize_object(val, self._stem(f"p{i}"))
            self.assertEqual(h.format, "skeleton")
            self.assertEqual(h.leaves, ())
            self.assertEqual(deserialize_object(h), val)

    def test_unsupported_type_raises(self):
        class Foo:
            pass

        with self.assertRaises(ValueError):
            serialize_object(Foo(), self._stem())

    def test_no_tmp_after_serialize(self):
        serialize_object(np.arange(10), self._stem())
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_delete_bare(self):
        h = serialize_object("hello", self._stem())
        self.assertTrue(h.path.exists())
        delete_object(h)
        self.assertFalse(h.path.exists())
        delete_object(h)  # idempotent


class TestSkeleton(unittest.TestCase):
    """Containers keep structured leaves in native files and pickle the rest."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self._n = 0

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _stem(self) -> Path:
        self._n += 1
        return self.root / f"obj{self._n}"

    def test_tuple_of_structured_uses_native_leaves(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        arr = np.arange(5, dtype=np.float64)
        h = serialize_object((df, arr), self._stem())
        self.assertEqual(h.format, "skeleton")
        self.assertEqual(h.path.suffix, ".pkl")
        self.assertEqual(len(h.leaves), 2)
        self.assertTrue(all(isinstance(l, _LeafRef) for l in h.leaves))
        # frames/arrays stay portable
        self.assertEqual({Path(l.path).suffix for l in h.leaves}, {".parquet", ".npy"})
        out_df, out_arr = deserialize_object(h)
        pd.testing.assert_frame_equal(out_df, df)
        np.testing.assert_array_equal(out_arr, arr)

    def test_nested_dict_with_list(self):
        obj = {
            "df": pd.DataFrame({"a": [1, 2]}),
            "arrs": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            "label": "train",
            "n": 3,
        }
        h = serialize_object(obj, self._stem())
        self.assertEqual(len(h.leaves), 3)  # 1 frame + 2 arrays; primitives inline
        out = deserialize_object(h)
        pd.testing.assert_frame_equal(out["df"], obj["df"])
        for a, b in zip(out["arrs"], obj["arrs"]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(out["label"], "train")
        self.assertEqual(out["n"], 3)

    def test_list_of_primitives_is_single_file(self):
        # The reviewer's case: many small objects must not fan out to N files.
        data = [f"item_{i}" for i in range(1000)]
        h = serialize_object(data, self._stem())
        self.assertEqual(h.leaves, ())
        self.assertEqual(len(list(self.root.iterdir())), 1)  # one skeleton pickle
        self.assertEqual(deserialize_object(h), data)

    def test_non_string_dict_keys_roundtrip(self):
        obj = {1: np.arange(3), (2, 3): "pair", None: 4}
        h = serialize_object(obj, self._stem())
        out = deserialize_object(h)
        np.testing.assert_array_equal(out[1], np.arange(3))
        self.assertEqual(out[(2, 3)], "pair")
        self.assertEqual(out[None], 4)

    def test_unsupported_leaf_raises(self):
        class Foo:
            pass

        with self.assertRaises(ValueError):
            serialize_object([pd.DataFrame({"a": [1]}), Foo()], self._stem())

    def test_delete_removes_skeleton_and_leaves(self):
        h = serialize_object((np.arange(3), np.arange(4)), self._stem())
        paths = [h.path, *(Path(l.path) for l in h.leaves)]
        self.assertTrue(all(p.exists() for p in paths))
        delete_object(h)
        self.assertFalse(any(p.exists() for p in paths))
        delete_object(h)  # idempotent

    def test_size_on_disk_matches_files(self):
        h = serialize_object([np.arange(10), np.arange(20)], self._stem())
        on_disk = h.path.stat().st_size + sum(Path(l.path).stat().st_size for l in h.leaves)
        self.assertEqual(h.size_on_disk, on_disk)


if __name__ == "__main__":
    unittest.main()
