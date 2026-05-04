import unittest
from stratum.runtime._buffer_pool import BufferPool
from stratum.runtime._object_size import get_size
from stratum.tests.runtime.runtime_test_utils import RuntimeTest, simple_pipeline, _make_op
from stratum._api import grid_search
import numpy as np
import pandas as pd
import polars as pl
class TestBufferPool(unittest.TestCase):
    """Tests for BufferPool as a pure cache."""

    def test_put_and_get(self):
        pool = BufferPool()
        op = _make_op("x")
        pool.put(op, "data_x")
        self.assertEqual(pool.pin(op), "data_x")
        self.assertEqual(pool.active_count, 1)

    def test_get_missing_returns_none(self):
        pool = BufferPool()
        self.assertIsNone(pool.pin(_make_op("missing")))

    def test_remove_drops_data(self):
        pool = BufferPool()
        op = _make_op("x")
        pool.put(op, "data_x")
        removed = pool.remove(op)
        self.assertTrue(removed)
        self.assertIsNone(pool.pin(op))
        self.assertEqual(pool.active_count, 0)
        self.assertEqual(pool.total_removed, 1)

    def test_remove_missing_returns_false(self):
        pool = BufferPool()
        self.assertFalse(pool.remove(_make_op("missing")))

    def test_remove_all(self):
        pool = BufferPool()
        ops = [_make_op(f"op{i}") for i in range(3)]
        for i, op in enumerate(ops):
            pool.put(op, f"data_{i}")

        removed = pool.remove_all()
        self.assertEqual(set(removed), set(ops))
        self.assertEqual(pool.active_count, 0)
        self.assertEqual(pool.total_removed, 3)

    def test_put_overwrites_existing(self):
        pool = BufferPool()
        op = _make_op("x")
        pool.put(op, "old_data")
        pool.put(op, "new_data")
        self.assertEqual(pool.pin(op), "new_data")
        self.assertEqual(pool.active_count, 1)

    def test_memory_usage(self):
        pool = BufferPool()
        op = _make_op("x")
        data_x = np.random.random(1024).astype(np.float64)
        pool.put(op, data_x)
        self.assertEqual(pool.memory_usage, 1024*8)
        pool.remove(op)
        self.assertEqual(pool.memory_usage, 0)
        pool.memory_usage = 2*1024**5
        self.assertEqual(pool.total_size, "2048.00 GB")

    def test_unknown_object_sizes(self):
        class Foo:
            pass

        with self.assertRaises(ValueError):
            get_size(Foo())

        with self.assertRaises(ValueError):
            get_size(pd.Index([1, 2, 3]))

        with self.assertRaises(ValueError):
            get_size(pl.LazyFrame({"a": [1, 2, 3]}))

        with self.assertRaises(ValueError):
            get_size(np.dtype("float64"))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestBufferPoolIntegration(RuntimeTest):

    def test_evaluate_matches_baseline(self):
        """Buffer-managed evaluate produces same results as skrub baseline."""
        pred_opt = simple_pipeline()
        self.compare_evaluate(pred_opt)

    def test_grid_search_runs(self):
        """Grid search with buffer manager completes without error."""
        pred_opt = simple_pipeline()
        results = grid_search(pred_opt, cv=2)
        self.assertIsNotNone(results)


if __name__ == "__main__":
    unittest.main()
