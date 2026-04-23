import unittest
from stratum.runtime._buffer_pool import BufferPool
from stratum.tests.runtime.runtime_test_utils import RuntimeTest, simple_pipeline, _make_op
from stratum._api import grid_search

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
