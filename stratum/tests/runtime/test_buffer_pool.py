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
        self.assertEqual(pool.get(op), "data_x")
        self.assertEqual(pool.active_count, 1)

    def test_get_missing_returns_none(self):
        pool = BufferPool()
        self.assertIsNone(pool.get(_make_op("missing")))

    def test_release_drops_data(self):
        pool = BufferPool()
        op = _make_op("x")
        pool.put(op, "data_x")
        released = pool.release(op)
        self.assertTrue(released)
        self.assertIsNone(pool.get(op))
        self.assertEqual(pool.active_count, 0)
        self.assertEqual(pool.total_released, 1)

    def test_release_missing_returns_false(self):
        pool = BufferPool()
        self.assertFalse(pool.release(_make_op("missing")))

    def test_release_all(self):
        pool = BufferPool()
        ops = [_make_op(f"op{i}") for i in range(3)]
        for i, op in enumerate(ops):
            pool.put(op, f"data_{i}")

        released = pool.release_all()
        self.assertEqual(set(released), set(ops))
        self.assertEqual(pool.active_count, 0)
        self.assertEqual(pool.total_released, 3)

    def test_put_overwrites_existing(self):
        pool = BufferPool()
        op = _make_op("x")
        pool.put(op, "old_data")
        pool.put(op, "new_data")
        self.assertEqual(pool.get(op), "new_data")
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
