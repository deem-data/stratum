import unittest

import numpy as np
import polars as pl
from stratum.optimizer.logical._split_ops import SplitOp
from .test_dataframe_ops import _inp, _inputs_for


class TestSplitOp(unittest.TestCase):
    def _make(self, x, y, indices):
        return SplitOp(inputs=[_inp(x), _inp(y), _inp(indices)])

    def test_polars(self):
        op = self._make(pl.DataFrame({"a": [10, 20, 30]}),
                        pl.DataFrame({"b": [1, 2, 3]}), ([0, 2], [1]))
        x_train, y_train, x_test, y_test = op.process(_inputs_for(op))
        self.assertEqual(2, len(x_train))
        self.assertEqual([20], x_test["a"].to_list())

    def test_numpy(self):
        op = self._make(np.array([10, 20, 30, 40]), np.array([1, 2, 3, 4]), ([1, 3], [0]))
        x_train, y_train, x_test, y_test = op.process(_inputs_for(op))
        self.assertEqual([20, 40], x_train.tolist())
        self.assertEqual([2, 4], y_train.tolist())
        self.assertEqual([10], x_test.tolist())
        self.assertEqual([1], y_test.tolist())

    def test_unsupported_type_raises(self):
        op = self._make("not_a_df", "not_a_df", ([0], [0]))
        with self.assertRaises(ValueError):
            op.process(_inputs_for(op))


if __name__ == "__main__":
    unittest.main()
