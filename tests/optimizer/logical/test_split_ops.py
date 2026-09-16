import unittest

import numpy as np
import polars as pl
from stratum.optimizer.logical._split_ops import SplitOp
from .test_dataframe_ops import _inp, _inputs_for


class TestSplitOp(unittest.TestCase):
    def _make(self, x, y, indices):
        op = SplitOp(inputs=[_inp(x), _inp(y)])
        op.indices = indices
        return op

    def test_polars(self):
        op = self._make(pl.DataFrame({"a": [10, 20, 30]}),
                        pl.DataFrame({"b": [1, 2, 3]}), [0, 2])
        result = op.process("fit_transform", _inputs_for(op))
        self.assertEqual(2, len(result[0]))

    def test_numpy(self):
        op = self._make(np.array([10, 20, 30, 40]), np.array([1, 2, 3, 4]), [1, 3])
        result = op.process("fit_transform", _inputs_for(op))
        self.assertEqual([20, 40], result[0].tolist())
        self.assertEqual([2, 4], result[1].tolist())

    def test_unsupported_type_raises(self):
        op = self._make("not_a_df", "not_a_df", [0])
        with self.assertRaises(ValueError):
            op.process("fit_transform", _inputs_for(op))


if __name__ == "__main__":
    unittest.main()
