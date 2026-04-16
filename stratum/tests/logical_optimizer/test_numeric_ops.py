import unittest
import pandas as pd
import stratum as skrub
import numpy as np
from sklearn.dummy import DummyRegressor
from stratum.optimizer.ir._numeric_ops import NumericOp

class TestNumericOps(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        })

    def test_to_numeric_op1(self):
        data = skrub.as_data_op(self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        t1 = X.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.log1p)
        y_exp = y.skb.apply_func(np.exp)
        pred = t2.skb.apply(DummyRegressor(), y=y_exp)

        with skrub.config(scheduler=True):
            pred.skb.make_grid_search(cv=3)

    def test_unsupported_numeric_op(self):
        op = NumericOp(np.cos, None, None, [], [])
        op.type = "unsupported"
        with self.assertRaises(ValueError):
            op.process("fit", {}, [])