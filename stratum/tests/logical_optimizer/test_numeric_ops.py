import unittest
import pandas as pd
import stratum as st
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
        data = st.as_data_op(self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        t1 = X.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.log1p)
        y_exp = y.skb.apply_func(np.exp)
        pred = t2.skb.apply(DummyRegressor(), y=y_exp)

        with st.config(scheduler=True):
            pred.skb.make_grid_search(cv=3)

    def test_to_numeric_op_abs(self):
        data = st.as_data_op(self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        t1 = X.skb.apply_func(np.abs)
        pred = t1.skb.apply(DummyRegressor(), y=y)

        with st.config(scheduler=True):
            pred.skb.make_grid_search(cv=3)

    def test_process_log(self):
        op = NumericOp(inputs=[], outputs=None, func=np.log)
        result = op.process("fit", {}, [np.array([1.0, np.e, np.e**2])])
        np.testing.assert_array_almost_equal(result, np.array([0.0, 1.0, 2.0]))

    def test_process_exp(self):
        op = NumericOp(inputs=[], outputs=None, func=np.exp)
        result = op.process("fit", {}, [np.array([0.0, 1.0, 2.0])])
        np.testing.assert_array_almost_equal(result, np.array([1.0, np.e, np.e**2]))

    def test_process_sqrt(self):
        op = NumericOp(inputs=[], outputs=None, func=np.sqrt)
        result = op.process("fit", {}, [np.array([4.0, 9.0, 16.0])])
        np.testing.assert_array_almost_equal(result, np.array([2.0, 3.0, 4.0]))

    def test_process_abs(self):
        op = NumericOp(inputs=[], outputs=None, func=np.abs)
        result = op.process("fit", {}, [np.array([-3.0, 0.0, 5.0])])
        np.testing.assert_array_almost_equal(result, np.array([3.0, 0.0, 5.0]))

    def test_process_square(self):
        op = NumericOp(inputs=[], outputs=None, func=np.square)
        result = op.process("fit", {}, [np.array([2.0, 3.0, 4.0])])
        np.testing.assert_array_almost_equal(result, np.array([4.0, 9.0, 16.0]))

    def test_unsupported_numeric_op(self):
        op = NumericOp(inputs=[], outputs=None, func=np.cos)
        op.type = "unsupported"
        with self.assertRaises(ValueError):
            op.process("fit", {}, [])