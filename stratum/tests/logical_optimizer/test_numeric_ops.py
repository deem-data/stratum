import unittest
import pandas as pd
import stratum as st
import numpy as np
from sklearn.dummy import DummyRegressor
from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer._optimize import optimize

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

    def test_process_add_var_const(self):
        op = NumericOp([], [], type=NumericOpType.ADD, constant=2.0, reversed=False)
        result = op.process("fit", {}, [np.array([1.0, 2.0, 3.0])])
        np.testing.assert_array_almost_equal(result, np.array([3.0, 4.0, 5.0]))

    def test_process_add_const_var(self):
        op = NumericOp([], [], type=NumericOpType.ADD, constant=10.0, reversed=True)
        result = op.process("fit", {}, [np.array([1.0, 2.0, 3.0])])
        np.testing.assert_array_almost_equal(result, np.array([11.0, 12.0, 13.0]))

    def test_process_subtract_var_const(self):
        op = NumericOp([], [], type=NumericOpType.SUBTRACT, constant=1.0, reversed=False)
        result = op.process("fit", {}, [np.array([4.0, 5.0, 6.0])])
        np.testing.assert_array_almost_equal(result, np.array([3.0, 4.0, 5.0]))

    def test_process_subtract_const_var(self):
        op = NumericOp([], [], type=NumericOpType.SUBTRACT, constant=10.0, reversed=True)
        result = op.process("fit", {}, [np.array([1.0, 2.0, 3.0])])
        np.testing.assert_array_almost_equal(result, np.array([9.0, 8.0, 7.0]))

    def test_process_multiply_var_const(self):
        op = NumericOp([], [], type=NumericOpType.MULTIPLY, constant=3.0, reversed=False)
        result = op.process("fit", {}, [np.array([1.0, 2.0, 3.0])])
        np.testing.assert_array_almost_equal(result, np.array([3.0, 6.0, 9.0]))

    def test_process_multiply_const_var(self):
        op = NumericOp([], [], type=NumericOpType.MULTIPLY, constant=2.0, reversed=True)
        result = op.process("fit", {}, [np.array([1.0, 2.0, 3.0])])
        np.testing.assert_array_almost_equal(result, np.array([2.0, 4.0, 6.0]))

    def test_process_divide_var_const(self):
        op = NumericOp([], [], type=NumericOpType.DIVIDE, constant=2.0, reversed=False)
        result = op.process("fit", {}, [np.array([2.0, 4.0, 6.0])])
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_process_divide_const_var(self):
        op = NumericOp([], [], type=NumericOpType.DIVIDE, constant=12.0, reversed=True)
        result = op.process("fit", {}, [np.array([2.0, 3.0, 4.0])])
        np.testing.assert_array_almost_equal(result, np.array([6.0, 4.0, 3.0]))

    def test_extract_add_var_const(self):
        df = st.as_data_op(5)
        t1 = df + 3
        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.ADD)
        self.assertEqual(out[1].constant, 3)
        self.assertFalse(out[1].reversed)

    def test_extract_add_const_var(self):
        df = st.as_data_op(5)
        t1 = 3 + df
        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.ADD)
        self.assertEqual(out[1].constant, 3)
        self.assertTrue(out[1].reversed)

    def test_extract_subtract_var_const(self):
        df = st.as_data_op(5)
        t1 = df - 2
        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.SUBTRACT)

    def test_extract_multiply_var_const(self):
        df = st.as_data_op(5)
        t1 = df * 4
        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.MULTIPLY)

    def test_extract_divide_var_const(self):
        df = st.as_data_op(10)
        t1 = df / 2
        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.DIVIDE)

    def test_no_extract_var_var(self):
        """BinOp(var + var) must not be converted — keep as BinOp."""
        df1 = st.as_data_op(2)
        df2 = st.as_data_op(3)
        t1 = df1 + df2
        out, *_ = optimize(t1)
        binary_ops = [op for op in out if isinstance(op, NumericOp) and op.type == NumericOpType.ADD]
        self.assertEqual(len(binary_ops), 0)

    def test_extract_add_produces_correct_result(self):
        df = st.as_data_op(5)
        t1 = df + 3
        out, *_ = optimize(t1)
        add_op = next(op for op in out if isinstance(op, NumericOp) and op.type == NumericOpType.ADD)
        self.assertEqual(add_op.process("fit", {}, [5]), 8)

    def test_extract_np_add_callop(self):
        """CallOp with np.add should be extracted to NumericOp ADD."""
        df = st.as_data_op(5)
        t1 = df.skb.apply_func(np.add, 3)
        out, *_ = optimize(t1)
        add_ops = [op for op in out if isinstance(op, NumericOp) and op.type == NumericOpType.ADD]
        self.assertEqual(len(add_ops), 1)

    def test_extract_np_multiply_callop(self):
        """CallOp with np.multiply should be extracted to NumericOp MULTIPLY."""
        df = st.as_data_op(5)
        t1 = df.skb.apply_func(np.multiply, 4)
        out, *_ = optimize(t1)
        mul_ops = [op for op in out if isinstance(op, NumericOp) and op.type == NumericOpType.MULTIPLY]
        self.assertEqual(len(mul_ops), 1)

    def test_no_extract_np_add_var_var(self):
        """apply_func(np.add, var) with two DataOp inputs must not produce a binary NumericOp."""
        df1 = st.as_data_op(2)
        df2 = st.as_data_op(3)
        t1 = df1.skb.apply_func(np.add, df2)
        out, *_ = optimize(t1)
        binary_ops = [op for op in out if isinstance(op, NumericOp) and op.type == NumericOpType.ADD]
        self.assertEqual(len(binary_ops), 0)

    def test_no_extract_np_subtract_var_var(self):
        """apply_func(np.subtract, var) with two DataOp inputs must not produce a binary NumericOp."""
        df1 = st.as_data_op(5)
        df2 = st.as_data_op(3)
        t1 = df1.skb.apply_func(np.subtract, df2)
        out, *_ = optimize(t1)
        binary_ops = [op for op in out if isinstance(op, NumericOp) and op.type == NumericOpType.SUBTRACT]
        self.assertEqual(len(binary_ops), 0)

    def test_no_extract_np_multiply_var_var(self):
        """apply_func(np.multiply, var) with two DataOp inputs must not produce a binary NumericOp."""
        df1 = st.as_data_op(2)
        df2 = st.as_data_op(3)
        t1 = df1.skb.apply_func(np.multiply, df2)
        out, *_ = optimize(t1)
        binary_ops = [op for op in out if isinstance(op, NumericOp) and op.type == NumericOpType.MULTIPLY]
        self.assertEqual(len(binary_ops), 0)

    def test_no_extract_np_divide_var_var(self):
        """apply_func(np.divide, var) with two DataOp inputs must not produce a binary NumericOp."""
        df1 = st.as_data_op(6)
        df2 = st.as_data_op(2)
        t1 = df1.skb.apply_func(np.divide, df2)
        out, *_ = optimize(t1)
        binary_ops = [op for op in out if isinstance(op, NumericOp) and op.type == NumericOpType.DIVIDE]
        self.assertEqual(len(binary_ops), 0)

    def test_extract_subtract_const_var_produces_correct_result(self):
        df = st.as_data_op(3)
        t1 = 10 - df
        out, *_ = optimize(t1)
        op = next(o for o in out if isinstance(o, NumericOp) and o.type == NumericOpType.SUBTRACT)
        self.assertEqual(op.process("fit", {}, [3]), 7)

    def test_extract_divide_const_var_produces_correct_result(self):
        df = st.as_data_op(4)
        t1 = 12 / df
        out, *_ = optimize(t1)
        op = next(o for o in out if isinstance(o, NumericOp) and o.type == NumericOpType.DIVIDE)
        self.assertEqual(op.process("fit", {}, [4]), 3.0)

    def test_make_binary_numeric_op_raises_on_invalid_args(self):
        """make_binary_numeric_op must raise ValueError when neither or both args are placeholders."""
        from stratum.optimizer.ir._numeric_ops import make_binary_numeric_op
        from stratum.optimizer.ir._ops import CallOp
        op = CallOp(func=np.add, args=None)
        op.args = (1.0, 2.0)  # neither arg is DATA_OP_PLACEHOLDER
        with self.assertRaises(ValueError):
            make_binary_numeric_op(op, NumericOpType.ADD)
