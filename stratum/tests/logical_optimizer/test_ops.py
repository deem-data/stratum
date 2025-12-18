import unittest
import pandas as pd
import stratum as skrub
from stratum.logical_optimizer._ops import (
    ImplOp, Op, ChoiceOp, ValueOp, MethodCallOp, CallOp, GetAttrOp, GetItemOp, SearchEvalOp, as_op
)
from stratum.logical_optimizer._optimize import optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

class TestOpCloning(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        })

    def test_op_clone_basic(self):
        """Test cloning a basic Op without skrub_impl."""
        op = Op(outputs=[1], inputs=[2])
        op.name = "test_op"
        try:
            cloned = op.clone()
        except NotImplementedError as e:
            self.assertEqual(str(e), f"Cloning of {op.__class__.__name__} objects is not implemented yet. Please implement it.")

    
    def test_clone_ops(self):
        """Test cloning an Op with skrub_impl."""
        data = skrub.as_data_op(self.df)
        data_op = data[["x"]].apply(lambda x: x + 1)
        pred = data_op.skb.apply(DummyRegressor(), y=data["y"])
        pred = pred.skb.apply_func(lambda x,a, b: x, 1, b=1)
        choice = skrub.choose_from([pred], name="choice").as_data_op()
        out = choice.empty
        ops = optimize(out)

        try:
            ops[0].clone()
        except ValueError as e:
            self.assertEqual(str(e), "We should not clone ValueOp objects.")
        
        cloned = ops[1].clone()
        self.assertIsNot(cloned, ops[1])
        self.assertTrue(ops[1].key == cloned.key)

        cloned = ops[2].clone()
        self.assertIsNot(cloned, ops[2])
        self.assertTrue(ops[2].key == cloned.key)

        cloned = ops[3].clone()
        self.assertIsNot(cloned, ops[3])
        self.assertTrue(ops[3].method_name == cloned.method_name)

        cloned = ops[4].clone()
        self.assertIsNot(cloned, ops[4])
        self.assertIsNot(ops[4].skrub_impl, cloned.skrub_impl)
        self.assertIsNot(ops[4].skrub_impl.estimator, cloned.skrub_impl.estimator)

        cloned = ops[5].clone()
        self.assertIsNot(cloned, ops[5])
        self.assertEqual(ops[5].func, cloned.func)
        self.assertEqual(ops[5].args, cloned.args)
        self.assertEqual(ops[5].kwargs, cloned.kwargs)

        cloned = ops[6].clone()
        self.assertIsNot(cloned, ops[6])
        self.assertTrue(ops[6].attr_name == cloned.attr_name)

        try:
            op = SearchEvalOp([])
            op.clone()
        except ValueError as e:
            self.assertEqual(str(e), "We should not clone SearchEvalOp objects.")