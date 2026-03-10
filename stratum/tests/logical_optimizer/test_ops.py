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

    def test_op_clone_value_op(self):
        op = ValueOp(1)
        try:
            op.clone()
        except ValueError as e:
            self.assertEqual(str(e), "We should not clone ValueOp objects.")

    def test_op_clone_search_eval_op(self):
        try:
            op = SearchEvalOp([])
            op.clone()
        except ValueError as e:
            self.assertEqual(str(e), "We should not clone SearchEvalOp objects.")

    
    def test_clone_ops(self):
        """Test cloning an Op with skrub_impl."""
        data = skrub.as_data_op(self.df)
        #data_op = data[["x"]].apply(lambda x: x + 1)
        data_op = data.apply(lambda x: x + 1)

        pred = data_op.skb.apply(DummyRegressor(), y=data["y"])
        pred = pred.skb.apply_func(lambda x,a, b: x, 1, b=1)
        choice = skrub.choose_from([pred], name="choice").as_data_op()
        out = choice.empty
        with skrub.config(fast_dataops_convert=True):
            ops = optimize(out)
        print(ops)
        try:
            ops[0].clone()
        except ValueError as e:
            self.assertEqual(str(e), "We should not clone DataSourceOp objects.")
        
        cloned = ops[1].clone()
        self.assertIsNot(cloned, ops[1])
        self.assertTrue(ops[1].args == cloned.args)
        self.assertTrue(ops[1].columns == cloned.columns)

        cloned = ops[2].clone()
        self.assertIsNot(cloned, ops[2])
        self.assertTrue(ops[2].key == cloned.key)

        cloned = ops[3].clone()
        self.assertIsNot(cloned, ops[3])
        self.assertIsNot(ops[3].estimator, cloned.estimator)

        cloned = ops[4].clone()
        self.assertIsNot(cloned, ops[4])
        self.assertEqual(ops[4].func, cloned.func)
        self.assertEqual(ops[4].args, cloned.args)
        self.assertEqual(ops[4].kwargs, cloned.kwargs)

        cloned = ops[5].clone()
        self.assertIsNot(cloned, ops[5])
        self.assertTrue(ops[5].attr_name == cloned.attr_name)


    def test_replace_non_existing_input(self):
        op = Op(outputs=[1], inputs=[2])
        op.name = "test_op"
        try:
            op.replace_input(3, 4)
        except ValueError as e:
            self.assertEqual(str(e), "Input 3 not found in Op.")

    def test_replace_non_existing_output(self):
        op = Op(outputs=[1], inputs=[2])
        op.name = "test_op"
        try:
            op.replace_output(3, 4)
        except ValueError as e:
            self.assertEqual(str(e), "Output 3 not found in Op.")