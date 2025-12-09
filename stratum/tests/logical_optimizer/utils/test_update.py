import unittest

from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

import stratum as skrub
from stratum.logical_optimizer.utils import update_data_op
import pandas as pd

# dummy function
def pre_process(df):
    return df

def pre_process2(df, arg2):
    return df


class TestUpdate(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_update_method_call_op1(self):
        data = skrub.var("data", self.df)
        t1 = data.skb.apply_func(pre_process)
        t2 = data.skb.apply_func(pre_process)
        y2 = t2.skb.apply_func(pre_process)
        assert y2._skrub_impl.args[0] is t2
        update_data_op(y2, t2, t1)
        assert y2._skrub_impl.args[0] is t1

    def test_update_method_call_op2(self):
        data = skrub.as_data_op("aa")
        t1 = skrub.as_data_op("aa")
        t2 = skrub.as_data_op("aa")
        out = data.replace(t1, "bb")
        assert out._skrub_impl.args[0] is t1
        update_data_op(out, t1, t2)
        assert out._skrub_impl.args[0] is t2

    def test_update_apply_op_x(self):
        data = skrub.var("data", self.df)
        x1 = data["x"]
        x2 = data["x"]
        y = data["y"]
        pred = x1.skb.apply(DummyRegressor(), y=y)
        assert pred._skrub_impl.X is x1
        update_data_op(pred, x1, x2)
        assert pred._skrub_impl.X is x2

    def test_update_apply_op_y(self):
        data = skrub.var("data", self.df)
        x = data["x"]
        y1 = data["y"]
        y2 = data["y"]
        pred = x.skb.apply(DummyRegressor(), y=y1)
        assert pred._skrub_impl.y is y1
        update_data_op(pred, y1, y2)
        assert pred._skrub_impl.y is y2

    def test_update_apply_op_cols(self):
        data = skrub.var("data", self.df)
        y = data["y"]
        cols1 = skrub.as_data_op(["x"])
        cols2 = skrub.as_data_op(["x"])
        pred = data.skb.apply(StandardScaler(), y=y, cols=cols1)
        assert pred._skrub_impl.cols is cols1
        try:
            update_data_op(pred, cols1, cols2)
        except Exception as e:
            self.assertEqual("Could not find old DataOp <Value list> during input update for <Apply StandardScaler>", str(e))
        # TODO: This should be true, but it is currently false because the replace for aplly cols is not implemented yet.
        # assert pred._skrub_impl.y is cols2

    def test_update_call_op(self):
        data = skrub.var("data", self.df)
        t1 = data.skb.apply_func(pre_process2, 123)
        t2 = data.skb.apply_func(pre_process2, 123)
        y2 = t2.skb.apply_func(pre_process2, 123)
        assert y2._skrub_impl.args[0] is t2
        update_data_op(y2, t2, t1)
        assert y2._skrub_impl.args[0] is t1

    def test_update_dataop_not_found(self):
        data = skrub.as_data_op(1)
        t1 = skrub.as_data_op(2)
        t2 = skrub.as_data_op(2)
        out = data + t1
        try:
            update_data_op(out, t2, t1)
            self.fail("Expected Exception")
        except Exception as e:
            self.assertEqual("Could not find old DataOp <Value int> during input update for <BinOp: add>", str(e))

    def test_update_binary_op_right(self):
        data = skrub.as_data_op(1)
        t1 = skrub.as_data_op(2)
        t2 = skrub.as_data_op(2)
        out = data + t1
        assert out._skrub_impl.right is t1
        update_data_op(out, t1, t2)
        assert out._skrub_impl.right is t2

    def test_update_binary_op_left(self):
        data = skrub.as_data_op(1)
        t1 = skrub.as_data_op(2)
        t2 = skrub.as_data_op(2)
        out = t1 + data
        assert out._skrub_impl.left is t1
        update_data_op(out, t1, t2)
        assert out._skrub_impl.left is t2

    def test_update_choose_op(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2)
        t3 = skrub.as_data_op(3)
        choice = skrub.choose_from([t1, t2]).as_data_op()
        assert choice._skrub_impl.value.outcomes[0] is t1
        assert choice._skrub_impl.value.outcomes[1] is t2
        update_data_op(choice, t1, t3)
        assert choice._skrub_impl.value.outcomes[0] is t3

    def test_update_call_op_fail(self):
        data = skrub.var("data", self.df)
        t1 = data.skb.apply_func(pre_process2, 123)
        t2 = data.skb.apply_func(pre_process2, 123)
        y2 = t2.skb.apply_func(pre_process2, 123)
        y2._skrub_impl.args = list(y2._skrub_impl.args)
        try:
            update_data_op(y2, t2, t1)
            self.fail("Expected NotImplementedError")
        except NotImplementedError as e:
            self.assertEqual("Non-tuple arguments of method call are not supported yet.", str(e))

    def test_update_getitem_op_fail(self):
        data = skrub.var("data", self.df)
        x = data["x"]
        t = skrub.as_data_op(1)
        try:
            update_data_op(x, t, x)
            self.fail("Expected Exception")
        except Exception as e:
            self.assertEqual("Could not find old DataOp <Value int> during input update for <GetItem 'x'>", str(e))

    def test_update_getattrs_op_fail(self):
        data = skrub.var("data", self.df)
        cols = data.columns
        t = skrub.as_data_op(1)
        try:
            update_data_op(cols, t, cols)
            self.fail("Expected Exception")
        except Exception as e:
            self.assertEqual("Could not find old DataOp <Value int> during input update for <GetAttr 'columns'>", str(e))
    
    def test_update_call_op_fail2(self):
        data = skrub.var("data", self.df)
        t1 = data.skb.apply_func(pre_process2, 123)
        t2 = skrub.as_data_op(1)
        try:
            update_data_op(t1, t2, t1)
            self.fail("Expected Exception")
        except Exception as e:
            self.assertEqual("Could not find old DataOp <Value int> during input update for <Call 'pre_process2'>", str(e))

    def test_update_callmethod_op_fail(self):
        data = skrub.as_data_op("aa")
        t1 = skrub.as_data_op("a")
        t2 = data.replace(t1, "b")
        t3 = skrub.as_data_op(1)
        try:
            update_data_op(t2, t3, t1)
            self.fail("Expected Exception")
        except Exception as e:
            self.assertEqual("Could not find old DataOp <Value int> during input update for <CallMethod 'replace'>", str(e))

    def test_update_choose_op_fail(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2)
        t3 = skrub.as_data_op(3)
        choice = skrub.choose_from([t1, t2]).as_data_op()
        try:
            update_data_op(choice, t3, t3)
        except Exception as e:
            self.assertEqual("Could not find old DataOp <Value int> during input update for <Value Choice>", str(e))

    def test_update_value_fail(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(t1)
        t3 = skrub.as_data_op(3)
        try:
            update_data_op(t2, t3, t3)
        except Exception as e:
            self.assertEqual("Could not find old DataOp <Value int> during input update for <Value DataOp>", str(e))

    def test_update_concat_fail(self):
        df = skrub.as_data_op(self.df)
        df2 = df.skb.concat([df])
        try:
            update_data_op(df2, df, df)
        except Exception as e:
            self.assertEqual("Could not find old DataOp <Value DataFrame> during input update for <Concat: 2 dataframes>", str(e))


if __name__ == '__main__':
    unittest.main()

