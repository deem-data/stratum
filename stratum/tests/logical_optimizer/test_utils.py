import unittest

from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer

import stratum as skrub
from stratum.logical_optimizer.utils import equals_data_op, hash_data_op, update_data_op
import pandas as pd

# dummy function
def pre_process(df):
    return df

def pre_process2(df, arg2):
    return df

class LogicalOptimizerUtilsTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_check_for_equivalence1(self):
        data = skrub.var("data", self.df)
        y1 = data.skb.apply_func(pre_process)
        y2 = data.skb.apply_func(pre_process)
        self.assertTrue(equals_data_op(y1, y2))

    def test_check_for_equivalence2(self):
        data = skrub.var("data", self.df)
        y1 = data.skb.apply_func(pre_process)
        y2 = data.skb.apply_func(lambda a: a)
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_equivalence3(self):
        data = skrub.var("data", self.df)
        t1 = data.skb.apply_func(pre_process)
        t2 = data.skb.apply_func(pre_process)
        y1 = t1 + 1
        y2 = t2 + 1
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_equivalence4(self):
        data = skrub.var("data", self.df)
        y1 = data["x"]
        y2 = data["x"]
        self.assertTrue(equals_data_op(y1, y2))

    def test_check_for_equivalence5(self):
        data = skrub.var("data", self.df)
        y1 = data["x"]
        y2 = data["y"]
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_equivalence6(self):
        data = skrub.var("data", self.df)
        data2 = skrub.var("data2", self.df)
        y1 = data["x"]
        y2 = data2["x"]
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_equivalence7(self):
        data = skrub.var("data", self.df)
        x = data["x"]
        y1 = x.apply(pre_process)
        y2 = x.apply(pre_process)
        self.assertTrue(equals_data_op(y1, y2))

    def test_check_for_equivalence8(self):
        data = skrub.var("data", self.df)
        x = data["x"]
        y1 = x.apply(pre_process)
        y2 = x.abs()
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_equivalence9(self):
        data = skrub.var("data", self.df)
        x = data["x"]
        y1 = x.apply(lambda a: a)
        y2 = x.apply(lambda a: a)
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_equivalence10(self):
        data = skrub.var("data", self.df)
        y1 = data.columns
        y2 = data.columns
        self.assertTrue(equals_data_op(y1, y2))

    def test_check_for_equivalence11(self):
        data = skrub.var("data", self.df)
        y1 = data.columns
        y2 = data.values
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_equivalence12(self):
        data = skrub.var("data", self.df)
        enc = StandardScaler()
        y1 = data.skb.apply(enc)
        y2 = data.skb.apply(enc)
        self.assertTrue(equals_data_op(y1, y2))

    def test_check_for_equivalence13(self):
        data = skrub.var("data", self.df)
        enc = StandardScaler()
        enc2 = StandardScaler()
        y1 = data.skb.apply(enc)
        y2 = data.skb.apply(enc2)
        self.assertTrue(equals_data_op(y1, y2))

    def test_check_for_equivalence14(self):
        data = skrub.var("data", self.df)
        enc = StandardScaler()
        enc2 = StandardScaler()
        y1 = data.skb.apply(enc, cols=["x"])
        y2 = data.skb.apply(enc2, cols=["x"])
        self.assertTrue(equals_data_op(y1, y2))

    def test_check_for_equivalence15(self):
        data = skrub.var("data", self.df)
        enc = StandardScaler()
        enc2 = StandardScaler()
        y1 = data.skb.apply(enc, cols=[])
        y2 = data.skb.apply(enc2, cols=[])
        self.assertTrue(equals_data_op(y1, y2))

    def test_check_for_equivalence16(self):
        data = skrub.var("data", self.df)
        enc = StandardScaler(with_mean=False)
        enc2 = StandardScaler(with_mean=True)
        y1 = data.skb.apply(enc)
        y2 = data.skb.apply(enc2)
        self.assertFalse(equals_data_op(y1, y2))
    
    def test_check_for_equivalence17(self):
        data = skrub.var("data", self.df)
        enc = TableVectorizer()
        enc2 = TableVectorizer()
        y1 = data.skb.apply(enc)
        y2 = data.skb.apply(enc2)
        self.assertTrue(equals_data_op(y1, y2))

    def test_check_for_inequality_apply1(self):
        data = skrub.var("data", self.df)
        y1 = data.skb.apply_func(pre_process)
        self.assertFalse(equals_data_op(y1, data))

    def test_check_for_inequality_apply2(self):
        data = skrub.var("data", self.df)
        y1 = data.skb.apply(StandardScaler())
        y2 = data.skb.apply(TableVectorizer())
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_inequality_apply3(self):
        data = skrub.var("data", self.df)
        y1 = data.skb.apply(StandardScaler())
        y2 = y1.skb.apply(StandardScaler())
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_inequality_apply4(self):
        data = skrub.var("data", self.df)
        y1 = data.skb.apply(StandardScaler())
        y2 = data.skb.apply(StandardScaler(), cols=[])
        self.assertFalse(equals_data_op(y1, data))
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_inequality_apply5(self):
        data = skrub.var("data", self.df)
        enc = TableVectorizer()
        enc2 = TableVectorizer(datetime="passthrough")
        y1 = data.skb.apply(enc)
        y2 = data.skb.apply(enc2)
        self.assertFalse(equals_data_op(y1, y2))


    def assert_hash_consistency(self, op1, op2):
        """Helper: assert that equals -> equal hash."""
        eq = equals_data_op(op1, op2)
        h1, h2 = hash_data_op(op1), hash_data_op(op2)
        if eq:
            self.assertEqual(
                h1, h2,
                f"Equal DataOps must have equal hashes:\n{op1}\n{op2}"
            )
        else:
            # Not strictly required, but helps catch collisions in practice
            self.assertNotEqual(
                h1, h2,
                f"Unequal DataOps should not have equal hashes:\n{op1}\n{op2}"
            )


    def test_hash_equivalence1(self):
        data = skrub.var("data", self.df)
        y1 = data.skb.apply_func(pre_process)
        y2 = data.skb.apply_func(pre_process)
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence2(self):
        data = skrub.var("data", self.df)
        y1 = data.skb.apply_func(pre_process)
        y2 = data.skb.apply_func(lambda a: a)
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence3(self):
        data = skrub.var("data", self.df)
        t1 = data.skb.apply_func(pre_process)
        t2 = data.skb.apply_func(pre_process)
        y1 = t1 + 1
        y2 = t2 + 1
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence4(self):
        data = skrub.var("data", self.df)
        y1 = data["x"]
        y2 = data["x"]
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence5(self):
        data = skrub.var("data", self.df)
        y1 = data["x"]
        y2 = data["y"]
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence6(self):
        data = skrub.var("data", self.df)
        data2 = skrub.var("data2", self.df)
        y1 = data["x"]
        y2 = data2["x"]
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence7a(self):
        data = skrub.var("data", self.df)
        x = data["x"]
        y1 = x.apply(pre_process)
        y2 = x.apply(pre_process)
        self.assert_hash_consistency(y1, y2)

    def test_hash_equivalence7b(self):
        data = skrub.var("data", self.df)
        y1 = data.drop(["x"], axis=1)
        y2 = data.drop(["x"], axis=1)
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence8(self):
        data = skrub.var("data", self.df)
        x = data["x"]
        y1 = x.apply(pre_process)
        y2 = x.abs()
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence9(self):
        data = skrub.var("data", self.df)
        x = data["x"]
        y1 = x.apply(lambda a: a)
        y2 = x.apply(lambda a: a)
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence10(self):
        data = skrub.var("data", self.df)
        y1 = data.columns
        y2 = data.columns
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence11(self):
        data = skrub.var("data", self.df)
        y1 = data.columns
        y2 = data.values
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence12(self):
        data = skrub.var("data", self.df)
        enc = StandardScaler()
        y1 = data.skb.apply(enc)
        y2 = data.skb.apply(enc)
        self.assert_hash_consistency(y1, y2)


    def test_hash_equivalence13(self):
        data = skrub.var("data", self.df)
        enc = StandardScaler()
        enc2 = StandardScaler()
        y1 = data.skb.apply(enc)
        y2 = data.skb.apply(enc2)
        self.assert_hash_consistency(y1, y2)

    def test_hash_equivalence14(self):
        data = skrub.var("data", self.df)
        enc = StandardScaler()
        enc2 = StandardScaler()
        y1 = data.skb.apply(enc, cols=["x"])
        y2 = data.skb.apply(enc2, cols=["x"])
        self.assert_hash_consistency(y1, y2)

    def test_hash_equivalence15(self):
        data = skrub.var("data", self.df)
        enc = TableVectorizer()
        enc2 = TableVectorizer()
        y1 = data.skb.apply(enc)
        y2 = data.skb.apply(enc2)
        self.assert_hash_consistency(y1, y2)

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

    def test_update_call_op(self):
        data = skrub.var("data", self.df)
        t1 = data.skb.apply_func(pre_process2, 123)
        t2 = data.skb.apply_func(pre_process2, 123)
        y2 = t2.skb.apply_func(pre_process2, 123)
        assert y2._skrub_impl.args[0] is t2
        update_data_op(y2, t2, t1)
        assert y2._skrub_impl.args[0] is t1

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


if __name__ == '__main__':
    unittest.main()