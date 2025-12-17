import unittest

from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer

import stratum as skrub
from stratum.logical_optimizer._op_comparison import equals_data_op
import pandas as pd

# dummy function
def pre_process(df):
    return df


class TestCheckForEquivalence(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()

