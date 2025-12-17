import unittest

from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer

import stratum as skrub
from stratum.logical_optimizer._op_comparison import equals_data_op, hash_data_op
import pandas as pd

# dummy function
def pre_process(df):
    return df


class TestHashEquivalence(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

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


if __name__ == '__main__':
    unittest.main()

