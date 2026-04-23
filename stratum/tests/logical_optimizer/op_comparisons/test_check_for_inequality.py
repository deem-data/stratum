import unittest

from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer

import stratum as st
from stratum.optimizer._op_comparison import equals_data_op
import pandas as pd

# dummy function
def pre_process(df):
    return df


class TestCheckForInequality(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_check_for_inequality_apply1(self): 
        data = st.var("data", self.df)
        y1 = data.skb.apply_func(pre_process)
        self.assertFalse(equals_data_op(y1, data))

    def test_check_for_inequality_apply2(self):
        data = st.var("data", self.df)
        y1 = data.skb.apply(StandardScaler())
        y2 = data.skb.apply(TableVectorizer())
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_inequality_apply3(self):
        data = st.var("data", self.df)
        y1 = data.skb.apply(StandardScaler())
        y2 = y1.skb.apply(StandardScaler())
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_inequality_apply4(self):
        data = st.var("data", self.df)
        y1 = data.skb.apply(StandardScaler())
        y2 = data.skb.apply(StandardScaler(), cols=[])
        self.assertFalse(equals_data_op(y1, data))
        self.assertFalse(equals_data_op(y1, y2))

    def test_check_for_inequality_apply5(self):
        data = st.var("data", self.df)
        enc = TableVectorizer()
        enc2 = TableVectorizer(datetime="passthrough")
        y1 = data.skb.apply(enc)
        y2 = data.skb.apply(enc2)
        self.assertFalse(equals_data_op(y1, y2))
    
    def test_check_for_inequality_binary(self):
        t1 = st.as_data_op(1)
        t2 = st.as_data_op(2)
        t3 = t1 + t2
        t4 = t1 - t2
        self.assertFalse(equals_data_op(t3, t4))


if __name__ == '__main__':
    unittest.main()

