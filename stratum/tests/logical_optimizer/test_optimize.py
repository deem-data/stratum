from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer._optimize import OptConfig, optimize
import stratum as st
import pandas as pd
import unittest

# dummy function
def pre_process(df):
    return df

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "datetime": [
                "2025-11-01 10:00:00",
                "2025-11-02 15:30:00",
                "2025-11-03 09:45:00"
            ]
        })

    def test_optimize1(self):
        data = st.var("data", self.df).skb.subsample(3)
        X = data[["x", "datetime"]].skb.mark_as_X()

        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2 = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month)
        out = list(topological_iterator(optimize(X2, OptConfig(cse=True))))
        self.assertTrue(out[0].outputs[0] is out[1])
        self.assertTrue(len(out[0].inputs) == 0)

    def test_optimize2(self):
        data = st.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()

        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2 = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month)
        config = OptConfig(cse=False, algebraic_rewrites=False, numeric_ops=False, dataframe_ops=False, unroll_choices=False)
        out = list(topological_iterator(optimize(X2, config)))
        self.assertEqual(len(out), 10)
        
    def test_more_ops(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2 = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month)
        out = optimize(X2, OptConfig(cse=True))




if __name__ == '__main__':
    unittest.main()
