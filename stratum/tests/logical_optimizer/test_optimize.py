from stratum.logical_optimizer._optimize import OptConfig, optimize
import stratum as skrub
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

    def test_optimize(self):
        data = skrub.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()

        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2 = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month)
        out = optimize(X2, OptConfig(cse=True))
        self.assertEqual(out[0].skrub_impl, data._skrub_impl)
        self.assertTrue(out[0].children[0] is out[1])
        self.assertTrue(len(out[0].parents) == 0)
        self.assertTrue(len(out[0].parents) == 0)


if __name__ == '__main__':
    unittest.main()
