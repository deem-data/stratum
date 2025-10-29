from sklearn.ensemble import RandomForestRegressor
from stratum.logical_optimizer import optimize
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

    def test_cse(self):
        data = skrub.var("data", self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        t1 = X.skb.apply_func(pre_process)
        y1 = t1.skb.apply(RandomForestRegressor(random_state=42), y=y)

        # pipeline 2
        t2 = X.skb.apply_func(pre_process)
        y2 = t2.skb.apply(RandomForestRegressor(random_state=123), y=y)

        y = skrub.choose_from({"pipeline 1": y1, "pipeline 2": y2}).as_data_op()
        y = optimize(y)

    def test_cse2(self):
        data = skrub.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        X1A = X.assign(datetime=X["datetime"].apply(lambda a: a).apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X1B = X1A.assign(
            year=X1A["datetime"].dt.year,
            month=X1A["datetime"].dt.month)
        X1B = optimize(X1B)

    def test_cse4(self):
        data = skrub.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        X1A = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X1B = X1A.assign(
            year=X1A["datetime"].dt.year,
            month=X1A["datetime"].dt.month,
            dayofweek=X1A["datetime"].dt.dayofweek,
            hour=X1A["datetime"].dt.hour)
        X1B = X1B.drop(["datetime"], axis=1)
        y1 = X1B.skb.apply(RandomForestRegressor(random_state=42), y=y)
        y = optimize(y1)

    def test_cse5(self):
        data = skrub.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        X1A = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X1B = X1A.assign(
            year=X1A["datetime"].dt.year,
            month=X1A["datetime"].dt.month,
            dayofweek=X1A["datetime"].dt.dayofweek,
            hour=X1A["datetime"].dt.hour)
        X1B = X1B.drop(["datetime"], axis=1)
        y1 = X1B.skb.apply(RandomForestRegressor(random_state=42), y=y)

        X2 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2B = X2.assign(
            year=X2["datetime"].dt.year,
            month=X2["datetime"].dt.month,
            dayofweek=X2["datetime"].dt.dayofweek,
            hour=X2["datetime"].dt.hour)
        X2B = X2B.drop(["datetime"], axis=1)
        y2 = X2B.skb.apply(RandomForestRegressor(random_state=123), y=y)
        y = skrub.choose_from({"pipeline 1": y1, "pipeline 2": y2}).as_data_op()
        y = optimize(y)

    def test_cse6(self):
        data = skrub.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        X1A = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X1B = X1A.assign(
            year=X1A["datetime"].dt.year,
            month=X1A["datetime"].dt.month,
            dayofweek=X1A["datetime"].dt.dayofweek)
        X1B = X1B.drop(["datetime"], axis=1)
        y1 = X1B.skb.apply(RandomForestRegressor(random_state=42), y=y)

        X2 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2B = X2.assign(
            month=X2["datetime"].dt.month,
            dayofweek=X2["datetime"].dt.dayofweek,
            hour=X2["datetime"].dt.hour)
        X2B = X2B.drop(["datetime"], axis=1)
        y2 = X2B.skb.apply(RandomForestRegressor(random_state=123), y=y)
        y = skrub.choose_from({"pipeline 1": y1, "pipeline 2": y2}).as_data_op()
        y = optimize(y)



if __name__ == '__main__':
    unittest.main()
