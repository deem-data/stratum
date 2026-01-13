import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skrub._data_ops._data_ops import DataOp
from stratum.runtime._scheduler import evaluate
import stratum as skrub
from sklearn.dummy import DummyRegressor


def datetime_pipeline1(x: DataOp, y: DataOp) -> DataOp:
    x1 = x.assign(datetime=x["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
    x2 = x1.assign(
        year=x1["datetime"].dt.year,
        month=x1["datetime"].dt.month)
    x3 = x2.drop(["datetime"], axis=1)
    pred = x3.skb.apply(RandomForestRegressor(random_state=42), y=y)
    return pred


def datetime_pipeline2(x: DataOp, y: DataOp) -> DataOp:
    x2 = x.assign(datetime=x["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
    x3 = x2.assign(
        year=x2["datetime"].dt.year,
        month=x2["datetime"].dt.month,
        dayofweek=x2["datetime"].dt.dayofweek,
        hour=x2["datetime"].dt.hour)
    x4 = x3.drop(["datetime"], axis=1)
    pred = x4.skb.apply(RandomForestRegressor(random_state=123), y=y)
    return pred

def simple_pipeline() -> DataOp:
    data = {"x": np.linspace(0, 10, 100), "y": np.linspace(0, 10, 100) % 10}
    data = pd.DataFrame(data)
    data = skrub.as_data_op(data)
    x = data[["x"]].skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    x = x + 33
    x = x.assign(z=x["x"] + 1)
    model = DummyRegressor()
    pred = x.skb.apply(model, y=y)
    return pred

class RuntimeTest(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "datetime": [
                "2025-11-01 10:00:00",
                "2025-11-02 15:30:00",
                "2025-11-03 09:45:00",
                "2025-11-04 12:00:00",
                "2025-11-05 14:30:00",
                "2025-11-06 16:45:00",
                "2025-11-07 18:00:00",
                "2025-11-08 20:30:00",
                "2025-11-09 22:45:00",
                "2025-11-10 01:00:00",
            ]
        })
        self.seed = 42
        self.test_size = 0.5

    def compare_evaluate(self, pred_opt: DataOp):
        preds = evaluate(pred_opt, seed=self.seed, test_size=self.test_size)

        splits = pred_opt.skb.train_test_split(random_state=self.seed, test_size=self.test_size)
        learner = pred_opt.skb.make_learner()
        learner.fit(splits["train"])
        preds_skrub = learner.predict(splits["test"])
        np.testing.assert_array_equal(preds_skrub, preds)

