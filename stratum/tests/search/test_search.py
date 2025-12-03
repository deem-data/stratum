import unittest
import skrub
from stratum.search import grid_search
from stratum.logical_optimizer import optimize
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO) # switch to DEBUG for showing the DataOps plan during optimization

class SearchTest(unittest.TestCase):
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


    def test_search(self):
        data = skrub.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        X1A = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X1B = X1A.assign(
            year=X1A["datetime"].dt.year,
            month=X1A["datetime"].dt.month)
        X1B = X1B.drop(["datetime"], axis=1)
        y1 = X1B.skb.apply(RandomForestRegressor(random_state=42), y=y)

        X2 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2B = X2.assign(
            month=X2["datetime"].dt.year,
            dayofweek=X2["datetime"].dt.month)
        X2B = X2B.drop(["datetime"], axis=1)
        y2 = X2B.skb.apply(RandomForestRegressor(random_state=123), y=y)
        y = skrub.choose_from({"pipeline 1": y1, "pipeline 2": y2}).as_data_op()
        y = optimize(y)
        results, preds = grid_search(y, return_predictions=True)
        # results_skrub = y.skb.make_grid_search(fitted=True).results_
        print(type(results))
        print(results)

if __name__ == "__main__":
    unittest.main()