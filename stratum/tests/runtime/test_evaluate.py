import unittest
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import skrub
from skrub._data_ops._data_ops import DataOp

from stratum.logical_optimizer import optimize
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import logging

from stratum.runtime.runtime import evaluate
from stratum.tests.runtime.runtime_test_utils import RuntimeTest, datetime_pipeline1
import numpy as np


logging.basicConfig(level=logging.INFO) # switch to DEBUG for showing the DataOps plan during optimization


class EvaluateTest(RuntimeTest):
    def test_evaluate_datetime_pipe(self):
        data = skrub.as_data_op(self.df)
        x = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        pred = datetime_pipeline1(x, y)
        pred_opt = optimize(pred)

        self.compare_evaluate(pred_opt)

    def test_evaluate(self):
        # generate data using sklearn
        n_features = 20
        X, y = make_regression(n_samples=1000, n_features=n_features, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
        df["y"] = y

        data = skrub.as_data_op(df)
        x = data.drop("y", axis=1).skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        
        x = x.assign(new_x0 = x["x0"] + x["x1"] + x["x2"] + x["x3"] + x["x4"] + x["x5"] + x["x6"] + x["x7"] + x["x8"] + x["x9"])
        x = x.assign(new_x1 = x["x2"] * x["x3"])
        x = x.assign(new_x2 = x["x4"] / x["x5"])
        x = x.drop(["x0", "x1"], axis=1)
        x_scaled = x.skb.apply(StandardScaler())
        pred = x_scaled.skb.apply(RandomForestRegressor(random_state=42), y=y)
        pred.skb.draw_graph().open()
        pred_opt = optimize(pred)
        pred_opt.skb.draw_graph().open()
        self.compare_evaluate(pred_opt)

if __name__ == "__main__":
    unittest.main()