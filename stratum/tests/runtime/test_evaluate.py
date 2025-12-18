import unittest
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import skrub
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from stratum.runtime._scheduler import Scheduler, evaluate
from stratum.tests.runtime.runtime_test_utils import RuntimeTest, datetime_pipeline1

import logging
logging.basicConfig(level=logging.INFO)


class EvaluateTest(RuntimeTest):
    def test_evaluate_datetime_pipe(self):
        data = skrub.as_data_op(self.df)
        x = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        pred = datetime_pipeline1(x, y)
        self.compare_evaluate(pred)
    

    def test_evaluate_with_choice(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2)
        t3 = skrub.choose_from([t1, t2]).as_data_op()
        t4 = t3 + 5
        t5 = t4 - 3
        out = evaluate(t5, seed=self.seed, test_size=self.test_size)


    def test_evaluate_with_choice2(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2.5)
        t3 = skrub.as_data_op(3)
        t4 = skrub.as_data_op(4.5)
        t5 = skrub.choose_from([t1, t2]).as_data_op()
        t6 = skrub.choose_from([t3, t4]).as_data_op()
        t7 = (t5 + 4) * 2
        t8 = (t6 + 5) * 3
        t9 = skrub.choose_from([t7, t8]).as_data_op()
        t10 = t9 + 1
        # FIXME
        # out = evaluate(t10, seed=self.seed, test_size=self.test_size)

    def test_evaluate_with_choice3(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2.5)
        t5 = skrub.choose_from([t1, t2]).as_data_op()
        t6 = t5 + 5
        t7 = t6 * 2
        t8 = t6 / 3
        t9 = skrub.choose_from([t7, t8]).as_data_op()
        t10 = t9 + 1
        out = evaluate(t10, seed=self.seed, test_size=self.test_size)


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
        self.compare_evaluate(pred)

if __name__ == "__main__":
    unittest.main()