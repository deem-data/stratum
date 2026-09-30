import unittest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import skrub
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from stratum._api import evaluate
from tests.runtime.runtime_test_utils import RuntimeTest, datetime_pipeline1
import stratum
import logging
from contextlib import redirect_stdout
from io import StringIO
logging.basicConfig(level=logging.INFO)


class EvaluateTest(RuntimeTest):
    def test_evaluate_with_stats(self):
        data = skrub.as_data_op(self.df)
        x = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        pred = x.skb.apply(DummyRegressor(), y=y)

        with redirect_stdout(StringIO()) as stdout, stratum.config(stats=True, stats_top_k=0):
            evaluate(pred)

        report = stdout.getvalue()
        self.assertIn("Run timing (seconds", report)
        self.assertIn("Optimization:", report)
        self.assertIn("Other scheduler work:", report)
        self.assertIn("Unshown operators:", report)

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
        evaluate(t5, seed=self.seed, test_size=self.test_size)


    @unittest.skip("FIXME")
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
        
        evaluate(t10, seed=self.seed, test_size=self.test_size)

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
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0]["vals"], 13)
        self.assertEqual(out[1]["vals"], 3)
        self.assertEqual(out[2]["vals"], 3.5)
        self.assertEqual(out[3]["vals"], 16)


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


class TwoAxisIndexerTest(RuntimeTest):
    """A `.loc`/`.iloc` that indexes rows *and* columns in one call.

    The row indexer is a DataOp nested in the key's tuple; leaving it unbound used
    to orphan the whole mask sub-DAG, so the plan raised "op ... should not exist in
    the DAG" while compiling -- before any data was touched, and only on the
    scheduler path. These pin the values too, not just that a plan builds.
    """

    def setUp(self):
        super().setUp()
        self.side = pd.DataFrame({
            "gid": [1, 2, 3] * 10,
            "category": list("abc") * 10,
            "w": np.arange(30.0),
        })

    def _pipeline(self, select):
        data = skrub.as_data_op(self.df)
        side = skrub.var("side", self.side)
        x = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        return x.assign(n=select(side)).skb.apply(DummyRegressor(), y=y)

    def test_row_mask_and_column_list(self):
        self.compare_evaluate(self._pipeline(
            lambda s: s.loc[s["category"] == "a", ["gid"]]["gid"].nunique()))

    def test_row_mask_and_single_column(self):
        self.compare_evaluate(self._pipeline(
            lambda s: s.loc[s["category"] == "a", "w"].sum()))

    def test_compound_row_mask(self):
        self.compare_evaluate(self._pipeline(
            lambda s: s.loc[(s["category"] == "a") & (s["w"] > 3), ["gid"]]["gid"].nunique()))

    def test_row_mask_shared_by_two_indexers(self):
        # The same mask feeds two `.loc`s, so it is bound twice and CSE collapses it.
        self.compare_evaluate(self._pipeline(
            lambda s: s.loc[s["category"] == "a", ["gid"]]["gid"].nunique()
                    + s.loc[s["category"] == "a", ["w"]]["w"].sum()))

    def test_positional_two_axis_iloc(self):
        self.compare_evaluate(self._pipeline(lambda s: s.iloc[0:9, [0]]["gid"].nunique()))

    def test_scheduler_grid_search_scores_like_skrub(self):
        # The reported symptom: the plan compiled fine on the default engine and
        # raised only under `config(scheduler=True)`, at scoring time.
        pipeline = self._pipeline(
            lambda s: s.loc[s["category"] == "a", ["gid"]]["gid"].nunique())
        ours = stratum._api.grid_search(pipeline, cv=KFold(n_splits=2), scoring="r2")
        theirs = pipeline.skb.make_grid_search(
            fitted=True, cv=KFold(n_splits=2), scoring="r2")
        np.testing.assert_allclose(theirs.results_["mean_test_score"],
                                   ours.results_["scores"], rtol=1e-9)

if __name__ == "__main__":
    unittest.main()
