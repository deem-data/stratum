from sklearn.dummy import DummyRegressor
from stratum.runtime import grid_search
from stratum.logical_optimizer import optimize
from sklearn.model_selection import KFold
from stratum.tests.runtime.runtime_test_utils import RuntimeTest, datetime_pipeline1, datetime_pipeline2
from contextlib import redirect_stdout
from io import StringIO
import time
import unittest
import numpy as np
import skrub
import logging

logging.basicConfig(level=logging.INFO)

class SearchTest(RuntimeTest):
    def test_search(self):
        data = skrub.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        y1 = datetime_pipeline1(X, y)
        y2 = datetime_pipeline2(X, y)
        y = skrub.choose_from({"pipeline 1": y1, "pipeline 2": y2}).as_data_op()
        y = optimize(y)

        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        results, preds = grid_search(y, cv=cv, scoring="neg_mean_squared_error", return_predictions=True)

        search = y.skb.make_grid_search(cv=cv, fitted=True,scoring="neg_mean_squared_error")
        assert(np.allclose(search.results_["mean_test_score"], results["scores"]))



    def test_search_with_no_X(self):
        start = skrub.as_data_op(True)
        end = start.skb.apply_func(lambda a: a).skb.mark_as_y()

        try:
            grid_search(end, return_predictions=True, show_stats=True)
            self.fail("Expected RuntimeError")
        except RuntimeError as e:
            self.assertEqual("X and y nodes not found in the DAG",str(e))

    def test_search_with_no_y(self):
        start = skrub.as_data_op(True)
        end = start.skb.apply_func(lambda a: a).skb.mark_as_X()

        try:
            grid_search(end, return_predictions=True, show_stats=True)
            self.fail("Expected RuntimeError")
        except RuntimeError as e:
            self.assertEqual("X and y nodes not found in the DAG",str(e))


    def test_search_choice_not_at_the_end(self):
        data = skrub.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        y = y + skrub.choose_from([0,1]).as_data_op()
        pred = X.skb.apply(DummyRegressor(), y=y)
        try:
            grid_search(pred)
            self.fail("Expected NotImplementedError")
        except NotImplementedError as e:
            self.assertEqual("Choices with non-DataOp outcomes are not supported yet.", str(e))

    def test_search_error_during_dataop_processing(self):
        data = skrub.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        y = y.skb.apply_func(lambda a, m: (a, print(m))[0] if m != 'predict' else int("grr"), skrub.eval_mode())
        pred = X.skb.apply(DummyRegressor(), y=y)
        try:
            grid_search(pred)
            self.fail("Expected RunTimeError")
        except RuntimeError as e:
            self.assertTrue(e.args[0].startswith("Error processing implementation '<Call '<lambda>'>' "))



    def test_search_with_stats(self):
        data = skrub.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        X2 = X.skb.apply_func(lambda a: (a, time.sleep(0.01))[0])
        pred = X2.skb.apply(DummyRegressor(), y=y)
        # capture stdout
        with redirect_stdout(StringIO()) as stdout:
            grid_search(pred, return_predictions=False, show_stats=True)
            out = stdout.getvalue()
        out = out.split("\n")
        self.assertIn("Heavy hitters", out[2])
        self.assertIn("<Call '<lambda>'>", out[4])
        assert(out[4].split(" ")[-1] == "10")
        self.assertIn("<Apply DummyRegressor>", out[5])
        assert(out[5].split(" ")[-1] == "10")


if __name__ == "__main__":
    unittest.main()