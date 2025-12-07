import unittest
import skrub
from stratum.runtime import grid_search
from stratum.logical_optimizer import optimize
from sklearn.model_selection import KFold
from stratum.tests.runtime.runtime_test_utils import RuntimeTest, datetime_pipeline1, datetime_pipeline2
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
        results, preds = grid_search(y, cv=cv, return_predictions=True, print_heavy_hitters=True)

        search = y.skb.make_grid_search(fitted=True)
        results = search.cv_results_
        # preds = cross_val_predict(search.best_learner_, X.skb.preview(), y.skb.preview(), cv=cv)
        print(type(results))
        print(results)


if __name__ == "__main__":
    unittest.main()