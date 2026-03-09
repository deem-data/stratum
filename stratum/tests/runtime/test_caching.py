import unittest
import os
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
import stratum as skrub
from stratum.logical_optimizer._op_utils import topological_iterator
from stratum.runtime._scheduler import SchedulerFlags
from stratum.tests.runtime.runtime_test_utils import RuntimeTest
import logging
from stratum.logical_optimizer._optimize import optimize
import pandas as pd
logging.basicConfig(level=logging.DEBUG)


class SearchTest(RuntimeTest):
    def compare_hashes(self, op, expected_hash, simple = False,):
        hash_val = op.simple_hash() if simple else op.get_hash()
        self.assertEqual(expected_hash, hash_val, f"Hash mismatch for {op}")



    def test_hashes(self):
        file_path = os.path.join(os.path.dirname(__file__), "data.csv")
        self.df.to_csv(file_path, index=False)
        data = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        x_vec = X.skb.apply(skrub.TableVectorizer())
        pred = x_vec.skb.apply(DummyRegressor(), y=y)
        pred = optimize(pred)
        ops = list(topological_iterator(pred))
        print([op.simple_hash() for op in ops])
        print([op.get_hash() for op in ops])
        expected_simple_hashes = [14466646976231713574, 4283753923329093683, 11672455255761944456, 1, 3, 2, 17673706173561179344, 6346118744052152261]
        expected_hashes = [17214955316726503821, 11824152000386466899, 18298532774759976535, 7513694150800269850, 5537892472318521177, 168195864670644233, 1997578848421863092, 14476433947220053316]
        for i, op in enumerate(ops):
            self.compare_hashes(op, expected_simple_hashes[i], simple=True)
        for i, op in enumerate(ops):
            self.compare_hashes(op, expected_hashes[i])
            
    def test_search(self):
        SchedulerFlags.stratum_gc = False
        data = skrub.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        x_vec = X.skb.apply(skrub.TableVectorizer())
        pred = x_vec.skb.apply(DummyRegressor(), y=y)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        with skrub.config(scheduler=True, stats=20, caching=True):
            search = pred.skb.make_grid_search(cv=cv, fitted=True,scoring="neg_mean_squared_error")
        SchedulerFlags.stratum_gc = True

if __name__ == "__main__":
    unittest.main()