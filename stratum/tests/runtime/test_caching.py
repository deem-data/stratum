import unittest
import os
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
import stratum as skrub
from stratum.logical_optimizer._op_utils import topological_iterator
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
        expected_simple_hashes = [4857146406207644845,4283753923329093683,11672455255761944456,1,2,3,17673706173561179344,6346118744052152261]
        expected_hashes = [18362580156690481027,14547820100968022052,6083659439458326793,7241456746530273154,12567261729024511446,11467303812476240355,3547814939573418220,9737559621915143817]
        for i, op in enumerate(ops):
            self.compare_hashes(op, expected_simple_hashes[i], simple=True)
        for i, op in enumerate(ops):
            self.compare_hashes(op, expected_hashes[i])
            
    def test_search(self):
        data = skrub.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        x_vec = X.skb.apply(skrub.TableVectorizer())
        pred = x_vec.skb.apply(DummyRegressor(), y=y)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        with skrub.config(scheduler=True, stats=20, caching=True):
            search = pred.skb.make_grid_search(cv=cv, fitted=True,scoring="neg_mean_squared_error")

if __name__ == "__main__":
    unittest.main()