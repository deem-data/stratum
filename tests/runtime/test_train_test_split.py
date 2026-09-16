import unittest
import stratum as st
from tests.runtime.runtime_test_utils import simple_pipeline

class SplitOpTest(unittest.TestCase):
    def test_train_test_split(self):
        pipeline = simple_pipeline()
        with st.config(scheduler=True):
            search = pipeline.skb.make_grid_search(scoring="neg_mean_squared_error")
            print(search.results_)


if __name__ == '__main__':
    unittest.main()
