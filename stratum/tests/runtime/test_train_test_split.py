import unittest
import stratum as st
from stratum.tests.runtime.runtime_test_utils import simple_pipeline

class SplitOpTest(unittest.TestCase):
    def test_train_test_split(self):
        pipeline = simple_pipeline()
        with st.config(scheduler=True):
            search = pipeline.skb.make_grid_search()
            print(search.results_)


if __name__ == '__main__':
    unittest.main()
