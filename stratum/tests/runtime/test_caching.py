import unittest
import os
import sys
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
    expected_simple_hashes = {
        "local": [
            14466646976231713574,
            4283753923329093683,
            11672455255761944456,
            1,
            3,
            2,
            17673706173561179344,
            6346118744052152261,
        ],
        "linux": [
            17843118638478979946,
            4283753923329093683,
            11672455255761944456,
            1,
            3,
            2,
            17673706173561179344,
            6346118744052152261,
        ],
        "macos": [
            17237841316323807291,
            4283753923329093683,
            11672455255761944456,
            1,
            3,
            2,
            17673706173561179344,
            6346118744052152261,
        ],
        "windows": [
            9534843511007154554,
            4283753923329093683,
            11672455255761944456,
            1,
            3,
            2,
            17673706173561179344,
            6346118744052152261,
        ],
    }
    expected_hashes = {
        "local": [
            17214955316726503821,
            11824152000386466899,
            18298532774759976535,
            7513694150800269850,
            5537892472318521177,
            168195864670644233,
            1997578848421863092,
            14476433947220053316,
        ],
        "linux": [
            7056806754431583388,
            7639690250793122720,
            17532383718078189923,
            10707031619699354836,
            7435966898669112865,
            8941144976148573683,
            16675763945336090482,
            13801223252098341323,
        ],
        "macos": [
            11800167861632073492,
            13894009875302220469,
            2903296657173264096,
            4207835120194851649,
            11109528315728706675,
            17956785590977498290,
            10919015601046973997,
            12242410082145359458,
        ],
        "windows": [
            6235675172187585043,
            6926154378978508485,
            3072755605188723418,
            18085496009191016749,
            15337519220874548500,
            10614601562615527768,
            15722255280919350770,
            508214583577843344,
        ],
    }

    @classmethod
    def _detect_mode(cls) -> str | None:
        """Detect environment to pick the right expected hash set.

        - \"local\": developer machine at /Users/elias/PycharmProjects/stratum/
        - \"linux\" / \"macos\" / \"windows\": GitHub runners on the respective OS.
        """
        file_path = os.path.abspath(__file__)
        local_root = "/Users/elias/PycharmProjects/stratum/"
        if file_path.startswith(local_root):
            return "local"
        if sys.platform.startswith("linux"):
            return "linux"
        if sys.platform.startswith("darwin"):
            return "macos"
        if sys.platform.startswith(("win32", "cygwin")):
            return "windows"
        return None

    def compare_hashes(self, op, expected_hash, simple = False,):
        hash_val = op.simple_hash() if simple else op.get_hash()
        self.assertEqual(expected_hash, hash_val, f"Hash mismatch for {op}")



    def test_hashes(self):
        file_path = os.path.join(os.path.dirname(__file__), "data.csv")
        mode = self._detect_mode()
        if mode not in self.expected_simple_hashes or mode not in self.expected_hashes:
            self.skipTest(f"No expected hashes defined for mode={mode!r}")
        self.df.to_csv(file_path, index=False)
        data = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        x_vec = X.skb.apply(skrub.TableVectorizer())
        pred = x_vec.skb.apply(DummyRegressor(), y=y)
        pred = optimize(pred)
        ops = list(topological_iterator(pred))

        for i, op in enumerate(ops):
            self.compare_hashes(op, self.expected_simple_hashes[mode][i], simple=True)
        for i, op in enumerate(ops):
            self.compare_hashes(op, self.expected_hashes[mode][i])
            
    def test_search(self):
        SchedulerFlags.stratum_gc = False
        file_path = os.path.join(os.path.dirname(__file__), "data.csv")
        mode = self._detect_mode()
        if mode not in self.expected_simple_hashes or mode not in self.expected_hashes:
            self.skipTest(f"No expected hashes defined for mode={mode!r}")
        self.df.to_csv(file_path, index=False)
        data = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv)
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