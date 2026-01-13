from time import perf_counter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, check_cv
from sklearn.metrics._scorer import _Scorer
from skrub._data_ops._data_ops import EvalMode
from skrub._data_ops import DataOp
from types import SimpleNamespace
import io
from contextlib import redirect_stdout, redirect_stderr
from stratum._config import FLAGS
from stratum.logical_optimizer._dataframe_ops import SplitOp
from stratum.logical_optimizer._optimize import optimize,OptConfig
from stratum.logical_optimizer._ops import CallOp, ChoiceOp, GetItemOp, ImplOp, MethodCallOp, Op, ValueOp

import pandas as pd
import logging
logger = logging.getLogger(__name__)


def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False, show_stats=False):
    """Perform grid search with cross-validation on a DataOp DAG."""
    # TODO maybe remove the option to show stats here and make args similar to scikit-learn's grid_search
    show_stats = FLAGS.stats or show_stats
    ops_ordered = optimize(dag, OptConfig(cse=True))
    sched = Scheduler(ops_ordered, show_stats)

    preds = sched.grid_search(cv, scoring, return_predictions)

    # Heavy hitters
    if show_stats:
        table = pd.DataFrame(sched.timings, columns=["Op", "time"])
        table = table.groupby("Op").aggregate(["sum", "count"])
        table.columns = ["Time", "Count"]
        table = table.reset_index().sort_values(by="Time", ascending=False)
        print("\n" + "=" * 80)
        print(f"Heavy hitters (sorted by time spent in DataOp evaluation):")
        print(table.head(20).to_string(index=False))
        print("=" * 80 + "\n")
    return (sched,preds) if return_predictions else sched

def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2, cse: bool = False):
    """Evaluate a DataOp DAG with train/test split."""
    ops_ordered = optimize(dag, OptConfig(cse=cse))
    return Scheduler(ops_ordered).evaluate(seed, test_size)

class Scheduler:
    """Scheduler for executing DataOpDAGs in topological order."""
    
    def __init__(self, ops_ordered: list[Op], print_heavy_hitters=False):
        """Initialize scheduler with a data operations DAG."""
        self.ops_ordered = ops_ordered
        self.mode = "fit_transform"
        self.env = {}
        self.flagged_for_recomputation = []
        self.pos_split_op = None
        self.timings = [] if print_heavy_hitters else None
        self.results_ = None

    def evaluate(self, seed: int = 42, test_size = 0.2):
        """Evaluate the pipeline with a train/test split and return predictions."""
        try:
            split_op = self.compute_xy()
        except RuntimeError as e:
            if "X and y nodes not found in the DAG" in str(e):
                logger.warning("X and y nodes not found in the DAG, returning the last node")
                return self.ops_ordered[-1].intermediate
            else:
                raise e
        
        train_index, test_index = train_test_split(range(len(split_op.inputs[0].intermediate)), test_size=test_size, random_state=seed)
        split_op.indices = train_index
        self.compute(self.pos_split_op)
        split_op.indices = test_index
        pred = self.compute(self.pos_split_op, mode="predict")
        return pred["vals"][0]


    def grid_search(self, cv=None, scoring=None, return_predictions=False):
        """Perform grid search with cross-validation on the DataOp DAG in a sequential top-down manner."""
        # default to scikit-learn's CV
        cv = check_cv(cv)

        # get scoring function
        if type(scoring) == str:
            coeff = -1 if scoring.startswith("neg_") else 1
            scoring_func = lambda test, pred: mean_squared_error(test, pred)*coeff
        elif type(scoring) == _Scorer:
            scoring_func = scoring._score_func
        else:
            scoring_func = mean_squared_error

        # start with computing till X and y node
        logger.debug("\n", "="*100, "\n", "Starting grid search", "\n","="*100, "\n")
        split_op = self.compute_xy()
        results, predictions = [], []

        logger.debug("\n", "="*100, "\n", "XY computed", "\n","="*100, "\n")
        #TODO we can parallelize over the folds
        for i, (train_index, test_index) in enumerate(cv.split(split_op.inputs[0].intermediate)):
            logger.debug(f"CV Fold Nr. {i+1}")

            split_op.indices = train_index

            # fit and predict the pipeline
            split_op.indices = train_index
            self.compute(self.pos_split_op)
            logger.debug("\n", "="*100, "\n", "Training done for fold", i+1, "\n","="*100, "\n")
            split_op.indices = test_index
            df = self.compute(self.pos_split_op, mode="predict")
            logger.debug("\n", "="*100, "\n", "Predicting done for fold", i+1, "\n","="*100, "\n")
            if return_predictions:
                predictions.append(df.copy())

            # scoring
            y_test = split_op.intermediate[1]
            df["scores"] = df["vals"].apply(
                lambda pred: scoring_func(y_test, pred))
            df = df.drop("vals", axis=1)
            results.append(df)

        results = pd.concat(results, axis=0)
        results = results.groupby("id").aggregate("mean").sort_values(by="scores", ascending=False)
        self.results_ = results
        return predictions if return_predictions else None

    def compute(self, start_pos: int, mode="fit_transform"):
        """Compute the pipeline from start_pos onwards with given inputs."""
        ops_to_compute = self.ops_ordered[start_pos:]
        if len(self.flagged_for_recomputation) != 0:
            ops_to_compute = self.flagged_for_recomputation + ops_to_compute
        self.mode = mode

        for node in ops_to_compute:
            self.process_op(node)

        if mode == "predict":
            pred = self.ops_ordered[-1].intermediate
            if isinstance(pred, list):
                return pd.DataFrame(pred)
            else:
                return pd.DataFrame({"vals": [pred], "id": ["default"]})
        return None

    def compute_xy(self) -> SplitOp:
        """Compute nodes until X and y nodes are found and store them."""
        for i, op in enumerate(self.ops_ordered):
            if op.is_split_op:
                self.pos_split_op = i
                return op
            self.process_op(op)
            if isinstance(op, ImplOp) and isinstance(op.skrub_impl, EvalMode):
                self.flagged_for_recomputation.append(op)
        raise RuntimeError("X and y nodes not found in the DAG")

    def process_op(self, op: Op):
        """Process a single DataOp node and return its output."""
        t0 = perf_counter() if self.timings is not None else 0
        try:
            op.process(mode=self.mode, environment=self.env)
        except Exception as e:
            raise RuntimeError(f"Error processing '{op}': {e}")

        if self.timings is not None:
            duration = perf_counter() - t0
            self.timings.append((str(op), duration))
        return op