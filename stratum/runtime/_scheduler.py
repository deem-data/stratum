from time import perf_counter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, check_cv
from sklearn.metrics._scorer import _Scorer
from skrub._data_ops._data_ops import EvalMode
from stratum.logical_optimizer._dataframe_ops import SplitOp
from stratum.logical_optimizer._op_utils import topological_iterator
from stratum.logical_optimizer._ops import ImplOp, Op
import polars as pl

import logging
logger = logging.getLogger(__name__)


def get_scoring_func(scoring):
    """Get scoring function from str or _Scorer object."""
    if type(scoring) == str:
        coeff = -1 if scoring.startswith("neg_") else 1
        scoring_func = lambda test, pred: mean_squared_error(test, pred) * coeff
    elif type(scoring) == _Scorer:
        scoring_func = scoring._score_func
    else:
        scoring_func = mean_squared_error
    return scoring_func

class Scheduler:
    """Scheduler for executing DataOpDAGs in topological order."""
    
    def __init__(self, dag_sink: Op, print_heavy_hitters=False):
        """Initialize scheduler with a data operations DAG."""
        self.ops_ordered = [op for op in topological_iterator(dag_sink)]
        self.mode = "fit_transform"
        self.env = {}
        self.flagged_for_recomputation = []
        self.pos_split_op = None
        self.timings = [] if print_heavy_hitters else None
        self.results_ = None

class SequentialScheduler(Scheduler):
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

        # start with computing till X and y node
        logger.debug("\n" + "="*100 + "\n" + "Starting grid search" + "\n" + "="*100 + "\n")
        split_op = self.compute_xy()
        results, predictions = [], []

        logger.debug("\n" + "="*100 + "\n" + "XY computed" + "\n" + "="*100 + "\n")
        results = self.cross_validate(split_op, cv, scoring, predictions, results, return_predictions)
        self.results_ = results
        return predictions if return_predictions else None

    def cross_validate(self, split_op, cv, scoring, predictions: list, results: list, return_predictions: bool):
        scoring_func = get_scoring_func(scoring)

        # TODO we can parallelize over the folds
        for i, (train_index, test_index) in enumerate(cv.split(split_op.inputs[0].intermediate)):
            logger.debug(f"CV Fold Nr. {i + 1}")

            # fit and predict the pipeline
            split_op.indices = train_index
            self.compute(self.pos_split_op)
            logger.debug("\n" + "="*100 + "\n" + "Training done for fold " + str(i+1) + "\n" + "="*100 + "\n")
            split_op.indices = test_index
            df = self.compute(self.pos_split_op, mode="predict")
            logger.debug("\n" + "="*100 + "\n" + "Predicting done for fold " + str(i+1) + "\n" + "="*100 + "\n")
            if return_predictions:
                predictions.append(df)

            # scoring
            y_test = split_op.intermediate[1]
            df = df.with_columns(df["vals"].map_elements(lambda pred: scoring_func(y_test, pl.Series(pred))).alias("scores"))
            df = df.drop("vals")
            results.append(df)

        results = pl.concat(results)
        results = results.group_by("id").mean().sort("scores", descending=True)
        return results

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
                return pl.DataFrame(pred)
            else:
                return pl.DataFrame({"vals": [pred], "id": ["default"]})
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
        logger.debug(f"Processing op: {op}")
        t0 = perf_counter() if self.timings is not None else 0
        try:
            op.process(mode=self.mode, environment=self.env)
        except Exception as e:
            raise RuntimeError(f"[{self.mode}] Error processing '{op}': {e}")

        if self.timings is not None:
            duration = perf_counter() - t0
            self.timings.append((str(op), duration))
        return op

class ParallelScheduler(Scheduler):
    pass