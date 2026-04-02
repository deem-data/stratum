from time import perf_counter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, check_cv
from sklearn.metrics._scorer import _Scorer, get_scorer
from skrub._data_ops._data_ops import EvalMode
from stratum.optimizer.ir._dataframe_ops import SplitOp
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.ir._ops import ImplOp, Op
import polars as pl

import logging
logger = logging.getLogger(__name__)

def get_scoring_func(scoring):
    """Get scoring function from str or _Scorer object."""
    if type(scoring) == str:
        scoring = get_scorer(scoring)
    if type(scoring) == _Scorer:
        logger.info(f"Using scorer: {scoring}")
        greater_is_better = scoring._sign > 0
        scoring_func = scoring._score_func
    else:
        greater_is_better = False
        scoring_func = mean_squared_error
    return scoring_func, greater_is_better

class Scheduler:
    """Scheduler for executing DataOpDAGs in topological order."""
    
    def __init__(self, print_heavy_hitters=False, env=None, t0 = None):
        """Initialize scheduler with a data operations DAG."""
        self.mode = "fit_transform"
        self.env = env if env else {}
        self.flagged_for_recomputation = []
        self.pos_split_op = None
        self.timings = [] if print_heavy_hitters else None
        self.results_ = None
        self.cv_id = -1
        self.t0 = t0 if t0 is not None else perf_counter()

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
        """Perform grid search with cross-validation on the logical DAG."""
        # default to scikit-learn's CV
        cv = check_cv(cv)

        # start with computing till we reach the split op
        logger.debug("\n" + "="*100 + "\n" + "Starting grid search" + "\n" + "="*100 + "\n")
        split_op = self.compute_xy()
        results, predictions = [], []

        logger.debug("\n" + "="*100 + "\n" + "XY computed" + "\n" + "="*100 + "\n")
        results = self.cross_validate(split_op, cv, scoring, predictions, results, return_predictions)
        self.results_ = results
        return predictions if return_predictions else None

    def cross_validate(self, split_op, cv, scoring, predictions: list, results: list, return_predictions: bool):
        """Perform cross-validation on the logical DAG."""
        scoring_func, greater_is_better = get_scoring_func(scoring)

        # TODO we can parallelize over the folds
        for i, (train_index, test_index) in enumerate(cv.split(split_op.inputs[0].intermediate)):
            self.cv_id = i
            logger.debug(f"CV Fold Nr. {i + 1}")

            # fit and predict the pipeline
            split_op.indices = train_index
            self.compute(self.pos_split_op)
            logger.debug("\n" + "="*100 + "\n" + "Training done for fold " + str(i+1) + "\n" + "="*100 + "\n")
            split_op.indices = test_index
            df, y_test = self.compute(self.pos_split_op, mode="predict")
            logger.debug("\n" + "="*100 + "\n" + "Predicting done for fold " + str(i+1) + "\n" + "="*100 + "\n")
            if return_predictions:
                predictions.append(df)

            # scoring
            df = df.with_columns(df["vals"].map_elements(lambda pred: scoring_func(y_test, pl.Series(pred))).alias("scores"))
            df = df.drop("vals")
            results.append(df)

        results = pl.concat(results)
        results = results.group_by("id").mean().sort("scores", descending=greater_is_better)
        return results

    def process_op(self, op: Op):
        """Process a single DataOp node and return its output."""
        logger.debug(f"[{perf_counter() - self.t0:.2f}s] Processing op: {op}")

        try:
            t0 = perf_counter() if self.timings is not None else 0
            op.process(mode=self.mode, environment=self.env)
            if self.timings is not None:
                duration = perf_counter() - t0
                self.timings.append((str(op), duration))

        except Exception as e:
            raise RuntimeError(f"[{self.mode}] Error processing '{op}': {e}")

        return op

    def _format_predict_result(self, pred):
        """Helper method to format prediction results consistently."""
        if isinstance(pred, list):
            return pl.DataFrame(pred)
        elif isinstance(pred, dict) and "id" in pred and "vals" in pred:
            return pl.DataFrame([pred])
        else:
            return pl.DataFrame({"vals": [pred], "id": ["default"]})

    def _flag_op_for_recomputation_if_needed(self, op: Op):
        """Helper method to flag an op for recomputation if it's an ImplOp with EvalMode."""
        if isinstance(op, ImplOp) and isinstance(op.skrub_impl, EvalMode):
            self.flagged_for_recomputation.append(op)

class SequentialScheduler(Scheduler):
    def __init__(self, dag_sink: Op, print_heavy_hitters=False, env=None, t0 = None):
        super().__init__(print_heavy_hitters, env=env, t0=t0)
        self.ops_ordered = [op for op in topological_iterator(dag_sink)]

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
        pred, _ = self.compute(self.pos_split_op, mode="predict")
        return pred["vals"][0]


    def compute(self, start_pos: int, mode="fit_transform"):
        """Compute the pipeline from start_pos onwards with given inputs."""
        ops_to_compute = self.ops_ordered[start_pos:]
        if len(self.flagged_for_recomputation) != 0:
            ops_to_compute = self.flagged_for_recomputation + ops_to_compute
        self.mode = mode

        y_true = None
        for node in ops_to_compute:
            self.process_op(node)
            if mode == "predict" and isinstance(node, SplitOp):
                y_true = node.intermediate[1]

        if mode == "predict":
            pred = self.ops_ordered[-1].intermediate
            return self._format_predict_result(pred), y_true
        return None

    def compute_xy(self) -> SplitOp:
        """Compute nodes until X and y nodes are found and store them."""
        for i, op in enumerate(self.ops_ordered):
            if op.is_split_op:
                self.pos_split_op = i
                return op
            self.process_op(op)
            self._flag_op_for_recomputation_if_needed(op)
        raise RuntimeError("X and y nodes not found in the DAG")