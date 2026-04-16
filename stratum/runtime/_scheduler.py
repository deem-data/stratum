from __future__ import annotations
from time import perf_counter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, check_cv
from sklearn.metrics._scorer import _Scorer, get_scorer
from stratum.optimizer.ir._dataframe_ops import SplitOp
from stratum.optimizer.ir._ops import Op
from stratum.runtime._buffer_pool import BufferPool
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
    """Scheduler for executing pre-planned Op DAGs in linearized order."""

    def __init__(self, print_heavy_hitters=False, env=None, t0=None):
        self.mode = "fit_transform"
        self.env = env if env else {}
        self.linearized_dag = None
        self.recompute_ops: list[Op] = []
        self.pos_split_op: int | None = None
        self.timings = [] if print_heavy_hitters else None
        self.results_ = None
        self.cv_id = -1
        self.pool = BufferPool()
        self.t0 = t0 if t0 is not None else perf_counter()
        self._pinned_ops: set[Op] = set()

    def _finish(self):
        """End of execution. Release all buffers."""
        self.pool.release_all()
        logger.debug(f"Scheduler finished: {self.pool.total_released} buffers released total")


    def evaluate(self, seed: int = 42, test_size=0.2):
        """Evaluate the pipeline with a train/test split and return predictions."""
        try:
            split_op = self.compute_xy()
        except RuntimeError as e:
            if "X and y nodes not found in the DAG" in str(e):
                logger.warning("X and y nodes not found in the DAG, returning the last node")
                return self.pool.get(self.linearized_dag[-1])
            else:
                raise e

        x_data = self.pool.get(split_op.inputs[0])
        train_index, test_index = train_test_split(range(len(x_data)), test_size=test_size, random_state=seed)
        split_op.indices = train_index
        self.compute(self.pos_split_op)
        split_op.indices = test_index
        pred = self.compute(self.pos_split_op, mode="predict")
        return pred["vals"][0]

    def grid_search(self, cv=None, scoring=None, return_predictions=False):
        """Perform grid search with cross-validation on the logical DAG."""
        cv = check_cv(cv)

        logger.debug("\n" + "="*100 + "\n" + "Starting grid search" + "\n" + "="*100 + "\n")
        split_op = self.compute_xy()

        results, predictions = [], []

        logger.debug("\n" + "="*100 + "\n" + "XY computed" + "\n" + "="*100 + "\n")
        results = self.cross_validate(split_op, cv, scoring, predictions, results, return_predictions)
        self.results_ = results
        self._finish()
        return predictions if return_predictions else None

    def cross_validate(self, split_op, cv, scoring, predictions: list, results: list, return_predictions: bool):
        """Perform cross-validation on the logical DAG."""
        scoring_func, greater_is_better = get_scoring_func(scoring)

        x_data = self.pool.get(split_op.inputs[0])
        for i, (train_index, test_index) in enumerate(cv.split(x_data)):
            self.cv_id = i
            logger.debug(f"CV Fold Nr. {i + 1}")

            split_op.indices = train_index
            self.compute(self.pos_split_op)
            logger.debug("\n" + "="*100 + "\n" + "Training done for fold " + str(i+1) + "\n" + "="*100 + "\n")
            split_op.indices = test_index
            df, y_test = self.compute(self.pos_split_op, mode="predict")
            logger.debug("\n" + "="*100 + "\n" + "Predicting done for fold " + str(i+1) + "\n" + "="*100 + "\n")
            if return_predictions:
                predictions.append(df)

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
            inputs = op.resolve_inputs(self.pool)
            result = op.process(mode=self.mode, environment=self.env, inputs=inputs)
            op.release_inputs(self.pool)
            if self.timings is not None:
                duration = perf_counter() - t0
                self.timings.append((str(op), duration))

        except Exception as e:
            raise RuntimeError(f"[{self.mode}] Error processing '{op}': {e}")

        self.pool.put(op, result)
        logger.debug(f"[{perf_counter() - self.t0:.2f}s] Pool size: {self.pool.active_count}")

        return op

    def _format_predict_result(self, pred):
        """Helper method to format prediction results consistently."""
        if isinstance(pred, list):
            return pl.DataFrame(pred)
        elif isinstance(pred, dict) and "id" in pred and "vals" in pred:
            return pl.DataFrame([pred])
        else:
            return pl.DataFrame({"vals": [pred], "id": ["default"]})


class SequentialScheduler(Scheduler):
    def __init__(self, linearized_dag, split_pos, recompute_ops,
                 print_heavy_hitters=False, env=None, t0=None):
        super().__init__(print_heavy_hitters, env=env, t0=t0)
        self.linearized_dag = linearized_dag
        self.pos_split_op = split_pos
        self.recompute_ops = recompute_ops

    def evaluate(self, seed: int = 42, test_size=0.2):
        """Evaluate the pipeline with a train/test split and return predictions."""

        try:
            split_op = self.compute_xy()
        except RuntimeError as e:
            if "X and y nodes not found in the DAG" in str(e):
                logger.warning("X and y nodes not found in the DAG, returning the last node")
                return self.pool.get(self.linearized_dag[-1])
            else:
                raise e

        x_data = self.pool.get(split_op.inputs[0])
        train_index, test_index = train_test_split(range(len(x_data)), test_size=test_size, random_state=seed)
        split_op.indices = train_index
        self.compute(self.pos_split_op)
        split_op.indices = test_index
        pred, _ = self.compute(self.pos_split_op, mode="predict")
        self._finish()
        return pred["vals"][0]

    def compute(self, start_pos: int, mode="fit_transform"):
        """Compute the pipeline from start_pos onwards with given inputs."""
        ops_to_compute = self.linearized_dag[start_pos:]
        if len(self.recompute_ops) != 0:
            ops_to_compute = self.recompute_ops + ops_to_compute
        self.mode = mode

        y_true = None
        for node in ops_to_compute:
            self.process_op(node)
            if mode == "predict" and isinstance(node, SplitOp):
                y_true = self.pool.get(node)[1]

        if mode == "predict":
            pred = self.pool.get(self.linearized_dag[-1])
            return self._format_predict_result(pred), y_true
        return None

    def compute_xy(self) -> SplitOp:
        """Compute nodes until the split op is reached."""
        for i, op in enumerate(self.linearized_dag):
            if op.is_split_op:
                return op
            self.process_op(op)
        raise RuntimeError("X and y nodes not found in the DAG")
