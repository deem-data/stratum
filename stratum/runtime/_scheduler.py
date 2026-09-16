from __future__ import annotations
from time import perf_counter
from sklearn.model_selection import train_test_split, check_cv
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._dataframe_ops import SplitOp
from stratum.optimizer.logical._ops import FITTING_MODE, Op
from stratum.runtime._buffer_pool import BufferPool
import polars as pl

import logging
logger = logging.getLogger(__name__)


class Scheduler:
    """Scheduler for executing pre-planned Op DAGs in linearized order."""

    def __init__(self, print_heavy_hitters=False, t0=None):
        self.mode = "fit_transform"
        self.linearized_dag = None
        self.recompute_ops: list[Op] = []
        self.pos_split_op: int | None = None
        self.timings = [] if print_heavy_hitters else None
        self.results_ = None
        self.cv_id = -1
        self.pool = BufferPool()
        self.t0 = t0 if t0 is not None else perf_counter()
        self._pinned_ops: set[Op] = set()
        self.buffer_pool_overhead = 0

    def _finish(self):
        """End of execution. Remove all buffers."""
        self.pool.remove_all()
        self.log_memory_usage()
        logger.debug(f"Scheduler finished: {self.pool.total_removed} buffers removed total")


    def evaluate(self, seed: int = 42, test_size=0.2):
        """Evaluate the pipeline with a train/test split and return predictions."""
        try:
            split_op = self.compute_xy()
        except RuntimeError as e:
            if "X and y nodes not found in the DAG" in str(e):
                logger.warning("X and y nodes not found in the DAG, returning the last node")
                return self.pool.pin(self.linearized_dag[-1])
            else:
                raise e

        x_data = self.pool.pin(split_op.inputs[0])
        train_index, test_index = train_test_split(range(len(x_data)), test_size=test_size, random_state=seed)
        split_op.indices = train_index
        self.compute(self.pos_split_op)
        split_op.indices = test_index
        rows = self.compute(self.pos_split_op, mode="predict")
        return self._format_predict_result(rows)["vals"][0]

    def grid_search(self, cv=None):
        """Perform grid search with cross-validation on the plan."""
        cv = check_cv(cv)
        logger.debug("\n" + "="*100 + "\n" + "Starting grid search" + "\n" + "="*100 + "\n")
        # Before the sink: a plan with no X/y has no candidate set either, and the
        # missing marks are the more useful thing to report.
        split_op = self.compute_xy()
        sink = self.linearized_dag[-1]
        if not isinstance(sink, ScoreCandidatesOp):
            raise RuntimeError(
                "this plan was not built for a search: it ends in"
                f" {type(sink).__name__}, not ScoreCandidates. Build it with"
                " `optimize(..., search=SearchConfig(...))`."
            )

        results, predictions = [], []

        logger.debug("\n" + "="*100 + "\n" + "XY computed" + "\n" + "="*100 + "\n")
        results = self.cross_validate(split_op, cv, predictions, results,
                                      sink.emit_predictions, sink.response_mode)
        self.results_ = results
        self._finish()
        return predictions if sink.emit_predictions else None

    def cross_validate(self, split_op, cv, predictions: list, results: list,
                       return_predictions: bool, response_mode: str = "predict"):
        """Perform cross-validation on the plan, one scored row per candidate per fold.

        ``response_mode`` is the estimator method the test-fold pass calls, which the
        scorer chose at plan time (see `ScoreCandidatesOp`).
        """
        # The split op's inputs are [X, y] (see add_splitting_op). Both are still
        # in the pool after compute_xy and are pinned by removal planning, so the
        # labels can be handed to the splitter: stratified splitters need them,
        # and every sklearn splitter accepts split(X, y=None, groups=None).
        x_data = self.pool.pin(split_op.inputs[0])
        y_data = self.pool.pin(split_op.inputs[1])
        splits = []
        for i, (train_index, test_index) in enumerate(cv.split(x_data, y_data)):
            train_key = ("__cv_split", "train", i)
            test_key = ("__cv_split", "test", i)
            splits.append((i, train_key, test_key))
            self.pool.put(test_key, test_index)
            self.pool.put(train_key, train_index)
            self.log_memory_usage()
        self.pool.unpin(split_op.inputs[0])
        self.pool.unpin(split_op.inputs[1])

        for i, train_ids_handle, test_ids_handle in splits:
            self.cv_id = i
            logger.debug(f"CV Fold Nr. {i + 1}")

            split_op.indices = self.pool.pin(train_ids_handle)
            self.compute(self.pos_split_op)
            self.pool.unpin(train_ids_handle)
            self.pool.remove(train_ids_handle)
            logger.debug("\n" + "="*100 + "\n" + "Training done for fold " + str(i+1) + "\n" + "="*100 + "\n")

            split_op.indices = self.pool.pin(test_ids_handle)
            fold = self.compute(self.pos_split_op, mode=response_mode)
            self.pool.unpin(test_ids_handle)
            self.pool.remove(test_ids_handle)
            logger.debug("\n" + "="*100 + "\n" + "Predicting done for fold " + str(i+1) + "\n" + "="*100 + "\n")
            if return_predictions:
                predictions.append(fold.select("id", "vals"))
            results.append(fold.select("id", "scores"))

        results = pl.concat(results)
        # A scorer returns a utility whatever its metric -- a `neg_*` scorer carries the
        # sign that makes it one -- so the best candidate is always the highest score.
        results = results.group_by("id").mean().sort("scores", descending=True)
        return results

    def log_memory_usage(self):
        logger.debug(f"Pool size: {self.pool.active_count}. Memory usage: {self.pool.total_size}")

    def process_op(self, op: Op):
        """Process a single DataOp node and return its output."""
        if self.mode == FITTING_MODE and op.dead_in_fit:
            # Nothing this pass runs reads its output, so it is not pinned, run or
            # stored. Its removals still happen: it is the last consumer of those
            # buffers, and skipping it is the only reason no one else drops them.
            for in_op in op.remove_after:
                self.pool.remove(in_op)
            return op
        logger.debug(f"[{perf_counter() - self.t0:.2f}s] Processing op: {op}")

        try:
            t0 = perf_counter()

            # 1. pin all inputs
            inputs = [self.pool.pin(in_op) for in_op in op.inputs]

            # 2. process operator
            t1 = perf_counter()
            result = op.process(mode=self.mode, inputs=inputs)
            t2 = perf_counter()

            # 3. unpin inputs
            for in_op in op.inputs:
                self.pool.unpin(in_op)

            # 4. remove unnecessary intermediates from buffer pool
            for in_op in op.remove_after:
                self.pool.remove(in_op)

            # 5. add output to the buffer pool
            self.pool.put(op, result)
            self.log_memory_usage()

            t3 = perf_counter()
            process_duration = t2 - t1
            buffer_overhead = t3 - t0 - process_duration
            self.buffer_pool_overhead += buffer_overhead

            if self.timings is not None:
                duration = t2 - t1
                self.timings.append((str(op), process_duration))

        except Exception as e:
            raise RuntimeError(f"[{self.mode}] Error processing '{op}': {e}")

        return op

    def _format_predict_result(self, rows):
        """The candidate-set sink's labelled rows as a frame."""
        return pl.DataFrame(rows)


class SequentialScheduler(Scheduler):
    def __init__(self, linearized_dag, split_pos, recompute_ops,
                 print_heavy_hitters=False, t0=None):
        super().__init__(print_heavy_hitters, t0=t0)
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
                return self.pool.pin(self.linearized_dag[-1])
            else:
                raise e

        x_data = self.pool.pin(split_op.inputs[0])
        train_index, test_index = train_test_split(range(len(x_data)), test_size=test_size, random_state=seed)
        split_op.indices = train_index
        self.compute(self.pos_split_op)
        split_op.indices = test_index
        rows = self.compute(self.pos_split_op, mode="predict")
        self._finish()
        return self._format_predict_result(rows)["vals"][0]

    def compute(self, start_pos: int, mode="fit_transform"):
        """Compute the plan from start_pos onwards; outside the fitting pass, return its sink."""
        ops_to_compute = self.linearized_dag[start_pos:]
        if len(self.recompute_ops) != 0:
            ops_to_compute = self.recompute_ops + ops_to_compute
        self.mode = mode

        for node in ops_to_compute:
            self.process_op(node)

        sink = self.linearized_dag[-1]
        out = self.pool.pin(sink) if mode != "fit_transform" else None
        self.pool.remove(sink)
        self.log_memory_usage()
        return out

    def compute_xy(self) -> SplitOp:
        """Compute nodes until the split op is reached."""
        for i, op in enumerate(self.linearized_dag):
            if op.is_split_op:
                return op
            self.process_op(op)
        raise RuntimeError("X and y nodes not found in the DAG")
