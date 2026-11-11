from __future__ import annotations
from copy import deepcopy
from time import perf_counter
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, check_cv
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._dataframe_ops import SplitOp
from stratum.optimizer.logical._ops import FITTING_MODE, Op
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.runtime._buffer_pool import BufferPool
from stratum.runtime._object_size import prettify_bytes
import polars as pl

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stratum.runtime._reuse_cache import ReuseCache


class Scheduler:
    """Scheduler for executing pre-planned Op DAGs in linearized order.

    - All intermediates are immutable, including nested objects. Operators must
      not modify them in place; mutable workspace must be operator-local.
    - Operators borrow inputs only for the duration of ``process`` and hand
      outputs to the runtime on return. Retaining references to either beyond
      execution is a memory leak and violates the operator contract.
    - The runtime owns intermediate lifetimes. The scheduler pins inputs during
      execution and performs the releases scheduled by lifetime planning.
      The pool and reuse cache share materialized results without copying.
    """

    def __init__(self, print_heavy_hitters=False, t0=None, reuse_cache: ReuseCache | None = None):
        self.mode = "fit_transform"
        self.linearized_dag = None
        self.recompute_ops: list[Op] = []
        self.pos_split_op: int | None = None
        self.timings = [] if print_heavy_hitters else None
        self.observer = None  # Inspection-only hooks, see stratum._profiler.
        self.results_ = None
        self.cv_id = -1
        self.pool = BufferPool()
        self.t0 = t0 if t0 is not None else perf_counter()
        self._pinned_ops: set[Op] = set()
        self.buffer_pool_overhead = 0
        #: Shared in-memory cache for read-only outputs.
        self.reuse_cache = reuse_cache
        #: Results may alias cached values, including through uncacheable ops.
        #: Detach them once when returning to the caller, outside execution.
        self._shares_cache_values = reuse_cache is not None

    def _pool_cache_hits(self) -> None:
        """Borrow the values retained during reuse planning without copying."""
        for op in self.linearized_dag:
            if op.cache_hit is not None:
                self._shares_cache_values = True
                self.pool.put(op, op.cache_hit.take())

    def _finish(self):
        """End of execution. Remove all buffers."""
        self.pool.remove_all()
        self.log_memory_usage()
        logger.debug(f"Scheduler finished: {self.pool.total_removed} buffers removed total")


    def evaluate(self, seed: int = 42, test_size=0.2):
        """Evaluate the pipeline with a train/test split and return predictions."""
        try:
            result = self._evaluate(seed, test_size)
            return _copy_result(result) if self._shares_cache_values else result
        finally:
            self._finish()

    def _evaluate(self, seed: int, test_size):
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
        """Perform grid search with cross-validation on the plan.

        An explicit ``cv`` takes precedence over the splitter the plan declares with
        ``mark_as_X(cv=...)``, as in skrub. With neither, the folds are ``KFold(5)``.
        """
        try:
            result = self._grid_search(cv)
            return _copy_result(result) if self._shares_cache_values else result
        finally:
            self._finish()

    def _grid_search(self, cv):
        if cv is not None:
            cv = check_cv(cv)
        logger.debug("\n" + "="*100 + "\n" + "Starting grid search" + "\n" + "="*100 + "\n")
        # Validate X/y markers before the sink type so a plan missing both
        # reports the missing input annotations first.
        split_op = self.compute_xy()
        sink = self.linearized_dag[-1]
        if not isinstance(sink, ScoreCandidatesOp):
            raise RuntimeError(
                "this plan was not built for a search: it ends in"
                f" {type(sink).__name__}, not ScoreCandidates. Build it with"
                " `optimize(..., search=SearchConfig(...))`."
            )
        if cv is None:
            cv = self._declared_splitter(split_op)

        results, predictions = [], []

        logger.debug("\n" + "="*100 + "\n" + "XY computed" + "\n" + "="*100 + "\n")
        results = self.cross_validate(split_op, cv, predictions, results,
                                      sink.emit_predictions, sink.response_mode)
        self.results_ = results
        return predictions if sink.emit_predictions else None

    def _declared_splitter(self, split_op: SplitOp):
        """The splitter the plan declares, or ``KFold(5)`` if it declares none.

        The pass up to the split has computed it, from operands such as ``groups``
        that the plan computes anyway (see `BuildSplitterOp`).
        """
        if split_op.splitter is None:
            return check_cv(None)
        splitter = self.pool.pin(split_op.splitter)
        self.pool.unpin(split_op.splitter)
        return splitter

    def cross_validate(self, split_op, cv, predictions: list, results: list,
                       return_predictions: bool, response_mode: str = "predict"):
        """Perform cross-validation on the plan, one scored row per candidate per fold.

        ``response_mode`` is the estimator method the test-fold pass calls, which the
        scorer chose at plan time (see `ScoreCandidatesOp`).
        """
        # The split op's inputs are [X, y] (see add_splitting_op). Both are still
        # in the pool after compute_xy because lifetime planning retains them.
        # Pin both while generating folds; stratified splitters need the labels.
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
        # sklearn scorers use a higher-is-better convention; `neg_*` scorers
        # negate loss metrics so candidates can be ranked in descending order.
        results = results.group_by("id").mean().sort("scores", descending=True)
        return results

    @property
    def retained_state_memory_usage(self) -> int:
        """Fitted state held by the plan, separate from evictable output buffers.

        Each operator supplies a cached scalar. State remains live after pool
        cleanup while the plan retains its operators; refits replace its size.
        Shared storage across operators or outputs can be counted more than once.
        TODO: reserve this unspillable state against the execution memory budget.
        """
        return sum(op.retained_state_size() for op in self.linearized_dag or ()
                   if isinstance(op, PhysicalOp))

    @property
    def memory_usage(self) -> int:
        """Estimated pool and operator-state bytes, excluding the reuse cache."""
        return self.pool.memory_usage + self.retained_state_memory_usage

    def log_memory_usage(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Pool size: %s. Buffer memory: %s. Retained operator state: %s",
                         self.pool.active_count, self.pool.total_size,
                         prettify_bytes(self.retained_state_memory_usage))

    def process_op(self, op: Op):
        """Process one operator, store its output in the pool and return the operator."""
        if op.cache_hit is not None:
            # Already loaded into the pool. Reuse planning cut its inputs, so
            # this op has no work or input removals left to perform.
            return op
        if self.mode == FITTING_MODE and op.dead_in_fit:
            # Skip execution, but release inputs scheduled for removal here.
            # Lifetime planning includes this operator even when fitting skips it.
            for in_op in op.remove_after:
                self.pool.remove(in_op)
            return op
        logger.debug(f"[{perf_counter() - self.t0:.2f}s] Processing op: {op}")

        obs = self.observer
        token = t1 = t2 = None
        try:
            t0 = perf_counter()

            # 1. pin all inputs
            inputs = [self.pool.pin(in_op) for in_op in op.inputs]

            # 2. process operator
            if obs is not None:
                token = obs.before(op)
            t1 = perf_counter()
            result = op.process(mode=self.mode, inputs=inputs)
            t2 = perf_counter()

            # Retain eligible results while they are still in memory. Admission
            # never needs to reload or take ownership of a spilled pool entry.
            # Operators read inputs without mutating them, so the cache can
            # retain the same objects as this execution.
            if self.reuse_cache is not None and op.cache_key is not None:
                self.reuse_cache.put(op.cache_key, result, label=str(op))

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
            if obs is not None:
                obs.after(op, self.mode, self.cv_id, token, t1, t2, buffer_overhead)

        except Exception as e:
            if obs is not None:
                obs.failed(op, self.mode, self.cv_id, token, t0, t1, t2)
            raise RuntimeError(f"[{self.mode}] Error processing '{op}': {e}")

        return op

    def _format_predict_result(self, rows):
        """The candidate-set sink's labelled rows as a frame."""
        return pl.DataFrame(rows)


class SequentialScheduler(Scheduler):
    def __init__(self, linearized_dag, split_pos, recompute_ops,
                 print_heavy_hitters=False, t0=None, reuse_cache: ReuseCache | None = None):
        super().__init__(print_heavy_hitters, t0=t0, reuse_cache=reuse_cache)
        self.linearized_dag = linearized_dag
        self.pos_split_op = split_pos
        self.recompute_ops = recompute_ops
        try:
            self._pool_cache_hits()
        except BaseException:
            self._finish()
            raise

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


def _copy_result(value):
    """Detach a public result from the read-only values shared during execution.

    Used only on return from evaluate/grid_search, never on cache admission or
    retrieval. Pandas 3 copy-on-write and polars clones avoid copying ordinary
    frame buffers. Python objects inside cells need recursive copies; NumPy
    arrays and arbitrary Python objects need independent storage.
    """
    if isinstance(value, pd.DataFrame):
        copied = value.copy(deep=False)
        for i, dtype in enumerate(value.dtypes):
            if dtype == object:
                # Work by position to preserve duplicate column labels and keep
                # object dtype rather than inferring a different dtype on copy.
                copied.isetitem(i, pd.Series(deepcopy(value.iloc[:, i].to_numpy()),
                                            index=value.index, dtype=object))
        return copied
    if isinstance(value, pd.Series):
        copied = value.copy(deep=False)
        if value.dtype == object:
            copied.iloc[:] = deepcopy(value.to_numpy())
        return copied
    if isinstance(value, pl.DataFrame):
        # Candidate predictions can be Python arrays stored in Object cells.
        columns = [_copy_result(value[name]) for name, dtype in value.schema.items()
                   if dtype == pl.Object]
        return value.with_columns(columns) if columns else value.clone()
    if isinstance(value, pl.Series):
        if value.dtype == pl.Object:
            return pl.Series(value.name, [_copy_result(item) for item in value], dtype=pl.Object)
        return value.clone()
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        return value.copy()
    if type(value) is list:
        return [_copy_result(item) for item in value]
    if type(value) is tuple:
        return tuple(_copy_result(item) for item in value)
    if type(value) is dict:
        return {deepcopy(key): _copy_result(item) for key, item in value.items()}
    return deepcopy(value)
