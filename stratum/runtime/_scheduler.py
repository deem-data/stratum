from __future__ import annotations
from copy import deepcopy
from time import perf_counter
from typing import TYPE_CHECKING
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._split_ops import SplitIndicesOp, SplitOp
from stratum.optimizer.logical._ops import Op
from stratum.runtime._buffer_pool import BufferPool
from stratum.runtime._object_size import get_size, prettify_bytes
from stratum.runtime._state_store import StateStore
import numpy as np
import pandas as pd
import polars as pl

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stratum.runtime._reuse_cache import ReuseCache


class Scheduler:
    """Scheduler for executing pre-planned Op DAGs in linearized order.

    - All intermediates and borrowed fitted state are immutable, including nested
      objects. Operators must not modify them in place; mutable workspace must
      be operator-local.
    - Operators borrow inputs only for the duration of ``process`` and transfer
      outputs to the runtime on return. Retaining references to either beyond
      execution is a memory leak and violates the operator contract.
    - The runtime owns intermediate lifetimes. The scheduler pins inputs during
      execution and performs the releases scheduled by lifetime planning.
      The pool and reuse cache share materialized results without copying.
    """

    def __init__(self, print_heavy_hitters=False, t0=None, reuse_cache: ReuseCache | None = None):
        self.linearized_dag = None
        self.pos_split_op: int | None = None
        #: Operators scheduled for the most recent fold, retained for inspection.
        self.fold_ops: list[Op] | None = None
        self.timings = [] if print_heavy_hitters else None
        self.observer = None  # Inspection-only hooks, see stratum._profiler.
        self.results_ = None
        self.cv_id = -1
        self.pool = BufferPool()
        self.state_store = StateStore()
        self.t0 = t0 if t0 is not None else perf_counter()
        self._pinned_ops: set[Op] = set()
        self.buffer_pool_overhead = 0
        #: Shared in-memory cache for read-only outputs.
        self.reuse_cache = reuse_cache
        #: Results may alias cached values, including through uncacheable ops.
        #: Detach them once when returning to the caller, outside execution.
        self._shares_cache_values = reuse_cache is not None

    def _pool_cache_hits(self, ops: list[Op]) -> None:
        """Register cached values retained during reuse planning in the buffer pool."""
        for op in ops:
            hit = op.cache_hit
            if hit is None:
                continue
            self._shares_cache_values = True
            self.pool.put(op, hit.take())

    def _finish(self):
        """Release execution buffers and fitted states, including after failures."""
        self.state_store.clear()
        self.pool.remove_all()
        self.log_memory_usage()
        logger.debug(f"Scheduler finished: {self.pool.total_removed} buffers removed total")

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
        logger.debug("\n" + "="*100 + "\n" + "Starting grid search" + "\n" + "="*100 + "\n")
        # Validate X/y markers before the sink type so a plan missing both
        # reports the missing input annotations first.
        split_op = self.compute_xy(cv=cv)
        sink = self.linearized_dag[-1]
        if not isinstance(sink, ScoreCandidatesOp):
            raise RuntimeError(
                "this plan was not built for a search: it ends in"
                f" {type(sink).__name__}, not ScoreCandidates. Build it with"
                " `optimize(..., search=SearchConfig(...))`."
            )
        results, predictions = [], []

        logger.debug("\n" + "="*100 + "\n" + "XY computed" + "\n" + "="*100 + "\n")
        results = self.cross_validate(split_op, predictions, results,
                                      sink.emit_predictions)
        self.results_ = results
        return predictions if sink.emit_predictions else None

    def cross_validate(self, split_op, predictions: list, results: list,
                       return_predictions: bool):
        """Perform cross-validation on the plan, one scored row per candidate per fold.

        Each execution of the split and its descendants fits on training rows
        and scores test-row outputs for one fold (see train/test unrolling).
        """
        n_splits = self.pool.pin(split_op.indices)
        self.pool.unpin(split_op.indices)
        for i in range(n_splits):
            logger.debug(f"CV Fold Nr. {i + 1}")

            fold = self.run_fold(i)
            logger.debug("\n" + "="*100 + "\n" + "Fold " + str(i+1) + " done" + "\n" + "="*100 + "\n")
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
        """Estimated bytes currently owned by the runtime state store."""
        return self.state_store.memory_usage

    @property
    def memory_usage(self) -> int:
        """Estimated pool and state bytes, excluding the reuse cache."""
        return self.pool.memory_usage + self.retained_state_memory_usage

    def log_memory_usage(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Pool size: %s. Buffer memory: %s. Resident state: %s",
                         self.pool.active_count, self.pool.total_size,
                         prettify_bytes(self.retained_state_memory_usage))

    def _result_without_split(self):
        """Return the sink value after ``compute_xy`` executes a plan without a split."""
        logger.warning("X and y nodes not found in the DAG, returning the last node")
        sink = self.linearized_dag[-1]
        out = self.pool.pin(sink)
        self.pool.unpin(sink)
        return out

    def process_op(self, op: Op):
        """Execute an operator, retain live results and release values at last use.

        Data outputs use the buffer pool; fitted states use the state store.
        Fold index pairs have separate pool entries, selected by ``cv_id``.
        Return the operator after execution or reuse of a cached result.
        """
        if op.cache_hit is not None:
            # The cached result is already in the pool. Reuse planning removed
            # input edges, so execution and input release are unnecessary.
            return op
        logger.debug(f"[{perf_counter() - self.t0:.2f}s] Processing op: {op}")

        obs = self.observer
        node = op
        token = t1 = t2 = None
        pinned = []
        try:
            t0 = perf_counter()

            # 1. Pin data inputs and borrow fitted state from the state store.
            inputs = []
            for k, in_op in enumerate(op.inputs):
                if op.reads_state(k):
                    inputs.append(self.state_store.get(in_op))
                else:
                    key = (in_op, self.cv_id) if isinstance(in_op, SplitIndicesOp) else in_op
                    inputs.append(self.pool.pin(key))
                    pinned.append(key)

            # 2. Execute the selected operator implementation.
            if obs is not None:
                token = obs.before(node)
            t1 = perf_counter()
            result = op.process(inputs=inputs)
            t2 = perf_counter()

            # 3. Release input pins and borrowed references.
            for in_op in pinned:
                self.pool.unpin(in_op)
            pinned.clear()
            del inputs

            # 4. Release data and fitted state at their planned last uses.
            for in_op in op.remove_after:
                self.pool.remove(in_op)
            for in_op in op.release_state_after:
                self.state_store.remove(in_op)
            if op.is_split_op:
                self.pool.remove((op.indices, self.cv_id))

            # 5. Retain live data outputs and fitted states in their respective stores.
            if op.produces_state:
                if op.has_state_reader():
                    size = result.size if result.size is not None else get_size(result.state)
                    self.state_store.put(op, result.state, size=size)
                result = result.output
            # Operators read inputs and fitted states without mutating them, so
            # the cache can retain the same objects as this execution.
            if self.reuse_cache is not None and op.cache_key is not None:
                self.reuse_cache.put(op.cache_key, result, label=str(op))
            if not op.produces_state or op.has_data_reader() or not op.outputs:
                if isinstance(op, SplitIndicesOp):
                    result = self._store_split_indices(op, result)
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
                obs.after(node, op.mode, self.cv_id, token, t1, t2, buffer_overhead)

        except Exception as e:
            if obs is not None:
                obs.failed(node, op.mode, self.cv_id, token, t0, t1, t2)
            raise RuntimeError(f"[{op.mode}] Error processing '{op}': {e}")
        finally:
            for in_op in pinned:
                self.pool.unpin(in_op)

        return op

    def _store_split_indices(self, op: SplitIndicesOp, splits: list) -> int:
        """Store each fold under ``(op, fold_id)``; ``op`` holds only the count."""
        for i, indices in enumerate(splits):
            self.pool.put((op, i), indices)
        return len(splits)

    def _format_predict_result(self, rows):
        """The candidate-set sink's labelled rows as a frame."""
        return pl.DataFrame(rows)

    def _release_states(self, ops) -> None:
        """Release any fitted states remaining for these operators after a fold."""
        for op in ops:
            self.state_store.remove(op)


class SequentialScheduler(Scheduler):
    def __init__(self, linearized_dag, split_pos,
                 print_heavy_hitters=False, t0=None, reuse_cache=None):
        super().__init__(print_heavy_hitters, t0=t0, reuse_cache=reuse_cache)
        self.linearized_dag = linearized_dag
        self.pos_split_op = split_pos
        try:
            self._pool_cache_hits(linearized_dag)
        except BaseException:
            self._finish()
            raise

    def evaluate(self, seed: int = 42, test_size=0.2):
        """Evaluate the pipeline with a train/test split and return predictions."""

        try:
            result = self._evaluate(seed, test_size)
            return _copy_result(result) if self._shares_cache_values else result
        finally:
            self._finish()

    def _evaluate(self, seed, test_size):
        try:
            self.compute_xy(holdout=(seed, test_size))
        except RuntimeError as e:
            if "X and y nodes not found in the DAG" in str(e):
                return self._result_without_split()
            else:
                raise e

        rows = self.run_fold(0)
        return self._format_predict_result(rows)["vals"][0]

    def run_fold(self, cv_id: int):
        """Execute one fold using its stored indices and return the sink value."""
        self.cv_id = cv_id
        self.fold_ops = self.linearized_dag[self.pos_split_op:]
        try:
            return self.compute(self.fold_ops)
        finally:
            self._release_states(self.fold_ops)

    def compute(self, ops: list[Op]):
        """Execute ``ops`` in order and return the final operator's output."""
        for node in ops:
            self.process_op(node)

        sink = ops[-1]
        out = self.pool.pin(sink)
        self.pool.unpin(sink)
        self.pool.remove(sink)
        self.log_memory_usage()
        return out

    def compute_xy(self, cv=None, *, holdout=None) -> SplitOp:
        """Compute full-data nodes and fold indices before per-fold execution."""
        for op in self.linearized_dag:
            if op.is_split_op:
                return op
            if isinstance(op, SplitIndicesOp):
                op.cv, op.holdout = cv, holdout
                try:
                    self.process_op(op)
                finally:
                    op.cv = op.holdout = None
            else:
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
