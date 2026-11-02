from __future__ import annotations
from time import perf_counter
from typing import Any
from sklearn.model_selection import train_test_split, check_cv
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._dataframe_ops import SplitOp
from stratum.optimizer._fold_compilation import FoldPlan, FoldProfile, Measured
from stratum.optimizer._lineage import state_key
from stratum.optimizer.logical._ops import Op
from stratum.runtime._buffer_pool import BufferPool
from stratum.runtime._object_size import get_size
from stratum.runtime._object_serialization import delete_object, deserialize_object
import numpy as np
import pandas as pd
import polars as pl

import logging
logger = logging.getLogger(__name__)


class Scheduler:
    """Scheduler for executing pre-planned Op DAGs in linearized order."""

    def __init__(self, print_heavy_hitters=False, t0=None, reuse_cache=None):
        self.linearized_dag = None
        self.pos_split_op: int | None = None
        #: The plan each fold runs a copy of, compiled once the split is reached.
        self.fold_plan: FoldPlan | None = None
        #: The copy the last fold ran, for inspection.
        self.fold_ops: list[Op] | None = None
        #: Compiles the fold plan, at the split and again after the first fold
        #: (see ``CompiledPlan.compile_folds``); None to run the plan as compiled.
        self._compile_folds = None
        #: What the fold running now measures, if it is measured; the pool's bytes
        #: when it started; and the bytes of the states it holds in the store.
        self._profile: FoldProfile | None = None
        self._profile_base = 0
        self._state_sizes: dict[Op, int] = {}
        self.timings = [] if print_heavy_hitters else None
        self.observer = None  # Inspection-only hooks, see stratum._profiler.
        self.results_ = None
        self.cv_id = -1
        self.pool = BufferPool()
        #: "fit"-stage op -> the state it fitted, until its last "predict" stage
        #: has applied it. Never spilled or evicted: a fold cannot finish without it.
        self.state_store: dict[Op, Any] = {}
        self.t0 = t0 if t0 is not None else perf_counter()
        self._pinned_ops: set[Op] = set()
        self.buffer_pool_overhead = 0
        #: Where buffers of ops flagged by reuse planning go once released.
        self.reuse_cache = reuse_cache
        #: op -> seconds to recompute its output from the plan's sources (or from
        #: cache hits, at the cost the entry took to compute). Kept with a cache only.
        self._lineage_cost: dict = {}

    def _pool_cache_hits(self, ops: list[Op]) -> None:
        """Put the values reuse planning took from the cache for ``ops`` into the
        pool, and the states into the state store."""
        for op in ops:
            hit = op.cache_hit
            if hit is None:
                continue
            if hit.has_output:
                self.pool.put(op, hit.take())
            if hit.has_state:
                self.state_store[op] = hit.take_state()
            self._lineage_cost[op] = hit.cost

    def _finish(self):
        """End of execution. Remove all buffers and states, handing flagged ones to
        the cache."""
        self._release_states(list(self.state_store))
        if self.reuse_cache is not None:
            for key in list(self.pool.live_variable_map):
                if getattr(key, "cache_key", None) is not None:
                    self._release(key)
        self.pool.remove_all()
        self.log_memory_usage()
        logger.debug(f"Scheduler finished: {self.pool.total_removed} buffers removed total")


    def grid_search(self, cv=None):
        """Perform grid search with cross-validation on the plan.

        An explicit ``cv`` takes precedence over the splitter the plan declares with
        ``mark_as_X(cv=...)``, as in skrub. With neither, the folds are ``KFold(5)``.
        """
        if cv is not None:
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
        if cv is None:
            cv = self._declared_splitter(split_op)

        results, predictions = [], []

        logger.debug("\n" + "="*100 + "\n" + "XY computed" + "\n" + "="*100 + "\n")
        results = self.cross_validate(split_op, cv, predictions, results,
                                      sink.emit_predictions)
        self.results_ = results
        self._finish()
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
                       return_predictions: bool):
        """Perform cross-validation on the plan, one scored row per candidate per fold.

        One run of the plan after the split is one fold: it fits on the fold's
        training rows and answers on its test rows (see fit/predict unrolling).
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

            # The first fold is measured, for the plan of the others to be compiled
            # again from what it measured.
            profile = (FoldProfile() if i == 0 and len(splits) > 1
                       and self._compile_folds is not None else None)
            fold = self.run_fold((self.pool.pin(train_ids_handle), self.pool.pin(test_ids_handle)),
                                 profile)
            for handle in (train_ids_handle, test_ids_handle):
                self.pool.unpin(handle)
                self.pool.remove(handle)
            if profile is not None:
                self._recompile_folds(profile)
            logger.debug("\n" + "="*100 + "\n" + "Fold " + str(i+1) + " done" + "\n" + "="*100 + "\n")
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

    def _result_without_split(self):
        """The sink's value, for a plan without X/y: ``compute_xy`` ran all of it."""
        logger.warning("X and y nodes not found in the DAG, returning the last node")
        sink = self.linearized_dag[-1]
        out = self.pool.pin(sink)
        shared = self.reuse_cache is not None and (
            sink.cache_key is not None or sink.cache_hit is not None)
        self._finish()
        # The cache holds the very object returned: give the caller its own.
        return _shielded(out) if shared else out

    def process_op(self, op: Op):
        """Process a single DataOp node and return its output."""
        if op.cache_hit is not None:
            # Its value is in the pool since the scheduler started, and it has no
            # inputs to release.
            return op
        logger.debug(f"[{perf_counter() - self.t0:.2f}s] Processing op: {op}")
        mode = op.mode

        obs = self.observer
        # A fold's copy of an op is profiled as the op of the plan it copies.
        node = op.instance_of or op
        token = t1 = t2 = None
        try:
            t0 = perf_counter()

            # 1. pin all inputs; a fitted state is read from the state store
            inputs = [self.state_store[in_op] if op.reads_state(k) else self.pool.pin(in_op)
                      for k, in_op in enumerate(op.inputs)]

            # 2. process operator
            if obs is not None:
                token = obs.before(node)
            t1 = perf_counter()
            result = op.process(mode=mode, inputs=inputs)
            t2 = perf_counter()

            # 3. unpin inputs
            for k, in_op in enumerate(op.inputs):
                if not op.reads_state(k):
                    self.pool.unpin(in_op)

            if self.reuse_cache is not None:
                # Shared inputs are counted once per path, so this overestimates a
                # diamond. It ranks entries for eviction; it is not an exact measure.
                self._lineage_cost[op] = (t2 - t1) + sum(
                    self._lineage_cost.get(in_op, 0.0) for in_op in op.inputs)

            # 4. remove unnecessary intermediates from buffer pool and state store
            if self._profile is not None:
                held = self.pool.memory_usage + sum(self._state_sizes.values())
            for in_op in op.remove_after:
                self._release(in_op)
            for in_op in op.release_state_after:
                self._state_sizes.pop(in_op, None)
                self._release_state(in_op, self.state_store.pop(in_op))

            # 5. add output to the buffer pool, and a fitted state to the store
            state = None
            if op.stage == "fit":
                state = result.state
                if op.has_state_reader():
                    self.state_store[op] = state
                else:
                    self._release_state(op, state)
                result = result.output
            if op.stage != "fit" or op.has_data_reader():
                self.pool.put(op, result)
            if self._profile is not None:
                self._measure(op, t2 - t1, state, held)
            self.log_memory_usage()

            t3 = perf_counter()
            process_duration = t2 - t1
            buffer_overhead = t3 - t0 - process_duration
            self.buffer_pool_overhead += buffer_overhead

            if self.timings is not None:
                duration = t2 - t1
                self.timings.append((str(op), process_duration))
            if obs is not None:
                obs.after(node, mode, self.cv_id, token, t1, t2, buffer_overhead)

        except Exception as e:
            if obs is not None:
                obs.failed(node, mode, self.cv_id, token, t0, t1, t2)
            raise RuntimeError(f"[{mode}] Error processing '{op}': {e}")

        return op

    def _measure(self, op: Op, seconds: float, state, held: int) -> None:
        """Record what running ``op`` measured in the profile of the fold.

        ``held`` is what the pool and the state store held while it ran, before its
        inputs were released: with its output and state, the fold's peak so far.
        """
        entry = self.pool.live_variable_map.get(op)
        measured = Measured(seconds, int(entry.size) if entry is not None else 0,
                            int(get_size(state)) if state is not None else 0)
        if op in self.state_store:
            self._state_sizes[op] = measured.state_size
        if op.origin is not None:
            self._profile.ops[op.origin] = measured
        self._profile.peak = max(self._profile.peak, held + measured.size
                                 + measured.state_size - self._profile_base)

    def _format_predict_result(self, rows):
        """The candidate-set sink's labelled rows as a frame."""
        return pl.DataFrame(rows)

    def _release(self, key) -> bool:
        """Drop a buffer the plan no longer needs, or hand it to the reuse cache.

        A flagged op's buffer moves into the cache instead of being freed, so the
        object is never held by both. A spilled buffer hands over its file; a
        container spill cannot move, so it is loaded and admitted in memory.
        """
        cache_key = getattr(key, "cache_key", None)
        if cache_key is None or self.reuse_cache is None:
            return self.pool.remove(key)
        entry = self.pool.detach(key)
        if entry is None:
            return False
        cost = self._lineage_cost.get(key, 0.0)
        label = str(key)
        if not entry.is_spilled:
            self.reuse_cache.put(cache_key, entry.data, cost, label)
        elif not self.reuse_cache.put_spilled(cache_key, entry.handle, entry.size, cost, label):
            if cache_key not in self.reuse_cache:
                self.reuse_cache.put(cache_key, deserialize_object(entry.handle), cost, label)
            delete_object(entry.handle)
        return True

    def _release_states(self, ops) -> None:
        """Release the states of ``ops`` still in the store. A plan releases each
        after its last reader; one left there has none that ran."""
        for op in ops:
            if op in self.state_store:
                logger.debug(f"Releasing the state of {op}, which nothing read")
                self._release_state(op, self.state_store.pop(op))

    def _release_state(self, op: Op, state) -> None:
        """Drop the state ``op`` fitted, or hand it to the reuse cache if flagged."""
        if self.reuse_cache is not None and op.cache_key is not None:
            self.reuse_cache.put(state_key(op.cache_key), state,
                                 self._lineage_cost.get(op, 0.0), f"{op} state")


def _shielded(value):
    """A copy of ``value`` its holder can mutate without touching the original.

    Cheap where the library allows: a shallow pandas copy is copy-on-write and a
    polars clone shares its buffers.
    """
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy(deep=False)
    if isinstance(value, (pl.DataFrame, pl.Series)):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


class SequentialScheduler(Scheduler):
    def __init__(self, linearized_dag, split_pos,
                 print_heavy_hitters=False, t0=None, reuse_cache=None, compile_folds=None):
        """``compile_folds``, when given, compiles the plan of a fold once the ops
        before the split have run (see ``CompiledPlan.compile_folds``). Without it,
        each fold runs a copy of ``linearized_dag`` from ``split_pos`` on."""
        super().__init__(print_heavy_hitters, t0=t0, reuse_cache=reuse_cache)
        self.linearized_dag = linearized_dag
        self.pos_split_op = split_pos
        self._compile_folds = compile_folds
        self._pool_cache_hits(linearized_dag)

    def evaluate(self, seed: int = 42, test_size=0.2):
        """Evaluate the pipeline with a train/test split and return predictions."""

        try:
            split_op = self.compute_xy()
        except RuntimeError as e:
            if "X and y nodes not found in the DAG" in str(e):
                return self._result_without_split()
            else:
                raise e

        x_data = self.pool.pin(split_op.inputs[0])
        indices = train_test_split(range(len(x_data)), test_size=test_size, random_state=seed)
        self.pool.unpin(split_op.inputs[0])
        rows = self.run_fold(tuple(indices))
        self._finish()
        return self._format_predict_result(rows)["vals"][0]

    def run_fold(self, indices, profile: FoldProfile | None = None):
        """Run a copy of the fold plan on the fold whose ``(train, test)`` row
        positions are ``indices``, and return its sink's value. ``profile``, if
        given, records what the run measures."""
        self.fold_ops = self.fold_plan.instantiate(indices)
        self._profile, self._profile_base = profile, self.pool.memory_usage
        self._pool_cache_hits(self.fold_ops)
        try:
            return self.compute(self.fold_ops)
        finally:
            self._profile = None
            self._state_sizes.clear()
            self._release_states(self.fold_ops)
            for op in self.fold_ops:
                if op.is_split_op:
                    op.indices = None
                # The copies are not run again.
                self._lineage_cost.pop(op, None)

    def compute(self, ops: list[Op]):
        """Run ``ops`` in order and return the value of the last, their sink."""
        for node in ops:
            self.process_op(node)

        sink = ops[-1]
        out = self.pool.pin(sink)
        self.pool.unpin(sink)
        self._release(sink)
        self.log_memory_usage()
        return out

    def compute_xy(self) -> SplitOp:
        """Compute nodes until the split op is reached, then compile the plan of a
        fold. Returns the split op of the plan as compiled up front."""
        for i, op in enumerate(self.linearized_dag):
            if op.is_split_op:
                self._enter_folds()
                return op
            self.process_op(op)
        raise RuntimeError("X and y nodes not found in the DAG")

    def _enter_folds(self) -> None:
        """Compile the fold plan; the plan the scheduler reports is the ops it ran
        before the split followed by it."""
        if self._compile_folds is not None:
            self._use_fold_plan(self._compile_folds())
        else:
            self._use_fold_plan(FoldPlan(self.linearized_dag[self.pos_split_op:]))

    def _recompile_folds(self, profile: FoldProfile) -> None:
        """Compile the fold plan again from what a fold measured; the next folds
        run what it gives."""
        plan = self._compile_folds(profile, self.fold_plan)
        if plan is not self.fold_plan:
            logger.debug("The plan of the remaining folds changed after the first")
            self._use_fold_plan(plan)

    def _use_fold_plan(self, plan: FoldPlan) -> None:
        self.fold_plan = plan
        self.linearized_dag = self.linearized_dag[:self.pos_split_op] + plan.ops
