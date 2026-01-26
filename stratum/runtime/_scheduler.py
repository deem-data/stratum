from time import perf_counter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, check_cv
from sklearn.metrics._scorer import _Scorer
from skrub._data_ops._data_ops import EvalMode
from stratum.logical_optimizer._dataframe_ops import SplitOp
from stratum.logical_optimizer._op_utils import show_graph, topological_iterator
from stratum.logical_optimizer._ops import EstimatorOp, ImplOp, Op
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import polars as pl
from stratum._config import FLAGS
import os

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
    
    def __init__(self, print_heavy_hitters=False):
        """Initialize scheduler with a data operations DAG."""
        self.mode = "fit_transform"
        self.env = {}
        self.flagged_for_recomputation = []
        self.pos_split_op = None
        self.timings = [] if print_heavy_hitters else None
        self.results_ = None

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

    def _format_predict_result(self, pred):
        """Helper method to format prediction results consistently."""
        if isinstance(pred, list):
            return pl.DataFrame(pred)
        else:
            return pl.DataFrame({"vals": [pred], "id": ["default"]})

    def _flag_op_for_recomputation_if_needed(self, op: Op):
        """Helper method to flag an op for recomputation if it's an ImplOp with EvalMode."""
        if isinstance(op, ImplOp) and isinstance(op.skrub_impl, EvalMode):
            self.flagged_for_recomputation.append(op)

class SequentialScheduler(Scheduler):
    def __init__(self, dag_sink: Op, print_heavy_hitters=False):
        super().__init__(print_heavy_hitters)
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
        pred = self.compute(self.pos_split_op, mode="predict")
        return pred["vals"][0]


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
            return self._format_predict_result(pred)
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

class ParallelScheduler(Scheduler):
    def __init__(self, dag_sink: Op, parallel_groups: dict[int, (int, list[Op])], print_heavy_hitters=False, backend="threading", max_workers=None):
        super().__init__(print_heavy_hitters)
        self.linearize_dag(dag_sink)
        self.backend = backend
        if max_workers is None:
            max_workers = os.cpu_count() or 8
        self.max_workers = max_workers

    def linearize_dag(self, dag_sink: Op):
        parallel_groups = {}
        for op in topological_iterator(dag_sink):
            if op.parallel_group is not None:
                group = parallel_groups.get(op.parallel_group, [])
                group.append(op)
                parallel_groups[op.parallel_group] = group
        groups_str = "\n".join("  ["+",".join(op.name for op in g) +"]" for g in parallel_groups.values()) #cant use f-string because of py3.11
        logger.debug(f"Parallel groups:\n{groups_str}\n")
        for group in parallel_groups.values():
            inputs_union = set()
            for op in group:
                inputs_union.update(op.inputs)
            for op in group:
                # add additional dependencies s.t. all ops in the group are ready to compute
                for in_ in inputs_union:
                    if in_ not in op.inputs:
                        if op.additional_inputs is None:
                            op.additional_inputs = []
                        op.additional_inputs.append(in_)
                        if in_.additional_outputs is None:
                            in_.additional_outputs = []
                        in_.additional_outputs.append(op)
        if FLAGS.DEBUG:
            show_graph(dag_sink, "parallel_process_plan")

        blocks = []
        group_added = {}
        for op in topological_iterator(dag_sink):

            if op.parallel_group is None:
                blocks.append(op)
            else:
                group = parallel_groups[op.parallel_group]
                if not group_added.get(op.parallel_group, False):
                    blocks.append(group)
                    group_added[op.parallel_group] = True
                
                
        self.blocks = blocks
    
    def compute(self, start_pos: int, mode="fit_transform"):
        """Compute the pipeline from start_pos onwards with given inputs."""
        blocks_to_compute = self.blocks[start_pos:]
        if len(self.flagged_for_recomputation) != 0:
            # Add flagged ops as individual blocks before the rest
            blocks_to_compute = [op for op in self.flagged_for_recomputation] + blocks_to_compute
        self.mode = mode

        for block in blocks_to_compute:
            self.process_block(block)

        if mode == "predict":
            # Get the last block's output
            last_block = self.blocks[-1]
            return self._format_predict_result(last_block.intermediate)
        return None

    def compute_xy(self) -> SplitOp:
        """Compute blocks until X and y nodes are found and store them."""
        for i, block in enumerate(self.blocks):
            if block.is_split_op:
                self.pos_split_op = i
                return block
            self.process_block(block)
            self._flag_op_for_recomputation_if_needed(block)
        raise RuntimeError("X and y nodes not found in the DAG")

    def process_block(self, block):
        """Process a single block - either an Op or a list of Ops (parallel group)."""
        if isinstance(block, list):
            # Parallel group - process ops in parallel
            ops = block
            logger.debug(f"Processing parallel block with {len(ops)} ops")
            t0 = perf_counter() if self.timings is not None else 0
            
            if self.backend == "process" or (self.backend == "auto" and all(isinstance(op, EstimatorOp) for op in ops)):
                logger.debug(f"Using process-based parallel processing with joblib)")
                results = Parallel(n_jobs=len(ops))(
                    delayed(op.get_process_task())(op.extract_args_from_inputs(self.mode)) 
                    for op in ops
                )
                
                for i, (result, fitted_estimator) in enumerate(results):
                    ops[i].intermediate = result
                    ops[i].estimator = fitted_estimator
            else:
                logger.debug(f"Using thread-based parallel processing with ThreadPoolExecutor")
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(self._process_op_task, op, self.mode, self.env) for op in ops]
                for i, future in enumerate(futures):
                    ops[i].intermediate = future.result()
            
            if self.timings is not None:
                duration = perf_counter() - t0
                self.timings.append((f"ParallelBlock({len(ops)} ops)", duration))
        else:
            # Single op - process sequentially
            self.process_op(block)

    def _process_op_task(self, op: Op, mode: str, environment: dict):
        """Helper task for thread-based parallel processing."""
        try:
            op.process(mode=mode, environment=environment)
            return op.intermediate
        except Exception as e:
            raise RuntimeError(f"[{mode}] Error processing '{op}': {e}")