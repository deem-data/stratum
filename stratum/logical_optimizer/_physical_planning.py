from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from stratum.logical_optimizer._op_utils import topological_iterator
from stratum.logical_optimizer._ops import EstimatorOp, Op, DATA_OP_PLACEHOLDER, process_estimator_task
from stratum.logical_optimizer._dataframe_ops import SplitOutput
from skrub._data_ops._data_ops import _wrap_estimator
from pandas import DataFrame
from polars import DataFrame as PlDataFrame, Series as PlSeries
import logging
import numpy as np
logger = logging.getLogger(__name__)
from time import perf_counter

def task(op: Op, mode: str, environment: dict):
    """Thread-based task (for when process_based=False)."""
    t0 = perf_counter()
    op.process(mode, environment)
    t1 = perf_counter()
    logger.debug(f"Parallel Task: {op.name} - Time taken: {t1 - t0} seconds")
    return op.intermediate    

class ParallelBlockOp(Op):
    def __init__(self, estimators: list[Op]):
        super().__init__(name=",".join([est_op.name for est_op in estimators]))
        self.estimators = estimators
        # allocate np array for estiamtors outputs, use dtype pointer
        self.intermediate = np.empty((len(estimators)), dtype=object)
        self.process_based = True
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.estimators)) if not self.process_based else None

    def process(self, mode: str, environment: dict):
        """
        Process the parallel block of estimators.
        
        Args:
            mode: Processing mode ('fit_transform' or 'predict')
            environment: Environment dict (not used in process-based mode)
        """
        if self.process_based:
            # use process-based parallel processing of the EstimatorOps with joblib
            # This avoids GIL limitations since estimators often don't release the GIL
            # Extract picklable data from each estimator_op before sending to workers
            
            task_data_list = [estimator_op.extract_args_from_inputs(mode) 
                             for estimator_op in self.estimators]
            
            results = Parallel(n_jobs=len(self.estimators))(
                delayed(process_estimator_task)(task_data) 
                for task_data in task_data_list
            )
            
            for i, (result, fitted_estimator) in enumerate(results):
                # Store the result in intermediate
                self.intermediate[i] = result
                self.estimators[i].estimator = fitted_estimator
        else:
            # use thread pool parallel processing of the EstimatorOps
            futures = [self.thread_pool.submit(task, estimator_op, mode, environment) for estimator_op in self.estimators]
            for i, future in enumerate(futures):
                self.intermediate[i] = future.result()


class ParBlockOut(Op):
    def __init__(self, name: str, position: int, inputs: list[Op], outputs: list[Op]):
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.position = position

    def process(self, mode: str, environment: dict):
        self.intermediate = self.inputs[0].intermediate[self.position]

def make_parallel_block(op: Op, ancestors: dict[Op]) -> ParallelBlockOp:
    # Get all estimator outputs
    est_ops = [out_ for out_ in op.outputs if isinstance(out_, EstimatorOp)]
    
    # Find the largest subset of estimators that don't depend on each other
    # Two estimators conflict if one is an ancestor of the other
    def have_dependency(est1: Op, est2: Op) -> bool:
        """Check if est1 and est2 have a dependency (one is ancestor of the other)."""
        return est1 in ancestors.get(est2, set()) or est2 in ancestors.get(est1, set())
    
    # Greedily find the largest independent set
    # Sort by number of conflicts (fewer conflicts first) to maximize the set size
    conflict_counts = {est: sum(1 for other in est_ops if have_dependency(est, other)) 
                      for est in est_ops}
    
    # Sort by conflict count (ascending) - estimators with fewer conflicts are prioritized
    sorted_ests = sorted(est_ops, key=lambda e: conflict_counts[e])
    
    # Greedily build the largest independent set
    independent_set = []
    for est in sorted_ests:
        # Check if this estimator conflicts with any already in the set
        if not any(have_dependency(est, added) for added in independent_set):
            independent_set.append(est)
    
    est_ops = independent_set

    pblock =  ParallelBlockOp(est_ops)
    pblock_outs = []
    pblock_inputs = set()
    for i, estimator_op in enumerate(est_ops):
        for in_ in estimator_op.inputs:
            if in_ in pblock_inputs:
                # we already added this input to the parallel block
                in_.outputs.remove(estimator_op)
            else:
                in_.replace_output(estimator_op, pblock)
                pblock_inputs.add(in_)

        model_name = estimator_op.estimator.__class__.__name__
        out_ = ParBlockOut(model_name, i, [pblock], estimator_op.outputs)
        pblock_outs.append(out_)
        estimator_op.replace_input_of_outputs(out_)

    pblock.outputs = pblock_outs
    pblock.inputs = list(pblock_inputs)



def compute_ancestors(sink: Op) -> dict[Op]:
    """ Compute the ancestors of each op in the DAG. """
    ancestors = {op: set() for op in topological_iterator(sink)}
    for op in topological_iterator(sink):
        ancestors[op] = set()
        for in_ in op.inputs:
            ancestors[op].update(ancestors[in_])
            ancestors[op].add(in_)
    return ancestors

def physical_planning(sink: Op) -> Op:
    """ Apply physical planning to the DAG. """
    done = False
    ancestors = compute_ancestors(sink)
    while not done:
        done = True
        for op in topological_iterator(sink):
            #not isinstance(op, SplitOutput)
            if len(op.outputs) > 1 and all(isinstance(out_, EstimatorOp) for out_ in op.outputs):
                # we can paralleize the estimators
                logger.debug(f"Detected {len(op.outputs)} potential estimators to run in parallel after op {op}")
                make_parallel_block(op, ancestors)
                done = False
                break
    return sink