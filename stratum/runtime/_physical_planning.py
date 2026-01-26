from skrub import StringEncoder, TableVectorizer
from stratum.logical_optimizer._op_utils import topological_iterator
from stratum.logical_optimizer._ops import EstimatorOp, Op, TransformerOp
from skrub._data_ops._data_ops import _wrap_estimator
from time import perf_counter
import uuid
import logging
logger = logging.getLogger(__name__)

def get_independent_set(ops: list[Op], ancestors: dict[Op]) -> list[Op]:
    # Find the largest subset of ops that don't depend on each other
    # Two ops conflict if one is an ancestor of the other
    def have_dependency(est1: Op, est2: Op) -> bool:
        """Check if est1 and est2 have a dependency (one is ancestor of the other)."""
        return est1 in ancestors.get(est2, set()) or est2 in ancestors.get(est1, set())

    # Greedily find the largest independent set
    # Sort by number of conflicts (fewer conflicts first) to maximize the set size
    conflict_counts = {est: sum(1 for other in ops if have_dependency(est, other))
                      for est in ops}

    # Sort by conflict count (ascending) - estimators with fewer conflicts are prioritized
    # prefer string encoder and table vectorizer over other estimators if they have the same conflict count
    sorted_ests = sorted(ops, key=lambda e: (conflict_counts[e], not (isinstance(e.estimator, StringEncoder) or isinstance(e.estimator, TableVectorizer))))

    # Greedily build the largest independent set
    independent_set = []
    for est in sorted_ests:
        # Check if this estimator conflicts with any already in the set
        if not any(have_dependency(est, added) for added in independent_set):
            independent_set.append(est)

    return independent_set

def mark_ops_for_parallelization(ops: list[Op], ancestors: dict[Op]):
    par_group_id = uuid.uuid4()
    selected_ops = get_independent_set(ops, ancestors)
    logger.debug(f"Selected {len(selected_ops)} ops for parallelization: {f"[{",".join(op.name for op in selected_ops)}]"}")
    for op in selected_ops:
        op.parallel_group = par_group_id


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
    t0 = perf_counter()
    ancestors = compute_ancestors(sink)

    estimators = [op for op in topological_iterator(sink) if isinstance(op, EstimatorOp)]
    transformers = [op for op in topological_iterator(sink) if isinstance(op, TransformerOp)]
    mark_ops_for_parallelization(estimators, ancestors)
    mark_ops_for_parallelization(transformers, ancestors)
    # make_parallel_block(estimators, ancestors)
    # make_parallel_block(transformers, ancestors)
    t1 = perf_counter()
    logger.info(f"Physical planning took: {t1 - t0:.2f} seconds")
    return sink