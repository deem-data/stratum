"""Propagate output schemas through the logical DAG."""
from stratum.optimizer._op_utils import topological_iterator
from stratum.utils._utils import start_time, log_time


def propagate_output_schema(root, check: bool = False):
    """Propagate each op's output schema from its inputs, bottom-up.

    With ``check``, compare each existing schema to the recomputed one to detect
    stale schema metadata left by a rewrite. Skip comparisons for missing schemas.
    """
    start = start_time()
    for op in topological_iterator(root):
        before = op.output_schema
        op.propagate_output_schema()
        if check and before is not None and before != op.output_schema:
            raise AssertionError(
                f"stale output schema on {op!r}: {before} was set, but its inputs "
                f"give {op.output_schema}")
    log_time("schema_propagation took", start)
