"""Extract mask selections after frame operations, before numeric rewrites and CSE."""
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._ops import GetItemOp, Op
from stratum.optimizer.logical._selection_ops import is_mask_selection, make_mask_selection_op


def extract_mask_selections(root: Op, *, fold: bool = True) -> Op:
    """Mark masked indexing and optionally fold its private predicate producers.

    Input kinds are supplied by frame extraction. With folding disabled, the
    GetItem retains its filter marker for backend selection and execution.
    """
    for op in list(topological_iterator(root)):
        if not isinstance(op, GetItemOp) or not is_mask_selection(op):
            continue
        op.is_filter = True
        if fold:
            selection = make_mask_selection_op(op)
            op.replace_input_of_outputs(selection)
            if root is op:
                root = selection
    return root
