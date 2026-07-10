from stratum.optimizer.ir._ops import Op, GetItemOp
from stratum.optimizer._op_utils import rewrite_pass, replace_op_in_outputs


def _is_list_of_column_labels(key) -> bool:
    """ 
        True for non-empty list of column labels. 
    """
    if not isinstance(key, list) or len(key) == 0:
        return False
    return not any(isinstance(c, bool) for c in key)


def match_consecutive_select(op: Op):
    """ 
        Detects df[cols1][cols2] where both keys are non-empty list of column labels 
        and 'cols2' only keeps columns already present in 'cols1'.
        Example:
        'cols2' = set(['A', 'C']) <= 'cols1' = set(['A', 'B', 'C']) -> True

        Function returns (op, x) / None:
        - op: first select (cols1) which will be eliminated.
        - x: the input to the first select (the base DataFrame). 
    """
    if (isinstance(op, GetItemOp) and _is_list_of_column_labels(op.key) and len(op.outputs) == 1):
        op2 = op.outputs[0]
        if (isinstance(op2, GetItemOp) and _is_list_of_column_labels(op2.key) and set(op2.key) <= set(op.key)):
            return (op, op.inputs[0])
        
    return None


def eliminate_redundant_select_action(op: Op, x: Op, root: Op) -> Op:
    """
        select(cols1) -> select(cols2) with cols2 subset of cols1: 
        drop the op - select(cols1), keep select(cols2) applied directly to DataFrame (x). 
        Wherever the first select (op) was the input, insert the DataFrame (x) directly.
    """
    x.outputs = [out for out in x.outputs if out is not op]
    replace_op_in_outputs(op, x)
    if op is root:
        root = x
    return root

fuse_consecutive_select = rewrite_pass(match_consecutive_select, eliminate_redundant_select_action)
