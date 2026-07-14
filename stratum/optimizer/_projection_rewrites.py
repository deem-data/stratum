from stratum.optimizer.ir._ops import Op
from stratum.optimizer.ir._dataframe_ops import DropOp
from stratum.optimizer._op_utils import rewrite_pass
from stratum.optimizer._numeric_rewrites import replace_two_op_chain

def _extract_drop_columns(op: DropOp):
    """
        Extracts a list of column names to be dropped from a DropOp node.
        Supports both `columns=['A']` and `df.drop(['A'], axis=1 / 'columns')` syntax.
        
        Returns a list of strings, or None if the operation doesn't target columns.
    """
    kwargs = op.kwargs or {}
    
    if "columns" in kwargs:
        cols = kwargs["columns"]
    elif kwargs.get("axis") in (1, "columns") and len(op.args) == 1:
        cols = op.args[0]
    else:
        return None
        
    return [cols] if isinstance(cols, str) else list(cols)


def match_consecutive_drop(op: Op):
    """
        Detects two back-to-back DropOp operations on columns.
        Example:
        df.drop(columns=['A']).drop(columns=['B'])
        
        Function returns (op, op2) / None:
        - op: the FIRST DropOp node (dropping cols1).
        - op2: the SECOND DropOp node (dropping cols2).
    """
    if (isinstance(op, DropOp) and len(op.outputs) == 1 and _extract_drop_columns(op) is not None):
        op2 = op.outputs[0]
        if isinstance(op2, DropOp) and _extract_drop_columns(op2) is not None:
            return (op, op2)
    return None


def fuse_consecutive_drop_action(op1: DropOp, op2: DropOp, root: Op) -> Op:
    """
        Merges two consecutive column drops into a single new DropOp.
        drop(cols1) -> drop(cols2) => single drop(cols1 | cols2)

        Order-preserving union of column lists is performed.
        The utility 'replace_two_op_chain' bypasses both op1 and op2, 
        splicing the new 'fused' node directly into the graph.
    """
    merged_columns = list(dict.fromkeys(
        _extract_drop_columns(op1) + _extract_drop_columns(op2)))
    fused = DropOp(kwargs={"columns": merged_columns}, inputs=[], outputs=[])
    replace_two_op_chain(op1, op2, fused)
    if op2 is root:
        root = fused
    return root

fuse_consecutive_drop = rewrite_pass(match_consecutive_drop, fuse_consecutive_drop_action)
