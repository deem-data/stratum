from skrub._data_ops._estimator import _Splitter
from sklearn.model_selection import check_cv
from stratum.optimizer.logical._ops import OperandBinder, OperandRef, OutputType, Op, SplitXOp, _resolve_operand
from stratum.optimizer._op_utils import topological_iterator
from stratum.utils._utils import start_time, log_time
import pandas as pd
import polars as pl
import numpy as np


class SplitOp(Op):
    logical_family = "Split"

    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None):
        super().__init__(name="Train/Test", is_X=False, is_y=False, inputs=inputs, outputs=outputs)
        self.is_split_op = True
        self.output_type = OutputType.FRAME
        self.indices = None

    @property
    def splitter(self) -> "BuildSplitterOp | None":
        """The op computing the splitter the plan declares, or None if it declares none.

        Inputs are ``[X, y]``, plus the splitter when X is marked with a ``cv``.
        """
        return self.inputs[2] if len(self.inputs) > 2 else None

    def process(self, mode: str, inputs: list):
        # we need to handle both pandas and polars dfs
        x = inputs[0]
        y = inputs[1]
        if isinstance(x, pd.DataFrame):
            return (x.iloc[self.indices], y.iloc[self.indices])
        elif isinstance(x, pl.DataFrame):
            return (x[self.indices], y[self.indices])
        elif isinstance(x, np.ndarray):
            return (x[self.indices], y[self.indices])
        else:
            raise ValueError(f"Unsupported dataframe type: {type(x)}")


class SplitOutput(Op):
    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None, is_x = True, ):
        name = "X" if is_x else "y"
        super().__init__(name=name, is_X=False, is_y=False, inputs=inputs, outputs=outputs)
        self.is_x = is_x
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        # Subsetting rows keeps the columns. The SplitOp itself is an (X, y)
        # fan-out with no schema of its own, so read past it to the matching input
        # (inputs[0] = X, inputs[1] = y, per add_splitting_op).
        split_op = self.inputs[0]
        src = split_op.inputs[0 if self.is_x else 1]
        self.output_schema = src.output_schema

    def process(self, mode: str, inputs: list):
        if self.is_x:
            return inputs[0][0]
        else:
            return inputs[0][1]


class BuildSplitterOp(Op):
    """Build the splitter declared by ``mark_as_X(cv=..., split_kwargs=...)``.

    No data is computed here: the op binds the declared ``cv`` and ``split_kwargs`` into
    a splitter. Either may hold DataOps (``groups`` for ``GroupKFold`` usually does).
    They are nodes of the plan, so the splitter is built by it, before the split, rather
    than by evaluating them separately upfront. It feeds the split op, which keeps it in
    the pool for the scheduler to cut the folds with.
    """
    logical_family = "BuildSplitter"
    fields = ("cv", "split_kwargs")

    def __init__(self, cv, split_kwargs, inputs: list[Op]=None, outputs: list[Op]=None):
        name = None if isinstance(cv, OperandRef) else type(cv).__name__
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.cv = cv
        self.split_kwargs = split_kwargs

    def process(self, mode: str, inputs: list):
        cv = _resolve_operand(self.cv, inputs)
        split_kwargs = _resolve_operand(self.split_kwargs, inputs)
        # `split_kwargs` defaults to None when only `cv` is passed.
        return _Splitter(check_cv(cv), split_kwargs or {})


def _declared_splitter(x_op: Op) -> BuildSplitterOp | None:
    """The splitter the X mark declares, wired to the operands it reads, or None."""
    if not isinstance(x_op, SplitXOp):
        return None
    impl = x_op.skrub_impl
    # The mark's inputs are its DataOp fields, so every DataOp in `cv` or `split_kwargs`
    # is already an op of the plan. The splitter reads those same ops.
    ops = {data_op_id: x_op.inputs[k] for data_op_id, k in x_op.operand_index.items()}
    binder = OperandBinder(ops)
    splitter = BuildSplitterOp(cv=binder.bind(impl.cv), split_kwargs=binder.bind(impl.split_kwargs))
    splitter.inputs = binder.inputs
    for in_op in splitter.inputs:
        in_op.add_output(splitter)
    return splitter


def add_splitting_op(root: Op) -> Op:
    start = start_time()
    x_op = None
    y_op = None
    for op in topological_iterator(root):
        if op.is_X:
            x_op = op
        if op.is_y:
            y_op = op
        if x_op and y_op:
            # SplitX is skrub's pass-through mark for an X with a declared CV.
            # Keep it until the splitter has bound its operands, then make the
            # split read its X source directly. Other users of that source (for
            # example a groups expression) must continue reading the full data.
            splitter = _declared_splitter(x_op)
            x_source = x_op.source if isinstance(x_op, SplitXOp) else None
            split_out_x = SplitOutput(outputs=x_op.outputs)
            x_op.replace_input_of_outputs(split_out_x)
            split_out_y = SplitOutput(outputs=y_op.outputs, is_x=False)
            y_op.replace_input_of_outputs(split_out_y)
            split_op = SplitOp(inputs=[x_op, y_op], outputs=[split_out_x, split_out_y])
            split_out_x.inputs = [split_op]
            split_out_y.inputs = [split_op]
            x_op.outputs = [split_op]
            y_op.outputs = [split_op]
            if splitter is not None:
                split_op.add_input(splitter)
                splitter.add_output(split_op)
            if x_source is not None:
                split_op.replace_input(x_op, x_source)
                for in_op in x_op.inputs:
                    in_op.outputs = [out for out in in_op.outputs if out is not x_op]
                x_source.add_output(split_op)
                x_op.inputs = []
                x_op.outputs = []
            break
    log_time("splitting took", start)
    return root
