"""Train/test split operators and their insertion into the plan.

``add_splitting_op`` connects the marked features (``X``) and targets (``y``)
to a shared ``SplitOp``. ``SplitIndicesOp`` computes the folds before per-fold
execution. ``SplitOp`` applies the current fold's indices to both inputs, and
``SplitOutput`` exposes the feature and target subsets separately.

``BuildSplitterOp`` constructs a splitter declared through
``mark_as_X(cv=..., split_kwargs=...)``. This keeps dependencies such as
group labels in the plan so they are computed before splitting.
"""

from skrub._data_ops._estimator import _Splitter
from sklearn.model_selection import check_cv, train_test_split
from stratum.frontend._conversion import OperandBinder
from stratum.optimizer.logical._ops import OperandRef, OutputType, Op, SplitXOp, _resolve_operand
from stratum.optimizer._op_utils import topological_iterator
from stratum.utils._utils import start_time, log_time
import pandas as pd
import polars as pl
import numpy as np


class SplitIndicesOp(Op):
    """Compute train/test row positions once for all folds.

    Inputs are ``[X, y]`` and optionally a declared splitter. An explicit ``cv``
    overrides that splitter; without either, use ``KFold(5)``. ``holdout`` selects
    one train/test split with ``(seed, test_size)`` for evaluation instead.

    Return a list of index pairs without retaining the splitter iterator. All
    indices are materialized during this call; afterward the runtime stores
    each pair separately and binds the current fold to readers.
    """
    logical_family = "SplitIndices"

    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.cv = None
        self.holdout = None

    def process(self, inputs: list):
        x, y = inputs[:2]
        if self.holdout is not None:
            seed, test_size = self.holdout
            return [tuple(train_test_split(range(len(x)), test_size=test_size,
                                           random_state=seed))]
        if self.cv is not None or len(inputs) == 2:
            cv = check_cv(self.cv)
        else:
            cv = inputs[2]
        return list(cv.split(x, y))


class SplitOp(Op):
    """Partition ``X`` and ``y`` into a fold's training and test subsets.

    Inputs are ``[X, y, indices]``, where the runtime binds the indices input to
    the current fold's ``(train, test)`` row positions. The output is the tuple
    ``(X_train, y_train, X_test, y_test)``. Each :class:`SplitOutput` operator
    extracts one component.
    """
    logical_family = "Split"

    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None):
        super().__init__(name="Train/Test", is_X=False, is_y=False, inputs=inputs, outputs=outputs)
        self.is_split_op = True
        self.output_type = OutputType.FRAME

    @property
    def indices(self) -> SplitIndicesOp:
        """The producer of the fold indices."""
        return self.inputs[2]

    def process(self, inputs: list):
        x, y = inputs[0], inputs[1]
        train, test = inputs[2]
        return (*_take(x, y, train), *_take(x, y, test))


def _take(x, y, rows) -> tuple:
    # Row positions require iloc on pandas and direct indexing on Polars/NumPy.
    if isinstance(x, pd.DataFrame):
        return x.iloc[rows], y.iloc[rows]
    elif isinstance(x, (pl.DataFrame, np.ndarray)):
        return x[rows], y[rows]
    else:
        raise ValueError(f"Unsupported dataframe type: {type(x)}")


class SplitOutput(Op):
    """``X`` or ``y`` of the training or the test rows of a fold."""

    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None, is_x = True, train=False):
        name = ("X" if is_x else "y") + ("_train" if train else "_test")
        super().__init__(name=name, is_X=False, is_y=False, inputs=inputs, outputs=outputs)
        self.is_x = is_x
        self.train = train
        self.output_type = OutputType.FRAME

    def clone(self):
        """Clone the output selection, preserving its train/test assignment.

        Train/test unrolling separately sets ``train=True`` on the training copy.
        """
        new_op = SplitOutput(is_x=self.is_x, train=self.train)
        new_op.was_cloned = True
        return new_op

    def propagate_output_schema(self):
        # Row selection preserves columns. Propagate the schema from the matching
        # SplitOp input: inputs[0] is X and inputs[1] is y.
        split_op = self.inputs[0]
        src = split_op.inputs[0 if self.is_x else 1]
        self.output_schema = src.output_schema

    def process(self, inputs: list):
        return inputs[0][(0 if self.train else 2) + (0 if self.is_x else 1)]

class BuildSplitterOp(Op):
    """Build the splitter declared by ``mark_as_X(cv=..., split_kwargs=...)``.

    Bind ``cv`` and ``split_kwargs`` to a splitter. Graph-fed arguments, such as
    ``groups`` for ``GroupKFold``, remain explicit dependencies and execute before
    splitter construction. ``SplitIndicesOp`` consumes the splitter before
    per-fold execution, after which the runtime can release it.
    """
    logical_family = "BuildSplitter"
    fields = ("cv", "split_kwargs")

    def __init__(self, cv, split_kwargs, inputs: list[Op]=None, outputs: list[Op]=None):
        name = None if isinstance(cv, OperandRef) else type(cv).__name__
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.cv = cv
        self.split_kwargs = split_kwargs

    def process(self, inputs: list):
        cv = _resolve_operand(self.cv, inputs)
        split_kwargs = _resolve_operand(self.split_kwargs, inputs)
        # `split_kwargs` defaults to None when only `cv` is passed.
        return _Splitter(check_cv(cv), split_kwargs or {})


def _declared_splitter(x_op: Op) -> BuildSplitterOp | None:
    """Build the declared splitter with bound input operands, or return None."""
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
            if x_source is not None:
                split_op.replace_input(x_op, x_source)
                for in_op in x_op.inputs:
                    in_op.outputs = [out for out in in_op.outputs if out is not x_op]
                x_source.add_output(split_op)
                x_op.inputs = []
                x_op.outputs = []
            indices = SplitIndicesOp(inputs=list(split_op.inputs), outputs=[split_op])
            if splitter is not None:
                indices.add_input(splitter)
            for in_op in indices.inputs:
                in_op.add_output(indices)
            split_op.add_input(indices)
            break
    log_time("splitting took", start)
    return root
