from __future__ import annotations
from types import SimpleNamespace
from typing import Callable

from sklearn import clone
from sklearn.base import BaseEstimator
from skrub._data_ops._choosing import Choice
from skrub._data_ops._data_ops import DataOp, Apply, Value, CallMethod, Call, GetAttr, GetItem, BinOp as SkrubBinOp, _wrap_estimator
from pandas import DataFrame
from polars import DataFrame as PlDataFrame, Series as PlSeries

class PlaceHolder():
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

# unique identifier for arguments, which need to be replaced with Op references later
DATA_OP_PLACEHOLDER = PlaceHolder("DATA_OP_PLACEHOLDER")

class Op():
    def __init__(self, inputs=None,outputs=None, name=None, is_X=False, is_y=False):
        self.name = name
        self.outputs = outputs if outputs is not None else []
        self.inputs = inputs if inputs is not None else []
        self.intermediate = None
        self.is_X = is_X
        self.is_y = is_y
        self.is_dataframe_op = False
        self.is_split_op = False
        self.was_cloned = False

    def to_str_helper(self):
        class_name = self.__class__.__name__
        is_df = " [df]" if self.is_dataframe_op else ""
        name = f"({self.name})" if self.name and len(self.name) > 0 else ""
        return class_name, name, is_df

    def __str__(self):
        return "".join(self.to_str_helper())
    
    def __repr__(self):
        class_name, name, is_df = self.to_str_helper()
        return f"{class_name}{name}[cloned={self.was_cloned}, id={id(self)}{is_df}]"

    def update_name(self):
        pass

    def has_outputs(self) -> bool:
        return self.outputs is not None and len(self.outputs) > 0

    def is_choice(self) -> bool:
        return isinstance(self, ChoiceOp)

    def add_output(self, output: Op):
        self.outputs.append(output)

    def add_input(self, input: Op):
        self.inputs.append(input)

    def replace_input(self, old_input: Op, new_input: Op):
        for i, in_ in enumerate(self.inputs):
            if in_ is old_input:
                self.inputs[i] = new_input
                return
        raise ValueError(f"Input {old_input} not found in {self.__class__.__name__}.")

    def replace_input_of_outputs(self, new_input):
        for out in self.outputs:
            out.replace_input(self, new_input)

    def replace_output(self, old_output: Op, new_output: Op):
        for i, out_ in enumerate(self.outputs):
            if out_ is old_output:
                self.outputs[i] = new_output
                return
        raise ValueError(f"Output {old_output} not found in {self.__class__.__name__}.")

    def replace_output_of_inputs(self, new_output):
        for in_ in self.inputs:
            in_.replace_output(self, new_output)

    def clone(self):
        if getattr(self.__class__, "fields", None) is None:
            raise NotImplementedError(f"Cloning of {self.__class__.__name__} objects is not implemented yet. Please implement it.")
        args, atts = self.__class__.fields, self.__dict__.items()
        fields = {k: clone_value(v) for k,v in atts if k in args}
        new_op = self.__class__(**fields)
        new_op.was_cloned = True
        return new_op

    def process(self, mode: str, environment: dict):
        raise NotImplementedError(f"Processing of {self.__class__.__name__} objects is not implemented yet. Please implement it.")

    def check_kwargs(self, kwargs):
        if not isinstance(kwargs, dict):
            raise TypeError(
                f"The `{self}'s kwargs` should be a dict of named arguments. Got an object of type"
                f" {type(kwargs).__name__!r} instead: {kwargs!r}"
            )

def clone_value(value):
    if isinstance(value, dict):
        return {k:clone_value(v) for k,v in value.items()}
    elif isinstance(value, tuple):
        return tuple(clone_value(el) for el in value)
    else:
        return value

class ImplOp(Op):
    def __init__(self, name: str, skrub_impl):
        super().__init__(name=name)
        self.skrub_impl = skrub_impl

    def clone(self):
        attributes = {}
        for att in self.skrub_impl._fields:
            attributes[att] = getattr(self.skrub_impl, att)
        new_impl = self.skrub_impl.__class__(**attributes)
        new_op = ImplOp(name=self.name, skrub_impl=new_impl)
        new_op.was_cloned = True
        return new_op

    def replace_fields_with_values(self):
        """Replace DataOp fields in implementation with their computed values."""
        parent_iter = iter(self.inputs)

        def replace_dataop(value):
            """Recursively replace DataOp instances with their actual values."""
            if isinstance(value, DataOp):
                return next(parent_iter).intermediate
            elif isinstance(value, (list, tuple)):
                new_seq = [replace_dataop(item) for item in value]
                return type(value)(new_seq)
            elif isinstance(value, dict):
                return {key: replace_dataop(val) for key, val in value.items()}
            else:
                return value

        return SimpleNamespace(**{field: replace_dataop(getattr(self.skrub_impl, field)) for field in self.skrub_impl._fields})

    def process(self, mode: str, environment: dict):
        if hasattr(self.skrub_impl, "eval"):
            # DataOp with eval method have a fused implementation of the generator and the compute method
            # we need to iterate over the generator and replace the requested fields with correct inputs
            last_yield = None
            gen = self.skrub_impl.eval(mode=mode, environment=environment)
            parent_iter = iter(self.inputs)
            while True:
                try:
                    last_yield = gen.send(last_yield)
                except StopIteration as e:
                    self.intermediate = e.value
                    break
                if isinstance(last_yield, DataOp):
                    last_yield = next(parent_iter).intermediate
        else:
            ns = self.replace_fields_with_values()
            self.intermediate = self.skrub_impl.compute(ns, mode, environment)

class EstimatorOp(Op):
    fields = ["estimator", "y", "cols", "how", "allow_reject", "unsupervised", "kwargs"]
    
    def __init__(self, estimator: BaseEstimator, y=None, cols=None, how="no-wrap", allow_reject=False, unsupervised=False, kwargs=None):
        super().__init__(name=estimator.__class__.__name__)
        if kwargs is None:
            kwargs = {}
        self.check_kwargs(kwargs)
        self.estimator = estimator
        self.y = DATA_OP_PLACEHOLDER if isinstance(y, DataOp) else y
        self.cols = DATA_OP_PLACEHOLDER if isinstance(cols, DataOp) else cols
        self.how = how
        self.allow_reject = allow_reject
        self.unsupervised = unsupervised
        self.kwargs = remove_datops_from_args(kwargs) if kwargs is not None else kwargs

    def clone(self):
        params = self.estimator.get_params()
        estimator_new = clone(self.estimator)
        estimator_new.set_params(**params)
        new_op = EstimatorOp(
            estimator=estimator_new, 
            y=self.y, 
            cols=self.cols, 
            how=self.how, 
            allow_reject=self.allow_reject, 
            unsupervised=self.unsupervised, 
            kwargs=self.kwargs
        )
        new_op.was_cloned = True
        return new_op

    def extract_args_from_inputs(self, mode: str):
        """
        Extract all necessary data from an EstimatorOp to make it picklable for multiprocessing.
        
        Returns a tuple of picklable data that can be sent to worker processes.
        """
        input_iter = iter(self.inputs)
        x = next(input_iter).intermediate
        
        if self.y == DATA_OP_PLACEHOLDER:
            y = next(input_iter).intermediate
        else:
            y = self.y

        if mode == "predict":
            y = None
        
        if self.cols == DATA_OP_PLACEHOLDER:
            cols = next(input_iter).intermediate
        else:
            cols = self.cols
        
        return (
            self.estimator,
            x,
            y,
            cols,
            self.how,
            self.allow_reject,
            self.unsupervised,
            self.kwargs,
            mode,
        )
    
    def process(self, mode: str, environment: dict):
        # we use a separate function to process the estimator to allow reuse for multiprocessing
        task_data = self.extract_args_from_inputs(mode)
        self.intermediate, self.estimator = process_estimator_task(task_data)

def process_estimator_task(task_data):
    """ Process an estimator task in a worker process. """
    (estimator, x, y, cols, how, allow_reject, unsupervised, kwargs, mode) = task_data
    
    if mode == "fit_transform":
        estimator = _wrap_estimator(estimator, cols, how=how, allow_reject=allow_reject, X=x)
        y_arg = () if unsupervised else (y,)
        if not hasattr(estimator, mode):
            # Predictors
            estimator.fit(x, *y_arg, **kwargs)
            result = estimator.predict(x, **kwargs)
        else:
            # Transformers
            result = estimator.fit_transform(x, *y_arg, **kwargs)
        # Return both result and fitted estimator (in case of multi-processing)
        return (result, estimator)
    elif mode == "predict":
        if not hasattr(estimator, mode):
            # Transformers
            result = estimator.transform(x, **kwargs)
        else:
            # Predictors
            result = estimator.predict(x, **kwargs)
        return (result, estimator)
    else:
        raise ValueError(f"Mode {mode} not supported for EstimatorOp.")


class ChoiceOp(Op):
    fields = ["outcome_names"]
    
    def __init__(self, outcome_names: list[str] = None, n_outcomes: int = None, choice_name: str=None, append_choice_name = True, inputs: list[Op] = None):
        if inputs is None:
            inputs = []
        if outcome_names is None:
            outcome_names = [[(choice_name, f"Opt{i}")] for i in range(n_outcomes)]
        elif append_choice_name:
            outcome_names = [[(choice_name, name)] for name in outcome_names]
        else:
            outcome_names = outcome_names
        super().__init__(inputs=inputs)
        self.outcome_names = outcome_names
        self.update_name()

    def make_outcome_names(self):
        # TODO find a better way for naming the unnamed choices
        return [", ".join(
                f"Choice{len(combi) - i - 1}:{value}" if choice_name is None else f"{choice_name}:{value}"
                for i, (choice_name, value) in enumerate(combi)
            ) for combi in self.outcome_names]

    def update_name(self):
        self.name = "  |  ".join(self.make_outcome_names())

    def clone(self):
        new_op = ChoiceOp(outcome_names=self.outcome_names, append_choice_name=False)
        new_op.name = self.name
        new_op.was_cloned = True
        return new_op

    def process(self, mode: str, environment: dict):
        results = [{"id" : name, "vals" : self.inputs[i].intermediate} for i, name in enumerate(self.make_outcome_names())]
        self.intermediate = results[0] if len(results) == 1 else results

class ValueOp(Op):
    fields = ["value"]
    
    def __init__(self, value):
        super().__init__(name="DataFrame" if isinstance(value, DataFrame) else str(value))
        self.value = value

    def clone(self):
        raise ValueError(f"We should not clone ValueOp objects.")

    def process(self, mode: str, environment: dict):
        self.intermediate = self.value

class MethodCallOp(Op):
    fields = ["method_name", "args", "kwargs"]
    
    def __init__(self, method_name: str, args = None, kwargs = None):
        super().__init__(name=method_name)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.method_name = method_name
        self.args = remove_datops_from_args(args) if args is not None else args
        self.kwargs = remove_datops_from_args(kwargs) if kwargs is not None else kwargs

    def process(self, mode: str, environment: dict):
        iter_ins = iter(self.inputs)
        _obj = next(iter_ins).intermediate
        _args = [next(iter_ins).intermediate if arg is DATA_OP_PLACEHOLDER else arg for arg in self.args]
        _kwargs = {k: next(iter_ins).intermediate if v is DATA_OP_PLACEHOLDER else v for k, v in self.kwargs.items()}
        self.intermediate = _obj.__getattribute__(self.method_name)(*_args, **_kwargs)

class CallOp(Op):
    fields = ["func", "args", "kwargs"]
    
    def __init__(self, name: str = "CallOp", func=None, args=None, kwargs=None):
        super().__init__(name=name)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.func = func
        self.args = remove_datops_from_args(args) if args is not None else args
        self.kwargs = remove_datops_from_args(kwargs) if kwargs is not None else kwargs

    def process(self, mode: str, environment: dict):
        iter_ins = iter(self.inputs)
        _args = [next(iter_ins).intermediate if arg is DATA_OP_PLACEHOLDER else arg for arg in self.args]
        _kwargs = {k: next(iter_ins).intermediate if v is DATA_OP_PLACEHOLDER else v for k, v in self.kwargs.items()}
        self.intermediate = self.func(*_args, **_kwargs)

class GetAttrOp(Op):
    fields = ["attr_name"]
    
    def __init__(self, attr_name: str=None):
        super().__init__(name=attr_name if attr_name else '?')
        self.attr_name = attr_name

    def process(self, mode: str, environment: dict):
        if self.is_dataframe_op:
            self.intermediate = self.inputs[0].intermediate
            for attr in self.attr_name:
                self.intermediate = self.intermediate.__getattribute__(attr)
        else:
            self.intermediate = self.inputs[0].intermediate.__getattribute__(self.attr_name)

class GetItemOp(Op):
    fields = ["key"]
    
    def __init__(self, key=None):
        super().__init__(name=str(key) if key is not None else '?')
        self.key = key

    def process(self, mode: str, environment: dict):
        self.intermediate = self.inputs[0].intermediate[self.key]

class BinOp(Op):
    fields = ["op", "left", "right"]
    
    def __init__(self, op: Callable, left, right):
        super().__init__(name=op.__name__.lstrip('__').rstrip('__'))
        self.op = op
        self.left = DATA_OP_PLACEHOLDER if isinstance(left, DataOp) else left
        self.right = DATA_OP_PLACEHOLDER if isinstance(right, DataOp) else right


    def process(self, mode: str, environment: dict):
        i = 0
        if self.left is DATA_OP_PLACEHOLDER:
            left = self.inputs[i].intermediate
            i += 1
        else:
            left = self.left
        if self.right is DATA_OP_PLACEHOLDER:
            right = self.inputs[i].intermediate
            i += 1
        else:
            right = self.right
        self.intermediate = self.op(left, right)

class SearchEvalOp(Op):    
    def __init__(self, outcome_names: list[str], parent: Op = None):
        super().__init__()
        self.name = "evaluate gridsearch" 
        self.outcome_names = outcome_names
        self.parents = [] if parent is None else [parent]
        self.children = []

    def clone(self, children: list[Op] = None, parents: list[Op] = None):
        raise ValueError(f"We should not clone SearchEvalOp objects.")

def remove_datops_from_args(args: tuple  | dict):
    if isinstance(args, tuple):
        return tuple(DATA_OP_PLACEHOLDER if isinstance(a, DataOp) else a for a in args)
    elif isinstance(args, dict):
        return {k: DATA_OP_PLACEHOLDER if isinstance(v, DataOp) else v for k,v in args.items()}
    else:
        raise ValueError(f"Expected tuple or dict, got {type(args)}")

def as_op(data_op: DataOp):
    impl = data_op._skrub_impl
    is_X = False
    is_y = False
    if impl is not None:
        is_X = impl.is_X
        is_y = impl.is_y
    return_op = None
    if isinstance(impl, Value):
        if isinstance(impl.value, Choice):
            choice = impl.value
            parents = [0]*len(choice.outcomes)
            for i, outcome in enumerate(choice.outcomes):
                if not isinstance(outcome, DataOp):
                    # TODO handle tuples of dataops
                    parents[i] = ValueOp(outcome)
            return_op = ChoiceOp(choice.outcome_names, len(choice.outcomes), choice.name, inputs=parents)
        else:
            return_op = ValueOp(impl.value)
    elif isinstance(impl, CallMethod):
        return_op = MethodCallOp(impl.method_name, impl.args, impl.kwargs)
    elif isinstance(impl, Call):
        return_op = CallOp(
            name=impl.get_func_name(),
            func=impl.func,
            args=impl.args,
            kwargs=impl.kwargs
        )
    elif isinstance(impl, GetAttr):
        return_op = GetAttrOp(attr_name=impl.attr_name)
    elif isinstance(impl, GetItem):
        return_op = GetItemOp(key=impl.key)
    elif isinstance(impl, SkrubBinOp):
        return_op = BinOp(op=impl.op, left=impl.left, right=impl.right)
    elif isinstance(impl, Apply):
        return_op = EstimatorOp(
            y=impl.y, 
            estimator=impl.estimator, 
            cols=impl.cols, 
            how=impl.how, 
            allow_reject=impl.allow_reject, 
            unsupervised=impl.unsupervised, 
            kwargs=impl.kwargs if hasattr(impl, "kwargs") else {})
    else:
        return_op = ImplOp(skrub_impl=impl, name=data_op.__skrub_short_repr__())

    return_op.is_X = is_X
    return_op.is_y = is_y
    return return_op