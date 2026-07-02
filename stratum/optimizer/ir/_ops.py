from __future__ import annotations
from enum import Enum, auto
from stratum._config import FLAGS
from types import SimpleNamespace
from typing import Callable

from joblib import parallel_config
from sklearn import clone
from sklearn.base import BaseEstimator
from skrub._data_ops._choosing import Choice
from skrub._data_ops._data_ops import DataOp, Apply, Value, CallMethod, Call, GetAttr, GetItem, BinOp as SkrubBinOp, UnaryOp as SkrubUnaryOp, Concat, Var, _wrap_estimator
from skrub._utils import PassThrough
from skrub.selectors._base import All
from pandas import DataFrame
from polars import DataFrame as PlDataFrame, Series as PlSeries
from stratum.utils._skrub_graph import _collect_child_data_ops
import logging
import os
logger = logging.getLogger(__name__)


class OutputType(Enum):
    """The kind of value an :class:`Op` produces.

    Replaces the old boolean ``is_dataframe_op`` flag with a small lattice so that
    rewrites can distinguish a single column (``SERIES``) from a whole table
    (``FRAME``). Telling the two apart is what lets selection detection recognise
    ``df[mask]`` as a relational selection (the ``mask`` is a ``SERIES``).

    Boolean-ness is *not* a separate output type: a boolean mask is just a
    ``SERIES`` whose values happen to be booleans (a value-level property), so it
    is tracked separately rather than as its own enum member.

    ``UNKNOWN`` is the default (the op produces a non-tabular Python value, e.g. a
    scalar or an arbitrary object) and corresponds to the old ``is_dataframe_op =
    False``. ``MATRIX`` is ndarray-valued (e.g. ``np.load``) and is deliberately
    *not* a frame type: numpy data is handled by the numeric extraction path, not
    the dataframe path. (A ``VECTOR`` type will be added once we have an op that
    produces one -- e.g. a GetItem/aggregation on a MATRIX.)
    """
    UNKNOWN = auto()
    FRAME = auto()
    SERIES = auto()
    SCALAR = auto()
    MATRIX = auto()


# Output types that belong to the dataframe (pandas/polars) world. A frame and a
# series are both manipulated by the dataframe extraction path; a MATRIX (numpy)
# is not. Used to decide whether an op consumes already-produced frame data (a
# dataframe operation) or a leaf/raw value (a read/source).
FRAME_TYPES = frozenset({OutputType.FRAME, OutputType.SERIES})


def is_frame_like(op) -> bool:
    """True if ``op`` produces frame-world data (a frame or a series)."""
    return op.output_type in FRAME_TYPES


def _operand_index_from_impl(skrub_impl) -> dict:
    """Map id(DataOp) -> operand index, in the canonical field-walk order.

    Uses the same walk as graph extraction (`_collect_child_data_ops` over
    `impl._fields`), de-duplicating repeated DataOps to the same index. This is
    the single source of truth shared with the converter so that an ImplOp's
    `inputs` list and its operand indices always agree.
    """
    index: dict = {}
    for field_name in skrub_impl._fields:
        for data_op in _collect_child_data_ops(getattr(skrub_impl, field_name)):
            if id(data_op) not in index:
                index[id(data_op)] = len(index)
    return index

class OperandRef:
    """Explicit reference to the ``k``-th entry of an :class:`Op`'s ``inputs`` list.

    Replaces the old opaque ``DATA_OP_PLACEHOLDER`` sentinel. Instead of relying on
    the *order* in which placeholders are walked at runtime, an operand now carries
    the exact index of the input that fills it, so ``process()`` can resolve
    ``inputs[ref.k]`` directly and rewrites that reorder inputs are checkable.
    """
    __slots__ = ("k",)

    def __init__(self, k: int):
        self.k = k

    def __eq__(self, other):
        return isinstance(other, OperandRef) and other.k == self.k

    def __hash__(self):
        return hash(("OperandRef", self.k))

    def __str__(self):
        return f"${self.k}"

    def __repr__(self):
        return f"OperandRef({self.k})"


class OperandBinder:
    """Builds an op's de-duplicated ``inputs`` list and replaces DataOps in
    structured fields with :class:`OperandRef`, all from a single ordered walk.

    The order in which ``ref``/``bind_seq``/``bind_map`` are called defines the
    canonical operand order (e.g. the implicit primary object is bound first, so
    it becomes ``OperandRef(0)``). Repeated DataOps map to the same index, so the
    same upstream op feeding two slots produces a single input edge.
    """

    def __init__(self, ids_to_ops: dict):
        self.ids_to_ops = ids_to_ops
        self.inputs: list = []
        self._index: dict = {}  # id(Op) -> position in self.inputs

    def ref_op(self, op: "Op") -> OperandRef:
        """Bind an already-converted Op (not a DataOp) to an OperandRef."""
        idx = self._index.get(id(op))
        if idx is None:
            idx = len(self.inputs)
            self.inputs.append(op)
            self._index[id(op)] = idx
        return OperandRef(idx)

    def ref(self, data_op: DataOp) -> OperandRef:
        """Bind a single DataOp to an OperandRef via the id->Op lookup."""
        return self.ref_op(self.ids_to_ops[id(data_op)])

    def bind(self, value):
        """Recursively replace DataOps nested in tuples/lists/dicts with OperandRefs.

        The recursion order mirrors ``_collect_child_data_ops`` so operand indices
        line up with the graph's child order (e.g. ``df.join([df2, df3])`` binds
        df2 then df3 inside the list argument).
        """
        if isinstance(value, DataOp):
            return self.ref(value)
        if isinstance(value, tuple):
            return tuple(self.bind(v) for v in value)
        if isinstance(value, list):
            return [self.bind(v) for v in value]
        if isinstance(value, dict):
            return {k: self.bind(v) for k, v in value.items()}
        return value

    def bind_seq(self, seq):
        """Bind DataOps in a tuple/list argument sequence to OperandRefs."""
        return tuple(self.bind(a) for a in seq)

    def bind_map(self, mapping):
        """Bind DataOps in a kwargs dict to OperandRefs."""
        return {k: self.bind(v) for k, v in mapping.items()}


def _resolve_operand(value, inputs):
    """Recursively replace OperandRefs nested in value with values from inputs."""
    if isinstance(value, OperandRef):
        return inputs[value.k]
    if isinstance(value, tuple):
        return tuple(_resolve_operand(v, inputs) for v in value)
    if isinstance(value, list):
        return [_resolve_operand(v, inputs) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_operand(v, inputs) for k, v in value.items()}
    return value


def _resolve_args(args, inputs):
    """Replace OperandRefs in an args sequence with values from the inputs list."""
    return [_resolve_operand(a, inputs) for a in args]


def _resolve_kwargs(kwargs, inputs):
    """Replace OperandRefs in a kwargs dict with values from the inputs list."""
    return {k: _resolve_operand(v, inputs) for k, v in kwargs.items()}


# --- Structure keys for common-subexpression elimination -------------------
# `Op.structure_key()` returns a hashable value that is equal for two ops iff
# they are the same computation. Sentinel keys carry a leading marker string so
# they stay disjoint from real values.
_ALL_SELECTOR_KEY = ("__all_selector__",)
# A graph-fed estimator hyper-parameter: its binding is captured by an op's
# `param_refs` plus the input ids, so the stale DataOp left in get_params() must
# not block two otherwise-equal estimators from merging.
_GRAPH_PARAM_KEY = ("__graph_param__",)


def config_key(value):
    """Turn a config-field value into a hashable, value-based key.

    OperandRefs and hashable scalars are kept by value (so equal operands and
    constants compare equal); containers are recursed into; estimators are keyed
    by type and parameters; unhashable leaves (DataFrames, arrays, ...) fall back
    to identity, which is conservative (distinct objects never compare equal).
    """
    if isinstance(value, OperandRef):
        return value
    if isinstance(value, All):
        return _ALL_SELECTOR_KEY
    if isinstance(value, BaseEstimator):
        return estimator_key(value)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(config_key(v) for v in value))
    if isinstance(value, dict):
        return ("__dict__", frozenset((k, config_key(v)) for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        return ("__set__", frozenset(config_key(v) for v in value))
    try:
        hash(value)
    except TypeError:
        return ("__id__", id(value))
    return value


def estimator_key(est: BaseEstimator):
    """Structure key for an estimator, consistent with parameter-wise equality.

    Graph-fed parameters (still DataOps in ``get_params()``) are normalized to a
    constant marker: their binding is represented by the op's ``param_refs`` field
    and input ids, not by the estimator object itself.
    """
    items = []
    for k, v in est.get_params().items():
        items.append((k, _GRAPH_PARAM_KEY if isinstance(v, DataOp) else config_key(v)))
    return ("__estimator__", type(est), frozenset(items))


class Op():
    def __init__(self, inputs=None,outputs=None, name=None, is_X=False, is_y=False):
        self.name = name
        self.outputs = outputs if outputs is not None else []
        self.inputs = inputs if inputs is not None else []
        self.is_X = is_X
        self.is_y = is_y
        self.output_type = OutputType.UNKNOWN
        self.is_split_op = False
        self.was_cloned = False
        self.remove_after: list[Op] = []

    def to_str_helper(self):
        class_name = self.__class__.__name__
        is_df = " [df]" if self.output_type is OutputType.FRAME else ""
        name = f"({self.name})" if self.name and len(self.name) > 0 else ""
        # truncate name if it is too long
        if len(name) > 50:
            name = name[:50] + "..."
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

    @property
    def num_input_operands(self) -> int:
        return len(self.inputs)

    def add_output(self, output: Op):
        """Add an output edge, de-duplicating so an op appears at most once."""
        for out_ in self.outputs:
            if out_ is output:
                return
        self.outputs.append(output)

    def add_input(self, input: Op) -> int:
        """Add an input edge, de-duplicating. Returns the operand index of `input`."""
        for i, in_ in enumerate(self.inputs):
            if in_ is input:
                return i
        self.inputs.append(input)
        return len(self.inputs) - 1

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

    def process(self, mode: str, inputs: list):
        raise NotImplementedError(f"Processing of {self.__class__.__name__} objects is not implemented yet. Please implement it.")

    def structure_key(self):
        """Hashable key for CSE: equal for two ops iff they are the same computation.

        Returns ``None`` for ops that must never be merged (opaque ops without a
        ``fields`` attribute, e.g. ImplOp/SearchEvalOp). The key combines the op
        type, its inputs by identity (already canonicalized when visited in
        topological order) and its configuration (the ``fields`` attributes,
        whose operands are index-based ``OperandRef``s). Subclasses override when
        identity- or name-based semantics are needed.
        """
        fields = getattr(type(self), "fields", None)
        if fields is None:
            return None
        input_ids = tuple(id(i) for i in self.inputs)
        config = tuple((name, config_key(getattr(self, name))) for name in fields)
        return (type(self), input_ids, config)

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

    @property
    def operand_index(self) -> dict:
        """Cached id(DataOp) -> operand index map for this impl's fields."""
        idx = getattr(self, "_operand_index", None)
        if idx is None:
            idx = _operand_index_from_impl(self.skrub_impl)
            self._operand_index = idx
        return idx

    def replace_fields_with_values(self, inputs):
        """Replace DataOp fields in implementation with their computed values."""
        index = self.operand_index

        def replace_dataop(value):
            """Recursively replace DataOp instances with their resolved input."""
            if isinstance(value, DataOp):
                return inputs[index[id(value)]]
            elif isinstance(value, (list, tuple)):
                new_seq = [replace_dataop(item) for item in value]
                return type(value)(new_seq)
            elif isinstance(value, dict):
                return {key: replace_dataop(val) for key, val in value.items()}
            else:
                return value

        return SimpleNamespace(**{field: replace_dataop(getattr(self.skrub_impl, field)) for field in self.skrub_impl._fields})

    def process(self, mode: str, inputs: list):
        if hasattr(self.skrub_impl, "eval"):
            # DataOp with eval method have a fused implementation of the generator and the compute method
            # we need to iterate over the generator and replace the requested fields with correct inputs.
            # Indices are assigned in yield order (de-duplicating repeated DataOps), matching the
            # order in which inputs are consumed.
            index = {}
            last_yield = None
            # Variables are resolved to constants at compile time, so the skrub
            # impl needs no environment -- its inputs arrive via the generator.
            gen = self.skrub_impl.eval(mode=mode, environment={})
            while True:
                try:
                    last_yield = gen.send(last_yield)
                except StopIteration as e:
                    return e.value
                if isinstance(last_yield, DataOp):
                    k = index.setdefault(id(last_yield), len(index))
                    last_yield = inputs[k]
        else:
            ns = self.replace_fields_with_values(inputs)
            return self.skrub_impl.compute(ns, mode, {})

class VariableOp(Op):
    def __init__(self, name: str, value = None):
        super().__init__(name=name)
        self.name = name
        if value is not None:
            self.value = value
        else:
            self.value = "EMPTY_VARIABLE"

    def clone(self):
        return VariableOp(name=self.name)

    def structure_key(self):
        # Two `var("x")` references denote the same input regardless of identity.
        return (VariableOp, self.name)

    def process(self, mode: str, inputs: list):
        # Variables are resolved to constant ValueOps at compile time (see as_op
        # with an `env`), so a VariableOp should never reach the runtime.
        raise RuntimeError(
            f"VariableOp({self.name!r}) reached the runtime; variables must be "
            f"resolved to constants at compile time by passing `env` to optimize().")

class BaseEstimatorOp(Op):
    fields = ["estimator", "y", "cols", "how", "allow_reject", "unsupervised", "kwargs", "param_refs"]

    def __init__(self, estimator: BaseEstimator, y=None, cols=None, how="no-wrap", allow_reject=False, unsupervised=False, kwargs=None, param_refs=None):
        super().__init__(name=estimator.__class__.__name__)
        if kwargs is None:
            kwargs = {}
        self.check_kwargs(kwargs)
        self.estimator = estimator
        self.original_estimator = clone(self.estimator)
        # X is the implicit primary operand (OperandRef(0)); y/cols are OperandRef
        # when fed by the graph, otherwise plain values. param_refs maps the names of
        # estimator hyper-parameters that are graph-fed to their OperandRef.
        self.y = y
        self.cols = cols
        self.how = how
        self.allow_reject = allow_reject
        self.unsupervised = unsupervised
        self.kwargs = kwargs
        self.param_refs = param_refs if param_refs is not None else {}
        self.parallelism = os.cpu_count() # TODO:this will should be set during physical planning phase

    def clone(self):
        params = self.estimator.get_params()
        estimator_new = clone(self.estimator)
        estimator_new.set_params(**params)
        new_op = self.__class__(
            estimator=estimator_new,
            y=self.y,
            cols=self.cols,
            how=self.how,
            allow_reject=self.allow_reject,
            unsupervised=self.unsupervised,
            kwargs=self.kwargs,
            param_refs=self.param_refs,
        )
        new_op.was_cloned = True
        return new_op

    def extract_args_from_inputs(self, mode: str, inputs: list):
        """
        Extract all necessary data from an EstimatorOp to make it picklable for multiprocessing.

        Returns a tuple of picklable data that can be sent to worker processes.
        """
        x = inputs[0]
        assert x is not None, f"X is None for {self}"
        y = None if mode == 'predict' else inputs[self.y.k] if isinstance(self.y, OperandRef) else self.y
        estm = self.estimator if mode == "predict" else self.original_estimator
        place_holders = {name: inputs[ref.k] for name, ref in self.param_refs.items()}
        estm.set_params(**place_holders)
        cols = inputs[self.cols.k] if isinstance(self.cols, OperandRef) else self.cols
        return (
            estm,
            x,
            y,
            cols,
            self.how,
            self.allow_reject,
            self.unsupervised,
            self.kwargs,
            mode,
            self.parallelism
        )

    def process(self, mode: str, inputs: list):
        # we use a separate function to process the estimator to allow reuse for multiprocessing
        task_data = self.extract_args_from_inputs(mode, inputs)
        process_task = self.get_process_task()
        result, self.estimator = process_task(task_data)
        return result

    def get_process_task(self):
        raise NotImplementedError(f"get_process_task must be implemented in {self.__class__.__name__}")

class EstimatorOp(BaseEstimatorOp):
    def get_process_task(self):
        return process_estimator_task

class TransformerOp(BaseEstimatorOp):
    def get_process_task(self):
        return process_transformer_task

class DummyConfigManager:
    """A no-op context manager that does nothing."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

def estimator_parallel_config(n_jobs: int = None):
    if n_jobs is not None:
        logger.debug(f"Using threading backend with {n_jobs} jobs")
        return parallel_config(backend='threading', n_jobs=n_jobs)
    else:
        return DummyConfigManager()

def estm_supports_polars(estimator):
    is_sklearn = estimator.__class__.__module__.startswith("sklearn.") or estimator.__class__.__module__.startswith("skrub.")
    is_stratum = estimator.__class__.__module__.startswith("stratum.") and estimator.__class__.__name__.startswith("Rusty")
    # other_frameworks = estimator.__class__.__module__.startswith("xgboost.")
    return is_sklearn or is_stratum #or other_frameworks

def check_estm_inputs(estimator, mode, x, y):
    input_is_polars = type(x) == PlDataFrame
    converted = False
    if estimator.__class__.__module__.startswith("skrub."):
        if estimator.__class__.__name__.startswith("ApplyTo"):
            estimator = estimator.transformer
    if input_is_polars and not estm_supports_polars(estimator):
        converted = True
        logger.debug(f"Estimator {estimator.__class__.__name__} does not support Polars DataFrame. Converting to Pandas DataFrame.")
        x = x.to_pandas()
        if y is not None and mode == "fit_transform":
            y = y.to_pandas()
    return converted, x, y

def process_estimator_task(task_data):
    """ Process a predictor (EstimatorOp) task in a worker process. """
    (estimator, x, y, cols, how, allow_reject, unsupervised, kwargs, mode, parallelism) = task_data
    _, x, y = check_estm_inputs(estimator, mode, x, y)
    if mode == "fit_transform":
        estimator = _wrap_estimator(estimator, cols, how=how, allow_reject=allow_reject, X=x)
        y_arg = () if unsupervised else (y,)
        estimator.fit(x, *y_arg, **kwargs)
        result = estimator.predict(x, **kwargs)
        # Return both result and fitted estimator (in case of multi-processing)
        return result, estimator
    elif mode == "predict":
        result = estimator.predict(x, **kwargs)
        return result, estimator
    else:
        raise ValueError(f"Mode {mode} not supported for EstimatorOp.")

def process_transformer_task(task_data):
    """ Process a transformer (TransformerOp) task in a worker process. """
    (estimator, x, y, cols, how, allow_reject, unsupervised, kwargs, mode, parallelism) = task_data
    converted, x, y = check_estm_inputs(estimator, mode, x, y)
    with estimator_parallel_config(parallelism):
        if mode == "fit_transform":
            estimator = _wrap_estimator(estimator, cols, how=how, allow_reject=allow_reject, X=x)
            y_arg = () if unsupervised else (y,)
            result = estimator.fit_transform(x, *y_arg, **kwargs)
        elif mode == "predict":
            result = estimator.transform(x, **kwargs)
        else:
            raise ValueError(f"Mode {mode} not supported for TransformerOp.")
    if converted:
        result = PlDataFrame(result)
    return result, estimator


class ChoiceOp(Op):
    fields = ["outcome_names"]
    
    def __init__(self, outcome_names: list[str] = None, n_outcomes: int = None, choice_name: str=None, append_choice_name = True, inputs: list = None):
        if inputs is None:
            inputs = []
        if outcome_names is None:
            outcome_names = [[(choice_name, f"Opt{i}")] for i in range(n_outcomes)]
        elif append_choice_name:
            outcome_names = [[(choice_name, name)] for name in outcome_names]

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
        opts = " | ".join(self.make_outcome_names())
        max_len = 50
        if len(opts) > max_len:
            opts = opts[:max_len] + "..."
        self.name = opts

    def clone(self):
        new_op = ChoiceOp(outcome_names=self.outcome_names, append_choice_name=False)
        new_op.name = self.name
        new_op.was_cloned = True
        return new_op

    def structure_key(self):
        # A choice models alternatives; merging two choices is never valid.
        return None

    def process(self, mode: str, inputs: list):
        results = [{"id" : name, "vals" : inputs[i]} for i, name in enumerate(self.make_outcome_names())]
        return results[0] if len(results) == 1 else results

class ValueOp(Op):
    fields = ["value"]
    
    def __init__(self, value):
        super().__init__(name="DataFrame" if isinstance(value, DataFrame) else str(value))
        self.value = value

    def clone(self):
        raise ValueError(f"We should not clone ValueOp objects.")

    def process(self, mode: str, inputs: list):
        out = self.value
        self.value = None
        return out

class MethodCallOp(Op):
    fields = ["method_name", "args", "kwargs"]
    
    def __init__(self, method_name: str, args = None, kwargs = None):
        super().__init__(name=method_name)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def process(self, mode: str, inputs: list):
        # The object the method is called on is the implicit primary operand (index 0).
        _obj = inputs[0]
        _args = _resolve_args(self.args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        if self.method_name == "apply" and isinstance(_obj, PlSeries):
            return _obj.map_elements(*_args, **_kwargs)
        return _obj.__getattribute__(self.method_name)(*_args, **_kwargs)

class CallOp(Op):
    fields = ["func", "args", "kwargs"]
    
    def __init__(self, name=None, func=None, args=None, kwargs=None):
        if name is None:
            name = "CallOp" if func is None else func.__name__
        super().__init__(name=name)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def process(self, mode: str, inputs: list):
        _args = _resolve_args(self.args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        return self.func(*_args, **_kwargs)

class GetAttrOp(Op):
    fields = ["attr_name"]
    
    def __init__(self, attr_name: str=None):
        super().__init__(name=attr_name if attr_name else '?')
        self.attr_name = attr_name

    def process(self, mode: str, inputs: list):
        if self.output_type is OutputType.FRAME:
            result = inputs[0]
            for attr in self.attr_name:
                result = getattr(result, attr)
            return result
        else:
            return getattr(inputs[0], self.attr_name)

class GetItemOp(Op):
    fields = ["key", "is_filter"]
    
    def __init__(self, key=None, name=None, is_filter=False):
        # key is either a constant or an OperandRef (when the key is graph-fed).
        self.key = key
        self.is_filter = is_filter
        if name is None:
            name = str(key)
        super().__init__(name=name)


    def process(self, mode: str, inputs: list):
        # The container being indexed is the implicit primary operand (index 0).
        key = inputs[self.key.k] if isinstance(self.key, OperandRef) else self.key
        if self.is_filter and FLAGS.force_polars:
            return inputs[0].filter(key)
        return inputs[0][key]

class BinOp(Op):
    fields = ["op", "left", "right"]
    
    def __init__(self, op: Callable, left, right):
        super().__init__(name=op.__name__.lstrip('__').rstrip('__'))
        self.op = op
        # left/right are OperandRefs when graph-fed, otherwise constants.
        self.left = left
        self.right = right


    def process(self, mode: str, inputs: list):
        left = inputs[self.left.k] if isinstance(self.left, OperandRef) else self.left
        right = inputs[self.right.k] if isinstance(self.right, OperandRef) else self.right
        return self.op(left, right)

class UnaryOp(Op):
    fields = ["op", "operand"]

    def __init__(self, op: Callable, operand):
        super().__init__(name=op.__name__.lstrip('__').rstrip('__'))
        self.op = op
        # operand is an OperandRef when graph-fed, otherwise a constant.
        self.operand = operand

    def process(self, mode: str, inputs: list):
        operand = inputs[self.operand.k] if isinstance(self.operand, OperandRef) else self.operand
        return self.op(operand)

class SearchEvalOp(Op):
    def __init__(self, outcome_names: list[str], parent: Op = None):
        super().__init__()
        self.name = "evaluate gridsearch" 
        self.outcome_names = outcome_names
        self.parents = [] if parent is None else [parent]
        self.children = []

    def clone(self, children: list[Op] = None, parents: list[Op] = None):
        raise ValueError(f"We should not clone SearchEvalOp objects.")

def _bind_or_value(binder: OperandBinder, value):
    """Bind a field that is either a single DataOp (-> OperandRef) or a constant."""
    return binder.ref(value) if isinstance(value, DataOp) else value


def _apply_estimator_op(impl: Apply, estimator, ids_to_ops: dict) -> Op:
    """Build the TransformerOp/EstimatorOp for one concrete estimator of an Apply impl.

    ``estimator`` is ``impl.estimator`` itself, or one outcome of it when the
    Apply's estimator is a ``Choice``. Each call uses its own binder so every op
    gets X as OperandRef(0) and its own de-duplicated inputs list.
    """
    if estimator is None or (isinstance(estimator, str) and estimator == "passthrough"):
        # Same normalization skrub's _wrap_estimator applies at fit time; needed
        # here already because BaseEstimatorOp clones its estimator on construction.
        estimator = PassThrough()
    estimator_class = EstimatorOp if hasattr(estimator, "predict") else TransformerOp
    binder = OperandBinder(ids_to_ops)
    binder.ref(impl.X)  # OperandRef(0)
    param_refs = {k: binder.ref(v) for k, v in estimator.get_params().items()
                  if isinstance(v, DataOp) and id(v) in ids_to_ops}
    y = _bind_or_value(binder, impl.y)
    cols = _bind_or_value(binder, impl.cols)
    op = estimator_class(
        estimator=estimator,
        y=y,
        cols=cols,
        how=impl.how,
        allow_reject=impl.allow_reject,
        unsupervised=impl.unsupervised,
        kwargs={},
        param_refs=param_refs,
    )
    op.inputs = binder.inputs
    return op


def _outcome_display(estimator) -> str:
    """Human-readable label for an estimator outcome (None/'passthrough' -> PassThrough)."""
    if estimator is None or (isinstance(estimator, str) and estimator == "passthrough"):
        return PassThrough.__name__
    return type(estimator).__name__


def _flatten_estimator_choice(choice: Choice):
    """Flatten an estimator ``Choice`` (possibly nesting further Choices) into leaves.

    Yields ``(name_path, estimator)`` per leaf estimator, where ``name_path`` is a
    list of ``(choice_name, value)`` pairs -- ChoiceOp's internal ``outcome_names``
    representation -- so a nested choice collapses into a single flat ChoiceOp over
    all leaf estimators, matching how skrub expands its parameter grid. An
    intermediate (nested) choice only contributes to the path when its outcome is
    named; the leaf always contributes its outcome name or estimator class name.
    """
    for i, outcome in enumerate(choice.outcomes):
        label = choice.outcome_names[i] if choice.outcome_names is not None else None
        if isinstance(outcome, Choice):
            prefix = [(choice.name, label)] if label is not None else []
            for sub_path, est in _flatten_estimator_choice(outcome):
                yield prefix + sub_path, est
        elif isinstance(outcome, DataOp):
            raise NotImplementedError(
                "Apply with a Choice estimator only supports concrete estimator "
                "(or None/'passthrough') outcomes; DataOp outcomes are not "
                f"supported (choice {choice.name!r}).")
        else:
            value = label if label is not None else _outcome_display(outcome)
            yield [(choice.name, value)], outcome


def as_op(data_op: DataOp, ids_to_ops: dict, env: dict | None = None) -> Op:
    """Convert a single skrub DataOp into an Op, building its de-duplicated
    ``inputs`` list and operand references in one canonical field walk.

    ``ids_to_ops`` maps ``id(DataOp) -> Op`` and must already contain every input
    of ``data_op`` (guaranteed by converting in topological order). Output edges
    are wired here too: each input op gets ``data_op``'s Op added to its outputs.

    ``env`` is the runtime environment (variable name -> value), when known at
    compile time. A ``Var`` whose name is bound in ``env`` is then resolved to a
    constant ``ValueOp`` instead of a ``VariableOp``, so the scheduler needs no
    environment to feed it at runtime.
    """
    impl = data_op._skrub_impl
    is_X = is_y = False
    if impl is not None:
        is_X = impl.is_X
        is_y = impl.is_y
    binder = OperandBinder(ids_to_ops)
    return_op = None

    if isinstance(impl, Value):
        if isinstance(impl.value, Choice):
            choice = impl.value
            # Choice outcomes are consumed positionally by ChoiceOp.process; keep one
            # input entry per outcome (constants become fresh ValueOps).
            inputs = [ids_to_ops[id(o)] if isinstance(o, DataOp) else ValueOp(o)
                      for o in choice.outcomes]
            return_op = ChoiceOp(choice.outcome_names, len(choice.outcomes), choice.name)
            return_op.inputs = inputs
        else:
            return_op = ValueOp(impl.value)
    elif isinstance(impl, CallMethod):
        binder.ref(impl.obj)  # implicit primary operand -> OperandRef(0)
        return_op = MethodCallOp(impl.method_name, binder.bind_seq(impl.args), binder.bind_map(impl.kwargs))
        return_op.inputs = binder.inputs
    elif isinstance(impl, Call):
        return_op = CallOp(
            name=impl.get_func_name(),
            func=impl.func,
            args=binder.bind_seq(impl.args),
            kwargs=binder.bind_map(impl.kwargs),
        )
        return_op.inputs = binder.inputs
    elif isinstance(impl, GetAttr):
        binder.ref(impl.source_object)  # OperandRef(0)
        return_op = GetAttrOp(attr_name=impl.attr_name)
        return_op.inputs = binder.inputs
    elif isinstance(impl, GetItem):
        binder.ref(impl.container)  # OperandRef(0)
        key = _bind_or_value(binder, impl.key)
        name = impl.key._skrub_impl.__class__.__name__ if isinstance(impl.key, DataOp) else str(impl.key)
        return_op = GetItemOp(key=key, name=name)
        return_op.inputs = binder.inputs
    elif isinstance(impl, SkrubBinOp):
        left = _bind_or_value(binder, impl.left)
        right = _bind_or_value(binder, impl.right)
        return_op = BinOp(op=impl.op, left=left, right=right)
        return_op.inputs = binder.inputs
    elif isinstance(impl, SkrubUnaryOp):
        operand = _bind_or_value(binder, impl.operand)
        return_op = UnaryOp(op=impl.op, operand=operand)
        return_op.inputs = binder.inputs
    elif isinstance(impl, Apply):
        if isinstance(impl.estimator, Choice):
            # An estimator choice expands to a ChoiceOp over one estimator op per
            # outcome (mirroring Value(Choice) above), so choice unrolling and
            # grid search work over the alternatives. Nested estimator choices are
            # flattened into a single ChoiceOp over all leaf estimators; the leaf
            # name paths use ChoiceOp's combi format so they concatenate correctly
            # if choice_unrolling later combines this choice with a downstream one.
            leaves = list(_flatten_estimator_choice(impl.estimator))
            outcome_ops = [_apply_estimator_op(impl, est, ids_to_ops) for _, est in leaves]
            for est_op in outcome_ops:
                # The trailing edge-wiring below only covers the returned op.
                for in_op in est_op.inputs:
                    in_op.add_output(est_op)
            return_op = ChoiceOp(outcome_names=[path for path, _ in leaves],
                                 append_choice_name=False, inputs=outcome_ops)
        else:
            return_op = _apply_estimator_op(impl, impl.estimator, ids_to_ops)
    elif isinstance(impl, Var):
        if env is not None and impl.name in env:
            # Resolve the variable to a compile-time constant; the runtime no
            # longer needs the environment to feed it.
            return_op = ValueOp(env[impl.name])
        else:
            return_op = VariableOp(name=impl.name, value=impl.value)
    elif isinstance(impl, Concat):
        from stratum.optimizer.ir._dataframe_ops import ConcatOp
        first = _bind_or_value(binder, impl.first)
        others = list(binder.bind_seq(impl.others))
        axis = _bind_or_value(binder, impl.axis)
        return_op = ConcatOp(first=first, others=others, axis=axis)
        return_op.inputs = binder.inputs
    else:
        for field_name in impl._fields:
            for child in _collect_child_data_ops(getattr(impl, field_name)):
                binder.ref(child)
        return_op = ImplOp(skrub_impl=impl, name=data_op.__skrub_short_repr__())
        return_op.inputs = binder.inputs

    # Wire output edges: every input op gets this op added to its outputs (deduped).
    for in_op in return_op.inputs:
        in_op.add_output(return_op)

    return_op.is_X = is_X
    return_op.is_y = is_y
    return return_op