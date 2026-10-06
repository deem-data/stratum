from __future__ import annotations
from numbers import Number
from types import SimpleNamespace
from typing import Callable

from joblib import parallel_config
from sklearn import clone
from sklearn.base import BaseEstimator
from skrub._data_ops._choosing import BaseChoice, Choice, DiscretizedNumericChoice, Match
from skrub._data_ops._data_ops import DataOp, Apply, Value, CallMethod, Call, GetAttr, GetItem, BinOp as SkrubBinOp, UnaryOp as SkrubUnaryOp, Concat, Scoring, SplitX, Var, _wrap_estimator
from skrub._data_ops._utils import NULL
from skrub._utils import PassThrough
from pandas import DataFrame
import polars as pl
from polars import DataFrame as PlDataFrame
from stratum.frontend._skrub_graph import _collect_child_data_ops
from stratum.optimizer.logical import _schema
# Shared IR foundation. Re-exported below so existing ``from ..._ops import X``
# call sites (OutputType, OperandRef, is_frame_like, config_key, ...) keep working.
from stratum.optimizer.logical._base import (
    IRNode, OutputType, FRAME_TYPES, is_frame_like, OperandRef,
    _resolve_operand, _resolve_args, _resolve_kwargs, remap_operand_refs,
    config_key, estimator_key, clone_value,
    _ALL_SELECTOR_KEY, _GRAPH_PARAM_KEY,
)
import logging
import os
logger = logging.getLogger(__name__)


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


class OperandBinder:
    """Builds an op's de-duplicated ``inputs`` list and replaces DataOps in
    structured fields with :class:`OperandRef`, all from a single ordered walk.

    The order in which ``ref``/``bind_seq``/``bind_map`` are called defines the
    canonical operand order (e.g. the implicit primary object is bound first, so
    it becomes ``OperandRef(0)``). Repeated DataOps map to the same index, so the
    same upstream op feeding two slots produces a single input edge.

    With ``choice_ops`` (``id(choice) -> ChoiceOp``, shared by the whole
    conversion), a skrub choice found in a field is bound like a DataOp: to the
    ChoiceOp built for that choice object (see :func:`_choice_op`). Without it,
    choices are left in place as plain values.
    """

    def __init__(self, ids_to_ops: dict, choice_ops: dict | None = None):
        self.ids_to_ops = ids_to_ops
        self.choice_ops = choice_ops
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
        if self.choice_ops is not None:
            if isinstance(value, (BaseChoice, Match)):
                return self.ref_op(_choice_op(value, self.ids_to_ops, self.choice_ops))
            if isinstance(value, BaseEstimator) and _contains_choice(value):
                raise NotImplementedError(
                    f"A choice inside the parameters of {type(value).__name__} is only "
                    "supported in the estimator of `.skb.apply()`.")
        if isinstance(value, tuple):
            return tuple(self.bind(v) for v in value)
        if isinstance(value, list):
            return [self.bind(v) for v in value]
        if isinstance(value, dict):
            return {k: self.bind(v) for k, v in value.items()}
        if isinstance(value, slice):
            return slice(self.bind(value.start), self.bind(value.stop), self.bind(value.step))
        return value

    def bind_seq(self, seq):
        """Bind DataOps in a tuple/list argument sequence to OperandRefs."""
        return tuple(self.bind(a) for a in seq)

    def bind_map(self, mapping):
        """Bind DataOps in a kwargs dict to OperandRefs."""
        return {k: self.bind(v) for k, v in mapping.items()}


class Op(IRNode):
    """Logical operator: a backend-agnostic node in the pre-lowering IR.

    Adds the two concerns specific to the logical layer on top of the shared
    :class:`~stratum.optimizer.logical._base.IRNode` structure: CSE structure keys
    and choice detection. Physical operators (post-lowering) share the same
    ``IRNode`` base but not this class.
    """

    def is_choice(self) -> bool:
        return isinstance(self, ChoiceOp)

    def structure_key(self):
        """Hashable key for CSE: equal for two ops iff they are the same computation.

        Returns ``None`` for ops that must never be merged (opaque ops without a
        ``fields`` attribute, e.g. ImplOp). The key combines the op
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

    def consumes_inputs_positionally(self) -> bool:
        # Inputs are resolved via the cached id(DataOp)->index map, so a collapsed
        # slot would misalign `operand_index` with `inputs`.
        return True

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


class SplitXOp(Op):
    """Temporary X boundary carrying skrub's declared splitter until split planning."""

    def __init__(self, skrub_impl: SplitX):
        super().__init__(name="X", is_X=True)
        self.skrub_impl = skrub_impl

    @property
    def operand_index(self) -> dict:
        return _operand_index_from_impl(self.skrub_impl)

    @property
    def source(self) -> Op:
        return self.inputs[self.operand_index[id(self.skrub_impl.X)]]

    def process(self, mode: str, inputs: list):
        raise RuntimeError("SplitXOp should be replaced by the splitting rewrite before execution")


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

# A plan execution's ``mode`` is the estimator method it calls, as it is in skrub.
# ``fit_transform`` fits; the rest are response methods, and which one a pass uses is
# the scorer's to decide (see ``ScoreCandidatesOp``).
FITTING_MODE = "fit_transform"
RESPONSE_MODES = frozenset({"predict", "predict_proba", "predict_log_proba",
                            "decision_function"})


class BaseEstimatorOp(Op):
    fields = ["estimator", "y", "cols", "exclude_cols", "no_wrap", "allow_reject", "unsupervised", "kwargs", "param_refs",
              "feeds_estimator"]
    # skrub keys `Apply.kwargs` by the estimator method the kwargs belong to, and
    # evaluates only the group for the method it is about to call. Subclasses name
    # the two groups stratum can reach: the one for the fitting call and the one
    # for the transform/predict call.
    fit_kwargs_key: str | None = None
    call_kwargs_key: str | None = None
    #: Response methods this kind of op can serve at all. A transformer serves none.
    response_modes: frozenset = frozenset()
    #: What it does instead in a response pass it cannot serve.
    fallback_mode: str = "transform"

    @classmethod
    def callable_kwargs_keys(cls, estimator) -> frozenset:
        """The `.skb.apply()` kwargs groups this op can actually splat for `estimator`.

        A response group is live only if the estimator serves that response; a
        `predict_proba_kwargs` on a regressor is as dead here as it is in skrub.
        """
        return frozenset({cls.fit_kwargs_key, cls.call_kwargs_key}) | {
            m for m in cls.response_modes if hasattr(estimator, m)}

    def __init__(self, estimator: BaseEstimator, y=None, cols=None, exclude_cols=None, no_wrap=False, allow_reject=False, unsupervised=False, kwargs=None, param_refs=None,
                 feeds_estimator=False):
        super().__init__()
        if kwargs is None:
            kwargs = {}
        self.check_kwargs(kwargs)
        self.estimator = estimator
        self.original_estimator = clone(self.estimator)
        # X is the implicit primary operand (OperandRef(0)); y/cols/exclude_cols/no_wrap
        # are OperandRefs when fed by the graph, otherwise plain values. param_refs maps
        # the names of estimator hyper-parameters that are graph-fed to their OperandRef.
        self.y = y
        self.cols = cols
        self.exclude_cols = exclude_cols
        self.no_wrap = no_wrap
        self.allow_reject = allow_reject
        self.unsupervised = unsupervised
        # {method name: kwargs for that method}, as built by `.skb.apply()`. The
        # inner values may contain OperandRefs (a `fit_kwargs` entry can be fed by
        # the graph, e.g. an eval set), so they are resolved per call in
        # `method_kwargs` rather than stored as ready-to-splat dicts.
        self.kwargs = kwargs
        self.param_refs = param_refs if param_refs is not None else {}
        # Whether another `.skb.apply()` consumes this op's output. Such an op is not
        # the last estimator, so like skrub it never answers a response pass itself.
        self.feeds_estimator = feeds_estimator
        # Which responses this op can serve, settled here rather than by a `hasattr` in
        # the execution path. A parameter fed by the graph is still unresolved at this
        # point, so the estimator is taken at its word, as it is in skrub.
        self.supported_modes = frozenset(m for m in self.response_modes
                                         if hasattr(self.estimator, m))
        self.parallelism = os.cpu_count() # TODO:this will should be set during physical planning phase

    def effective_mode(self, mode: str) -> str:
        """The method ``mode`` actually calls on this op's estimator.

        An op that cannot serve the requested response falls back the way skrub's
        ``Apply`` does: a transformer transforms, a predictor predicts. So does one
        that feeds another estimator, whatever responses it could serve.
        """
        if mode == FITTING_MODE or (mode in self.supported_modes and not self.feeds_estimator):
            return mode
        return self.fallback_mode

    def method_kwargs(self, key: str | None, inputs: list) -> dict:
        """Resolve the kwargs group `key` against `inputs`, as skrub's
        `Apply._eval_kwargs` does for the method it is about to call."""
        group = self.kwargs.get(key) if key else None
        if group is None:
            return {}
        # The whole group can itself be graph-fed (a DataOp evaluating to a dict),
        # so it is only known to be a dict once resolved.
        group = _resolve_operand(group, inputs)
        if group is None:
            return {}
        self.check_kwargs(group)
        return group

    def clone(self):
        params = self.estimator.get_params()
        estimator_new = clone(self.estimator)
        estimator_new.set_params(**params)
        new_op = self.__class__(
            estimator=estimator_new,
            y=self.y,
            cols=self.cols,
            exclude_cols=self.exclude_cols,
            no_wrap=self.no_wrap,
            allow_reject=self.allow_reject,
            unsupervised=self.unsupervised,
            kwargs=clone_value(self.kwargs),
            param_refs=self.param_refs,
            feeds_estimator=self.feeds_estimator,
        )
        new_op.was_cloned = True
        return new_op

    def extract_args_from_inputs(self, mode: str, inputs: list):
        """
        Extract all necessary data from an EstimatorOp to make it picklable for multiprocessing.

        Returns a tuple of picklable data that can be sent to worker processes.
        """
        mode = self.effective_mode(mode)
        fitting = mode == FITTING_MODE
        x = inputs[0]
        assert x is not None, f"X is None for {self}"
        y = (inputs[self.y.k] if isinstance(self.y, OperandRef) else self.y) if fitting else None
        # Every fit starts from an unfitted clone, as in skrub. Fitting the prototype
        # itself would leave it fitted for the next fold, and CatBoost refuses the
        # `set_params` below on a fitted model.
        estm = clone(self.original_estimator) if fitting else self.estimator
        place_holders = {name: inputs[ref.k] for name, ref in self.param_refs.items()}
        # Graph-fed hyperparameters configure the estimator before fitting. In a
        # response pass ``estm`` is already fitted, and CatBoost rejects any
        # ``set_params`` call at that point (even an empty one).
        if fitting and place_holders:
            estm.set_params(**place_holders)
        cols = _resolve_operand(self.cols, inputs)
        exclude_cols = _resolve_operand(self.exclude_cols, inputs)
        no_wrap = _resolve_operand(self.no_wrap, inputs)
        # A response pass never fits, so (like skrub) the fit group is left unevaluated.
        # Note the difference from skrub: skrub also never *computes* what that group
        # references, while our plan is eager, so a sub-DAG feeding only fit kwargs
        # still runs in predict mode and has to tolerate it (a transformer carving an
        # eval set out of the training fold must return placeholders in predict mode).
        fit_kwargs = self.method_kwargs(self.fit_kwargs_key, inputs) if fitting else {}
        # The call group belongs to the method about to run, so a `predict_proba` pass
        # gets the `predict_proba` group. The fitting pass makes this kind's own call.
        call_kwargs = self.method_kwargs(self.call_kwargs_key if fitting else mode, inputs)
        return (
            estm,
            x,
            y,
            cols,
            exclude_cols,
            no_wrap,
            self.allow_reject,
            self.unsupervised,
            (fit_kwargs, call_kwargs),
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

class PredictorOp(BaseEstimatorOp):
    logical_family = "Predictor"
    # fit_transform mode calls fit() then predict(); predict mode only predict().
    fit_kwargs_key = "fit"
    call_kwargs_key = "predict"
    response_modes = RESPONSE_MODES
    fallback_mode = "predict"

    def get_process_task(self):
        return process_estimator_task

class TransformerOp(BaseEstimatorOp):
    logical_family = "Transformer"
    # A transformer is fitted through fit_transform(), so (as in skrub) the "fit"
    # group never applies to one; predict mode calls transform().
    fit_kwargs_key = "fit_transform"
    call_kwargs_key = "transform"

    def process(self, mode: str, inputs: list):
        raise NotImplementedError("TransformerOp must be lowered before execution")

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
        logger.debug(f"Predictor {estimator.__class__.__name__} does not support Polars DataFrame. Converting to Pandas DataFrame.")
        x = x.to_pandas()
        if y is not None and mode == "fit_transform":
            y = y.to_pandas()
    return converted, x, y

def process_estimator_task(task_data):
    """ Process a predictor (EstimatorOp) task in a worker process. """
    (estimator, x, y, cols, exclude_cols, no_wrap, allow_reject, unsupervised, kwargs, mode, parallelism) = task_data
    fit_kwargs, call_kwargs = kwargs
    _, x, y = check_estm_inputs(estimator, mode, x, y)
    if mode == FITTING_MODE:
        estimator = _wrap_estimator(estimator, cols, exclude_cols=exclude_cols, no_wrap=no_wrap,
                                    allow_reject=allow_reject, X=x)
        y_arg = () if unsupervised else (y,)
        estimator.fit(x, *y_arg, **fit_kwargs)
        result = estimator.predict(x, **call_kwargs)
        # Return both result and fitted estimator (in case of multi-processing)
        return result, estimator
    elif mode in RESPONSE_MODES:
        # `mode` is already this op's effective mode, so the method is one it serves.
        result = getattr(estimator, mode)(x, **call_kwargs)
        return result, estimator
    else:
        raise ValueError(f"Mode {mode} not supported for PredictorOp.")

def process_transformer_task(task_data):
    """ Process a transformer (TransformerOp) task in a worker process. """
    (estimator, x, y, cols, exclude_cols, no_wrap, allow_reject, unsupervised, kwargs, mode, parallelism) = task_data
    fit_transform_kwargs, transform_kwargs = kwargs
    converted, x, y = check_estm_inputs(estimator, mode, x, y)
    with estimator_parallel_config(parallelism):
        if mode == FITTING_MODE:
            estimator = _wrap_estimator(estimator, cols, exclude_cols=exclude_cols, no_wrap=no_wrap,
                                    allow_reject=allow_reject, X=x)
            y_arg = () if unsupervised else (y,)
            result = estimator.fit_transform(x, *y_arg, **fit_transform_kwargs)
        elif mode == "transform":
            # A transformer serves no response method, so every response pass reaches
            # it as `transform` (see `BaseEstimatorOp.effective_mode`).
            result = estimator.transform(x, **transform_kwargs)
        else:
            raise ValueError(f"Mode {mode} not supported for TransformerOp.")
    if converted:
        result = PlDataFrame(result)
    return result, estimator


class ChoiceOp(Op):
    logical_family = "Choice"
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

    def consumes_inputs_positionally(self) -> bool:
        # Each outcome is consumed by position in `process`, so input slots must
        # stay distinct even when two outcomes are the same op.
        return True

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

class MethodCallOp(Op):
    fields = ["method_name", "args", "kwargs"]
    
    def __init__(self, method_name: str, args = None, kwargs = None):
        super().__init__(name=method_name)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

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

def _elementwise_schema(op: "Op", operands) -> "pl.Schema | None":
    """Output schema of an elementwise ``BinOp``/``UnaryOp`` over frame data.

    Keeps the frame operand's column set; the dtypes come from
    ``_schema.elementwise_result_dtype``. Unknown when the op doesn't produce
    frame data, when any frame operand is unknown, or -- for a frame-frame op --
    when the operands don't carry the same columns, since pandas aligns on column
    labels and unions them rather than keeping one side's.

    ``operands`` is the op's operand fields (e.g. ``(left, right)``), used only to
    tell whether *every* operand is a frame-like graph input, which decides
    whether a per-column dtype list can be assembled at all.
    """
    if op.output_type not in FRAME_TYPES:
        return None
    frame_inputs = [in_op for in_op in op.inputs if is_frame_like(in_op)]
    frame_schemas = [in_op.output_schema for in_op in frame_inputs]
    if not frame_schemas or any(s is None for s in frame_schemas):
        return None
    first = frame_schemas[0]
    if any(list(s.keys()) != list(first.keys()) for s in frame_schemas[1:]):
        return None
    # A constant or non-frame operand contributes a dtype we don't know, so we
    # can't assemble a complete per-column dtype list for it.
    all_operands_are_frames = (
        len(frame_inputs) == len(operands)
        and all(isinstance(o, OperandRef) for o in operands))
    return pl.Schema({
        name: _schema.elementwise_result_dtype(
            op.op, [s[name] for s in frame_schemas] if all_operands_are_frames else None)
        for name in first
    })


class GetItemOp(Op):
    fields = ["key", "is_filter"]
    
    def __init__(self, key=None, name=None, is_filter=False):
        # key is either a constant or an OperandRef (when the key is graph-fed).
        self.key = key
        self.is_filter = is_filter
        if name is None:
            name = str(key)
        super().__init__(name=name)

    def propagate_output_schema(self):
        """A literal key projects a sub-schema; a row slice or a boolean row mask
        keeps the input schema."""
        schema = self.inputs[0].output_schema
        if isinstance(self.key, slice):
            self.output_schema = schema  # row slice keeps all columns
        elif isinstance(self.key, OperandRef):
            # A graph-fed key has no static value, so classify by the producing
            # op's kind: SERIES means a boolean row mask (`df[df["x"] > 0]`) and
            # keeps the schema, anything else is an unresolvable column selector.
            # HEURISTIC: a SERIES of column *labels* is column projection too, and
            # is misclassified here, but that idiom doesn't occur in supported
            # pipelines.
            key_op = self.inputs[self.key.k]
            self.output_schema = schema if key_op.output_type is OutputType.SERIES else None
        else:
            self.output_schema = _schema.project_columns(schema, self.key)

    # Execution lives in the physical impls (physical/_getitem_execs.py).

class BinOp(Op):
    fields = ["op", "left", "right"]
    
    def __init__(self, op: Callable, left, right):
        super().__init__(name=op.__name__.lstrip('__').rstrip('__'))
        self.op = op
        # left/right are OperandRefs when graph-fed, otherwise constants.
        self.left = left
        self.right = right


    def propagate_output_schema(self):
        """An elementwise op (``df + 1``, ``df["x"] > 0``) keeps its frame
        operand's columns and re-derives the dtypes; see
        :func:`_elementwise_schema`."""
        self.output_schema = _elementwise_schema(self, (self.left, self.right))

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

    def propagate_output_schema(self):
        """As :class:`BinOp`, for the unary case (``~mask``, ``-df``, ``abs(df)``)."""
        self.output_schema = _elementwise_schema(self, (self.operand,))

    def process(self, mode: str, inputs: list):
        operand = inputs[self.operand.k] if isinstance(self.operand, OperandRef) else self.operand
        return self.op(operand)

def _bind_or_value(binder: OperandBinder, value):
    """Bind a field holding DataOps (-> OperandRefs), constants, or a mix of both.

    Delegates to :meth:`OperandBinder.bind`, which recurses the same containers
    ``_collect_child_data_ops`` does. Binding has to follow that walk exactly:
    graph extraction already reported every DataOp it finds -- including ones
    nested in a tuple, such as the row/column pair of ``df.loc[mask, cols]`` --
    as a child of this node, so a DataOp left unbound here stays in the DAG with
    no consumer and the topological walk then trips over a node it has no
    in-degree entry for.
    """
    return binder.bind(value)


def _getitem_key_name(key) -> str:
    """Display label for a ``df[key]`` indexer.

    A DataOp key is labelled by its impl class (``BinOp``, ...) rather than its
    repr; a tuple key -- the two axes of ``df.loc[rows, cols]`` -- is labelled
    per axis, so the op reads as ``(BinOp, ['gid'])`` instead of embedding a
    DataOp repr.
    """
    if isinstance(key, DataOp):
        return key._skrub_impl.__class__.__name__
    if isinstance(key, tuple):
        return f"({', '.join(_getitem_key_name(k) for k in key)})"
    return str(key)


# Method groups reachable from stratum's two execution modes: which of the two a
# group belongs to depends on the estimator kind (see BaseEstimatorOp subclasses).
_EXECUTABLE_KWARGS_KEYS = frozenset({"fit", "fit_transform", "transform"}) | RESPONSE_MODES
# Groups skrub routes to a method stratum never calls. Ignoring them would change
# results with no trace, so they are rejected instead. `score` is one Stratum will not
# gain: a search names its metric (ADR 0004).
_UNSUPPORTED_KWARGS_KEYS = frozenset({"score"})


def _bind_apply_kwargs(binder: OperandBinder, estimator_class, estimator,
                       kwargs: dict | None) -> dict:
    """Bind the DataOps nested in an Apply's per-method kwargs to OperandRefs.

    ``.skb.apply()`` always passes all seven method groups, most of them None.
    Dropping the empty ones keeps the op's config (and therefore its CSE key)
    down to the groups that carry something.

    Every remaining group is bound, including one this estimator kind will never
    call (a transformer is fitted through ``fit_transform``, so its ``fit`` group
    is dead in skrub too). Graph extraction already reported the DataOps inside
    each group as children of the Apply node, so a group left unbound leaves those
    ops in the DAG with no consumer, and the topological walk then trips over a
    node it has no in-degree entry for.
    """
    bound = {}
    for method, group in (kwargs or {}).items():
        if group is None:
            continue
        if any(isinstance(v, (BaseChoice, Match)) for v in _iter_nested(group)):
            # A choice here would have to expand the parameter grid; its DataOp
            # outcomes are graph children, so silently keeping it would orphan them.
            raise NotImplementedError(
                f"A choice inside `{method}_kwargs` of `.skb.apply()` is not "
                f"supported yet (choice found in {group!r}).")
        if method in _UNSUPPORTED_KWARGS_KEYS or method not in _EXECUTABLE_KWARGS_KEYS:
            raise NotImplementedError(
                f"`{method}_kwargs` passed to `.skb.apply()` is not supported yet: "
                f"stratum never calls `{method}()`, so the arguments would be "
                f"silently dropped.")
        if method not in estimator_class.callable_kwargs_keys(estimator):
            # Dead for this estimator kind in skrub as well, so it is bound (to keep
            # the DAG consistent) but never splatted. Mirroring skrub here rather
            # than raising keeps a pipeline skrub accepts working.
            logger.warning(
                f"`{method}_kwargs` is ignored for a {estimator_class.logical_family.lower()}: "
                f"it is fitted through `{estimator_class.fit_kwargs_key}()` and applied "
                f"through `{estimator_class.call_kwargs_key}()`.")
        bound[method] = binder.bind(group)
    return bound


def _iter_nested(value):
    """Yield ``value`` and every item nested in its tuples/lists/dicts."""
    yield value
    if isinstance(value, (tuple, list)):
        for v in value:
            yield from _iter_nested(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_nested(v)


def _apply_estimator_op(impl: Apply, estimator, ids_to_ops: dict, feeds_estimator: bool = False,
                        choice_ops: dict | None = None) -> Op:
    """Build the TransformerOp/EstimatorOp for one concrete estimator of an Apply impl.

    ``estimator`` is ``impl.estimator`` itself, or one outcome of it when the
    Apply's estimator is a ``Choice``. Each call uses its own binder so every op
    gets X as OperandRef(0) and its own de-duplicated inputs list.

    ``feeds_estimator`` says another Apply consumes this one's output. skrub then
    fit_transforms and transforms any estimator that can, a predictor included (a
    KMeans feeds on its distances, not its labels), so such an estimator becomes a
    TransformerOp here.

    ``choice_ops`` is :func:`as_op`'s: a choice in the Apply's other fields (``y``,
    ``cols``, ...) binds to the one ChoiceOp built for it.
    """
    if estimator is None or (isinstance(estimator, str) and estimator == "passthrough"):
        # Same normalization skrub's _wrap_estimator applies at fit time; needed
        # here already because BaseEstimatorOp clones its estimator on construction.
        estimator = PassThrough()
    if hasattr(estimator, "predict") and not (feeds_estimator and hasattr(estimator, "transform")):
        estimator_class = PredictorOp
    else:
        estimator_class = TransformerOp
    binder = OperandBinder(ids_to_ops, choice_ops)
    binder.ref(impl.X)  # OperandRef(0)
    param_refs = {k: binder.ref(v) for k, v in estimator.get_params().items()
                  if isinstance(v, DataOp) and id(v) in ids_to_ops}
    y = _bind_or_value(binder, impl.y)
    cols = _bind_or_value(binder, impl.cols)
    exclude_cols = _bind_or_value(binder, impl.exclude_cols)
    no_wrap = _bind_or_value(binder, impl.no_wrap)
    kwargs = _bind_apply_kwargs(binder, estimator_class, estimator, impl.kwargs)
    op = estimator_class(
        estimator=estimator,
        y=y,
        cols=cols,
        exclude_cols=exclude_cols,
        no_wrap=no_wrap,
        allow_reject=impl.allow_reject,
        unsupervised=impl.unsupervised,
        kwargs=kwargs,
        param_refs=param_refs,
        feeds_estimator=feeds_estimator,
    )
    op.inputs = binder.inputs
    return op


def _outcome_display(estimator) -> str:
    """Human-readable label for an estimator outcome (None/'passthrough' -> PassThrough)."""
    if estimator is None or (isinstance(estimator, str) and estimator == "passthrough"):
        return PassThrough.__name__
    return type(estimator).__name__


# skrub resolves choices nested in these exact types, not in their subclasses (it
# would not know how to rebuild one), so neither does the expansion below.
_CHOICE_CONTAINERS = (list, tuple, set, frozenset)


def _iter_choices(value):
    """Yield every choice (or ``Match``) in ``value`` that skrub would resolve.

    Walks what skrub's evaluation walks: a choice's outcomes, an estimator's
    parameters, and the built-in containers. DataOps are not entered: they are
    graph nodes, converted on their own.
    """
    if isinstance(value, Match):
        yield value
    elif isinstance(value, BaseChoice):
        yield value
        if isinstance(value, Choice):
            for outcome in value.outcomes:
                yield from _iter_choices(outcome)
    elif isinstance(value, BaseEstimator):
        yield from _iter_choices(value.get_params(deep=False))
    elif type(value) is dict:
        for v in value.values():
            yield from _iter_choices(v)
    elif type(value) in _CHOICE_CONTAINERS:
        for v in value:
            yield from _iter_choices(v)
    elif type(value) is slice:
        for bound in (value.start, value.stop, value.step):
            yield from _iter_choices(bound)


def _contains_choice(value) -> bool:
    return next(_iter_choices(value), None) is not None


def _outcome_label(outcome, index: int, estimator_slot: bool) -> str:
    """Name of an unnamed choice outcome in a grid point's name path.

    In the estimator slot, the outcome is the estimator (None/'passthrough' mean a
    PassThrough); inside an estimator's parameters it is the parameter value.
    """
    if estimator_slot:
        return _outcome_display(outcome)
    if isinstance(outcome, BaseEstimator):
        return type(outcome).__name__
    if outcome is None or isinstance(outcome, (Number, str)):
        return str(outcome)
    name = getattr(outcome, "__name__", None)  # functions and classes
    # Anything else gets the positional label Value(Choice) outcomes get.
    return name if isinstance(name, str) else f"Opt{index}"


def _expand_choice_outcomes(choice, outcomes, outcome_names, chosen: dict,
                            estimator_slot: bool):
    key = id(choice)
    if key in chosen:
        # skrub keys a choice by identity: every use of one choice object takes the
        # same outcome, and it is one grid dimension, so it adds nothing to the name.
        yield from _expand_choices(outcomes[chosen[key]], chosen, estimator_slot)
        return
    for i, outcome in enumerate(outcomes):
        if isinstance(outcome, DataOp):
            raise NotImplementedError(
                "A choice in the estimator of `.skb.apply()` only supports concrete "
                "outcomes (estimators, None/'passthrough' or parameter values); "
                f"DataOp outcomes are not supported (choice {choice.name!r}).")
        named = outcome_names is not None
        label = outcome_names[i] if named else _outcome_label(outcome, i, estimator_slot)
        # An unnamed nested choice is described by its own outcomes; the leaf always
        # contributes, so every grid point gets a name.
        prefix = [] if isinstance(outcome, BaseChoice) and not named else [(choice.name, label)]
        for path, concrete, now_chosen in _expand_choices(outcome, {**chosen, key: i},
                                                          estimator_slot):
            yield prefix + path, concrete, now_chosen


def _expand_items(items: list, chosen: dict):
    """The Cartesian product of the expansions of ``items``, in order."""
    if not items:
        yield [], [], chosen
        return
    for head_path, head, head_chosen in _expand_choices(items[0], chosen, False):
        for tail_path, tail, tail_chosen in _expand_items(items[1:], head_chosen):
            yield head_path + tail_path, [head] + tail, tail_chosen


def _expand_choices(value, chosen: dict, estimator_slot: bool):
    """Yield ``(name_path, concrete_value, chosen)`` for every grid point of ``value``.

    ``name_path`` is a list of ``(choice_name, outcome_label)`` pairs, ChoiceOp's
    internal ``outcome_names`` format. ``chosen`` maps ``id(choice)`` to the outcome
    fixed for it so far on this grid point. The grid is conditional, as skrub's: a
    choice nested in an outcome only varies when that outcome is taken.
    """
    if isinstance(value, Match):
        raise NotImplementedError(
            "`.match()` on a choice inside the estimator of `.skb.apply()` is not "
            f"supported yet (it matches choice {value.choice.name!r}).")
    if isinstance(value, Choice):
        yield from _expand_choice_outcomes(value, value.outcomes, value.outcome_names,
                                           chosen, estimator_slot)
    elif isinstance(value, DiscretizedNumericChoice):
        yield from _expand_choice_outcomes(value, list(value.grid), None, chosen, False)
    elif isinstance(value, BaseChoice):
        raise NotImplementedError(
            f"{value!r} is a continuous range, which a grid search cannot enumerate. "
            "Pass `n_steps` to search a grid of values from it.")
    elif not _contains_choice(value):
        # Returned as is, so an estimator without a choice keeps its identity.
        yield [], value, chosen
    elif isinstance(value, BaseEstimator):
        for path, params, now_chosen in _expand_choices(value.get_params(deep=False),
                                                        chosen, False):
            # As skrub's `handle_estimator`: a clone configured with the resolved params.
            estimator = clone(value)
            estimator.set_params(**params)
            yield path, estimator, now_chosen
    elif type(value) is dict:
        for path, items, now_chosen in _expand_items(list(value.values()), chosen):
            yield path, dict(zip(value.keys(), items)), now_chosen
    else:
        for path, items, now_chosen in _expand_items(list(value), chosen):
            yield path, type(value)(items), now_chosen


def _expand_estimator_choices(estimator) -> list[tuple[list, object]]:
    """Expand every choice in an Apply's estimator into ``(name_path, estimator)`` leaves.

    The choices may be the estimator itself (possibly nesting further choices) or sit
    in its parameters at any depth, e.g. ``LogisticRegression(C=choose_from(...))`` or
    a ``Pipeline`` step. All of them collapse into one flat list of concrete
    estimators, one per point of the parameter grid skrub would search. An
    intermediate (nested) choice only contributes to the path when its outcome is
    named; a leaf always contributes its outcome name or a label for its value.
    """
    return [(path, est) for path, est, _ in _expand_choices(estimator, {}, True)]


def check_choices_not_shared(data_ops) -> None:
    """Refuse a choice object used both in an Apply's estimator and anywhere else.

    skrub keys a choice by identity, so one ``choose_from`` used in two places is a
    single grid dimension: both uses always take the same outcome. Conversion gives
    every choice outside an Apply's estimator one ChoiceOp, shared by all its uses
    (see :func:`_choice_op`), so those keep skrub's grid. The choices of an Apply's
    estimator are instead expanded with that Apply alone, which would search a use
    elsewhere independently, a different grid, so this is refused rather than
    silently diverging from skrub.
    """
    owners = {}  # id(choice) -> (owner key, DataOp first found using it)
    for data_op in data_ops:
        impl = data_op._skrub_impl
        uses = []
        for field_name in impl._fields:
            value = getattr(impl, field_name)
            if isinstance(impl, Apply) and field_name == "estimator":
                uses.append((id(data_op), value))
            else:
                uses.append(("shared", value))
        for owner_key, value in uses:
            found = [c for c in _iter_choices(value) if isinstance(c, BaseChoice)]
            for choice in {id(c): c for c in found}.values():
                owner, first_user = owners.setdefault(id(choice), (owner_key, data_op))
                if owner != owner_key:
                    raise NotImplementedError(
                        f"The choice {choice!r} is used by more than one operator "
                        f"({first_user.__skrub_short_repr__()} and "
                        f"{data_op.__skrub_short_repr__()}), at least one of them as "
                        "the estimator of `.skb.apply()`. skrub searches it as one "
                        "parameter, so every use takes the same outcome; stratum does "
                        "not support that yet.")


def _choice_op(choice, ids_to_ops: dict, choice_ops: dict) -> "ChoiceOp":
    """The ChoiceOp of a choice used outside an Apply's estimator, built once per choice.

    skrub keys a choice by identity, so all uses of one choice object -- a
    ``Value(Choice)`` node, or a choice nested in another op's arguments such as the
    key of ``X[[choose_from(...)]]`` -- share this one op, and choice unrolling then
    searches it as one grid dimension. ``choice_ops`` maps ``id(choice)`` to the
    ops built so far; the outcome ops get their output edge to the ChoiceOp here.
    """
    op = choice_ops.get(id(choice))
    if op is not None:
        return op
    if isinstance(choice, Match):
        raise NotImplementedError(
            f"`.match()` on a choice is only supported in the estimator of "
            f"`.skb.apply()` (it matches choice {choice.choice.name!r}).")
    if isinstance(choice, Choice):
        outcomes, names = choice.outcomes, choice.outcome_names
    elif isinstance(choice, DiscretizedNumericChoice):
        outcomes = list(choice.grid)
        names = [_outcome_label(v, i, False) for i, v in enumerate(outcomes)]
    else:
        raise NotImplementedError(
            f"{choice!r} is a continuous range, which a grid search cannot enumerate. "
            "Pass `n_steps` to search a grid of values from it.")
    # Outcomes are consumed positionally by ChoiceOp.process: one input per outcome.
    inputs = [_outcome_op(o, ids_to_ops, choice_ops) for o in outcomes]
    op = ChoiceOp(names, len(outcomes), choice.name)
    op.inputs = inputs
    for in_op in inputs:
        in_op.add_output(op)
    choice_ops[id(choice)] = op
    return op


def _outcome_op(outcome, ids_to_ops: dict, choice_ops: dict) -> Op:
    """The op producing one outcome of a choice (see :func:`_choice_op`)."""
    if isinstance(outcome, DataOp):
        return ids_to_ops[id(outcome)]
    if isinstance(outcome, (BaseChoice, Match)):
        return _choice_op(outcome, ids_to_ops, choice_ops)
    binder = OperandBinder(ids_to_ops, choice_ops)
    value = binder.bind(outcome)
    if not binder.inputs:
        return ValueOp(outcome)
    # A container holding DataOps or choices (``choose_from([["a", other], ...])``)
    # is assembled at runtime from the values they produce.
    op = CallOp(name=type(outcome).__name__, func=_identity, args=(value,), kwargs={})
    op.inputs = binder.inputs
    for in_op in op.inputs:
        in_op.add_output(op)
    return op


def _identity(value):
    return value


# TODO: Move this to frontend package
def as_op(data_op: DataOp, ids_to_ops: dict, env: dict | None = None, feeds_estimator: bool = False,
          choice_ops: dict | None = None) -> Op:
    """Convert a single skrub DataOp into an Op, building its de-duplicated
    ``inputs`` list and operand references in one canonical field walk.

    ``ids_to_ops`` maps ``id(DataOp) -> Op`` and must already contain every input
    of ``data_op`` (guaranteed by converting in topological order). Output edges
    are wired here too: each input op gets ``data_op``'s Op added to its outputs.

    ``env`` is the runtime environment (variable name -> value), when known at
    compile time. A ``Var`` whose name is bound in ``env`` is then resolved to a
    constant ``ValueOp`` instead of a ``VariableOp``, so the scheduler needs no
    environment to feed it at runtime.

    ``feeds_estimator`` says another Apply consumes ``data_op``'s value (see
    ``_apply_estimator_op``).

    ``choice_ops`` maps ``id(choice) -> ChoiceOp`` across one conversion, so that
    every use of a choice outside an Apply's estimator -- a ``Value(Choice)`` node
    or a choice nested in an op's arguments -- binds to one shared ChoiceOp (see
    :func:`_choice_op`). Left ``None``, the choices of this op get fresh ones.
    """
    if choice_ops is None:
        choice_ops = {}
    impl = data_op._skrub_impl
    is_X = is_y = False
    if impl is not None:
        is_X = impl.is_X
        is_y = impl.is_y
    binder = OperandBinder(ids_to_ops, choice_ops)
    return_op = None

    if isinstance(impl, Value):
        # A choice (or a container holding one) becomes the choice's shared ChoiceOp,
        # anything else a ValueOp.
        return_op = _outcome_op(impl.value, ids_to_ops, choice_ops)
    elif isinstance(impl, CallMethod):
        binder.ref(impl.obj)  # implicit primary operand -> OperandRef(0)
        return_op = MethodCallOp(impl.method_name, binder.bind_seq(impl.args), binder.bind_map(impl.kwargs))
        return_op.inputs = binder.inputs
    elif isinstance(impl, Call):
        if _contains_choice(impl.func):
            raise NotImplementedError(
                f"Calling a choice of functions is not supported yet ({impl.func!r}).")
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
        return_op = GetItemOp(key=key, name=_getitem_key_name(impl.key))
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
        if _contains_choice(impl.estimator):
            # The choices in an estimator -- the estimator itself, or ones nested in
            # its parameters -- expand to a ChoiceOp over one estimator op per grid
            # point (mirroring Value(Choice) above), so choice unrolling and grid
            # search work over the alternatives and every op after conversion holds
            # concrete parameters. All of them are flattened into a single ChoiceOp;
            # the leaf name paths use ChoiceOp's combi format so they concatenate
            # correctly if choice_unrolling later combines this choice with a
            # downstream one.
            leaves = _expand_estimator_choices(impl.estimator)
            outcome_ops = [_apply_estimator_op(impl, est, ids_to_ops, feeds_estimator, choice_ops)
                           for _, est in leaves]
            for est_op in outcome_ops:
                # The trailing edge-wiring below only covers the returned op.
                for in_op in est_op.inputs:
                    in_op.add_output(est_op)
            return_op = ChoiceOp(outcome_names=[path for path, _ in leaves],
                                 append_choice_name=False, inputs=outcome_ops)
        else:
            return_op = _apply_estimator_op(impl, impl.estimator, ids_to_ops, feeds_estimator,
                                            choice_ops)
    elif isinstance(impl, SplitX):
        for field_name in impl._fields:
            for child in _collect_child_data_ops(getattr(impl, field_name)):
                binder.ref(child)
        return_op = SplitXOp(impl)
        return_op.inputs = binder.inputs
    elif isinstance(impl, Var):
        if env is not None and impl.name in env:
            # Resolve the variable to a compile-time constant; the runtime no
            # longer needs the environment to feed it.
            return_op = ValueOp(env[impl.name])
        elif env is not None and impl.becomes_default and impl.value is not NULL:
            # `skrub.var(name, value, becomes_default=True)`: an environment that
            # does not bind the variable falls back to its value, as in skrub.
            return_op = ValueOp(impl.value)
        else:
            return_op = VariableOp(name=impl.name, value=impl.value)
    elif isinstance(impl, Scoring):
        from stratum.optimizer.logical._candidate_ops import DeclaredScoringOp
        binder.ref(impl.pred)  # OperandRef(0): the value passed through
        return_op = DeclaredScoringOp(scorers=binder.bind(impl.scorers))
        return_op.inputs = binder.inputs
    elif isinstance(impl, Concat):
        from stratum.optimizer.logical._dataframe_ops import ConcatOp
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
