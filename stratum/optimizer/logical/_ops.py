from __future__ import annotations
import operator
from types import SimpleNamespace
from typing import Callable

from joblib import parallel_config
from sklearn import clone
from sklearn.base import BaseEstimator
from skrub._data_ops._data_ops import DataOp, SplitX, _wrap_estimator
from pandas import DataFrame, Series
import polars as pl
from polars import DataFrame as PlDataFrame
from stratum.frontend._pandas_calls import bind_pandas_method
from stratum.frontend._skrub_graph import _collect_child_data_ops
from stratum.optimizer.logical import _schema
# Shared IR foundation. Re-exported below so existing ``from ..._ops import X``
# call sites (OutputType, OperandRef, is_frame_like, config_key, ...) keep working.
from stratum.optimizer.logical._base import (
    IRNode, IndexEffect, OutputType, FRAME_TYPES, FITTING_MODE, Fitted, has_default_labels,
    is_frame_like, OperandRef,
    _resolve_operand, _resolve_args, _resolve_kwargs, remap_operand_refs,
    config_key, estimator_key, clone_value, rebuild_dict,
    _ALL_SELECTOR_KEY, _GRAPH_PARAM_KEY,
)
import logging
import os
logger = logging.getLogger(__name__)


def _operand_index_from_impl(skrub_impl) -> dict:
    """Map id(DataOp) -> operand index, in the canonical field-walk order.

    Uses the same walk as graph extraction (`_collect_child_data_ops` over
    `impl._fields`), de-duplicating repeated DataOps to the same index so that
    the map follows the converter's input ordering.
    """
    index: dict = {}
    for field_name in skrub_impl._fields:
        for data_op in _collect_child_data_ops(getattr(skrub_impl, field_name)):
            if id(data_op) not in index:
                index[id(data_op)] = len(index)
    return index


class Op(IRNode):
    """Logical operator: a backend-agnostic node in the pre-lowering IR.

    Adds the two concerns specific to the logical layer on top of the shared
    :class:`~stratum.optimizer.logical._base.IRNode` structure: CSE structure keys
    and choice detection. Physical operator families also derive from
    ``IRNode``; concrete implementations may additionally inherit a logical family.
    """

    def is_choice(self) -> bool:
        return isinstance(self, ChoiceOp)

    def structure_key(self):
        """Return a hashable structural key for computations eligible for CSE.

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


def substitute_op(op: Op, new_op: Op, keep_kind: bool = False) -> Op:
    """Put ``new_op`` in ``op``'s place and return it.

    ``new_op`` takes ``op``'s inputs and outputs, and ``op``'s producers feed it
    instead; extraction rewires the consumers. ``keep_kind``: ``new_op``
    produces what ``op``'s receiver (input 0) is, a frame or a series, as a row
    selection or a column-wise map does.
    """
    new_op.inputs, new_op.outputs = op.inputs, op.outputs
    if keep_kind and op.inputs:
        new_op.output_type = op.inputs[0].output_type
    op.replace_output_of_inputs(new_op)
    return new_op


class ImplOp(Op):
    index_effect = IndexEffect.KEEP

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
                return rebuild_dict(value, ((key, replace_dataop(val)) for key, val in value.items()))
            else:
                return value

        return SimpleNamespace(**{field: replace_dataop(getattr(self.skrub_impl, field)) for field in self.skrub_impl._fields})

    def process(self, mode: str, inputs: list):
        impl = self.skrub_impl
        if hasattr(impl, "eval"):
            # An eval method combines dependency requests with computation.
            # Resume its generator with the corresponding computed inputs.
            # Indices are assigned in yield order (de-duplicating repeated DataOps), matching the
            # order in which inputs are consumed.
            index = {}
            last_yield = None
            # Variables are resolved to constants at compile time, so the skrub
            # impl needs no environment -- its inputs arrive via the generator.
            gen = impl.eval(mode=mode, environment={})
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
            return impl.compute(ns, mode, {})


class SplitXOp(Op):
    """Temporary X boundary carrying skrub's declared splitter until split planning."""
    index_effect = IndexEffect.KEEP

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

# Scoring determines the requested test response (see ``ScoreCandidatesOp``).
# Planning resolves it to a supported estimator method or the fallback method.
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
    # Row-preserving, and positional on both backends.
    index_effect = IndexEffect.KEEP
    aligns_by_label = False

    @classmethod
    def callable_kwargs_keys(cls, estimator) -> frozenset:
        """Return the `.skb.apply()` kwargs groups this operator can pass to methods.

        Include a response group only if the estimator exposes that method.
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
        # the last estimator, so it uses its fallback method in response passes.
        self.feeds_estimator = feeds_estimator
        # Resolve response methods once during planning. Graph-fed parameters
        # are still unresolved, so availability is checked on the prototype.
        self.supported_modes = frozenset(m for m in self.response_modes
                                         if hasattr(self.estimator, m))
        self.parallelism = os.cpu_count() # TODO: Set this during physical planning.

    #: Keyed when the estimator is known and deterministic for its configuration.
    cacheable = True
    #: Fitted state consists of the estimator produced during training.
    has_state = True

    def lineage_config(self):
        """Encode the unfitted prototype cloned by each fitting pass."""
        from stratum.optimizer._estimator_lineage import estimator_lineage, selector_lineage

        config = {name: getattr(self, name) for name in type(self).fields}
        params = self.original_estimator.get_params()
        config["estimator"] = estimator_lineage(
            self.original_estimator, graph_inputs=tuple(params[name] for name in self.param_refs))
        config["cols"] = selector_lineage(self.cols)
        config["exclude_cols"] = selector_lineage(self.exclude_cols)
        return config

    def effective_mode(self, mode: str) -> str:
        """Resolve a requested response method during planning.

        Unsupported responses and outputs consumed by another estimator use
        the fallback: ``transform`` for transformers and ``predict`` for
        predictors, matching skrub's ``Apply``.
        """
        if mode in self.supported_modes and not self.feeds_estimator:
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
        """Resolve estimator arguments into a tuple for the execution task.

        Worker-process execution requires the estimator and argument values
        themselves to support serialization.
        """
        fitting = self.produces_state
        x = inputs[0]
        assert x is not None, f"X is None for {self}"
        y = (inputs[self.y.k] if isinstance(self.y, OperandRef) else self.y) if fitting else None
        # Clone the unfitted prototype for each fit to isolate fold state.
        # Response operators borrow the estimator from their fitted-state input.
        if fitting:
            estm = clone(self.original_estimator)
        else:
            estm = inputs[self.state_ref.k]
        place_holders = {name: inputs[ref.k] for name, ref in self.param_refs.items()}
        # Graph-fed hyperparameters configure the estimator before fitting. In a
        # response pass ``estm`` is already fitted, and CatBoost rejects any
        # ``set_params`` call at that point (even an empty one).
        if fitting and place_holders:
            estm.set_params(**place_holders)
        cols = _resolve_operand(self.cols, inputs)
        exclude_cols = _resolve_operand(self.exclude_cols, inputs)
        no_wrap = _resolve_operand(self.no_wrap, inputs)
        # Only fitting evaluates the fit group. Response operators omit it, so
        # branches used only by fit kwargs do not run on test rows.
        fit_kwargs = self.method_kwargs(self.fit_kwargs_key, inputs) if fitting else {}
        call_kwargs = {} if fitting else self.method_kwargs(mode, inputs)
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
        # A separate task function supports local and worker-process execution.
        task_data = self.extract_args_from_inputs(mode, inputs)
        process_task = self.get_process_task()
        result, estimator = process_task(task_data)
        return Fitted(result, estimator) if self.produces_state else result

    def get_process_task(self):
        raise NotImplementedError(f"get_process_task must be implemented in {self.__class__.__name__}")

class PredictorOp(BaseEstimatorOp):
    logical_family = "Predictor"
    # Training predictions execute in a separate operator with a state input.
    fit_kwargs_key = "fit"
    call_kwargs_key = "predict"
    response_modes = RESPONSE_MODES
    fallback_mode = "predict"
    fitting_mode = "fit"

    def get_process_task(self):
        return process_estimator_task

class TransformerOp(BaseEstimatorOp):
    logical_family = "Transformer"
    # A transformer is fitted through fit_transform(), so (as in skrub) the "fit"
    # group never applies to one; its response mode is transform().
    fit_kwargs_key = "fit_transform"
    call_kwargs_key = "transform"

    def process(self, mode: str, inputs: list):
        raise NotImplementedError("TransformerOp must be lowered before execution")

    def get_process_task(self):
        return process_transformer_task

class DummyConfigManager:
    """A context manager that leaves parallel execution settings unchanged."""
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
    """Whether ``estimator`` -- and every estimator it wraps -- accepts polars input.

    A meta-estimator (a pipeline, a voting ensemble) hands its input to the
    estimators it wraps, so it supports polars only if they all do: sklearn
    tries to set ``feature_names_in_`` on a wrapped LightGBM model, which is a
    read-only property there.
    """
    if not _supports_polars_itself(estimator):
        return False
    try:
        params = estimator.get_params(deep=True)
    except Exception:  # noqa: BLE001 - not an sklearn-style estimator
        return True
    return all(_supports_polars_itself(v) for v in params.values()
               if isinstance(v, BaseEstimator))


def _supports_polars_itself(estimator):
    module = estimator.__class__.__module__
    is_sklearn = module.startswith("sklearn.") or module.startswith("skrub.")
    is_stratum = module.startswith("stratum.") and estimator.__class__.__name__.startswith("Rusty")
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
        from stratum.optimizer.physical._conversion import to_pandas
        x = to_pandas(x)
        if y is not None and mode in (FITTING_MODE, "fit"):
            # y may already be pandas: the plan converts edges explicitly now.
            y = to_pandas(y)
    return converted, x, y

def process_estimator_task(task_data):
    """Execute a PredictorOp task, locally or in a worker process."""
    (estimator, x, y, cols, exclude_cols, no_wrap, allow_reject, unsupervised, kwargs, mode, parallelism) = task_data
    fit_kwargs, call_kwargs = kwargs
    _, x, y = check_estm_inputs(estimator, mode, x, y)
    if mode == "fit":
        # Training predictions, when required, execute in a separate operator.
        estimator = _wrap_estimator(estimator, cols, exclude_cols=exclude_cols, no_wrap=no_wrap,
                                    allow_reject=allow_reject, X=x)
        y_arg = () if unsupervised else (y,)
        estimator.fit(x, *y_arg, **fit_kwargs)
        return None, estimator
    elif mode in RESPONSE_MODES:
        # `mode` is already this op's effective mode, so the method is one it serves.
        result = getattr(estimator, mode)(x, **call_kwargs)
        return result, estimator
    else:
        raise ValueError(f"Mode {mode} not supported for PredictorOp.")

def process_transformer_task(task_data):
    """Execute a TransformerOp task, locally or in a worker process."""
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
            # Planning resolves transformer responses to transform.
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
        # Keep distinct choice objects as separate dimensions of the search grid.
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
    # The encoder supports plain constants and rejects dataframe and array payloads.
    cacheable = True

    @property
    def index_effect(self):
        return IndexEffect.DEFAULT if has_default_labels(self.value) else IndexEffect.NEW

    def __init__(self, value):
        super().__init__(name="DataFrame" if isinstance(value, DataFrame) else str(value))
        self.value = value

    def clone(self):
        raise ValueError(f"We should not clone ValueOp objects.")

class MethodCallOp(Op):
    fields = ["method_name", "args", "kwargs"]

    #: What is known about the pandas methods that stay recorded calls, by name:
    #: ``(keeps_kind, keeps_rows)``. ``keeps_kind``: the result is what the
    #: receiver is, a frame or a series. ``keeps_rows``: it holds the receiver's
    #: rows, in order, with their labels.
    OPAQUE_METHODS = {
        "where": (True, True), "map": (True, True), "fillna": (True, True),
        "rename": (True, True),
        "sort_values": (True, False), "dropna": (True, False), "set_axis": (True, False),
        "drop": (True, False), "drop_duplicates": (True, False), "sample": (True, False),
        "mask": (False, True), "astype": (False, True), "clip": (False, True),
        "abs": (False, True), "round": (False, True), "replace": (False, True),
        "isna": (False, True), "notna": (False, True), "isnull": (False, True),
        "notnull": (False, True), "isin": (False, True), "between": (False, True),
        "copy": (False, True),
    }

    @property
    def index_effect(self):
        _, keeps_rows = self.OPAQUE_METHODS.get(self.method_name, (False, False))
        return IndexEffect.COMMON if keeps_rows else IndexEffect.NEW

    def __init__(self, method_name: str, args = None, kwargs = None):
        super().__init__(name=method_name)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def bind_pandas(self, owner=None) -> dict | None:
        """The recorded call's arguments by parameter name, or ``None`` when pandas
        would reject them.

        It binds against ``owner``'s method, by default the receiver's own class,
        since a Series and a DataFrame method can take different parameters.
        """
        if owner is None:
            owner = Series if self.inputs[0].output_type is OutputType.SERIES else DataFrame
        return bind_pandas_method(owner, self.method_name, self.args, self.kwargs)

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


class GetItemOp(Op):
    fields = ["key", "is_filter"]
    cacheable = True
    
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
            # Approximate a SERIES key as a row mask (`df[df["x"] > 0]`).
            # TODO: Check boolean dtype; a Series of column labels selects columns
            # and must not preserve the entire input schema.
            key_op = self.inputs[self.key.k]
            self.output_schema = schema if key_op.output_type is OutputType.SERIES else None
        else:
            self.output_schema = _schema.project_columns(schema, self.key)

    # Execution lives in the physical impls (physical/_getitem_execs.py).

class BinOp(Op):
    fields = ["op", "left", "right"]
    index_effect = IndexEffect.COMMON
    
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
        return apply_binary(self.op, left, right)


_COMPARISONS = {operator.lt, operator.le, operator.gt, operator.ge, operator.eq}


def apply_binary(op, left, right):
    """``op(left, right)``, with pandas' result for a missing operand on polars.

    Match pandas' NaN comparison convention: False except for ``!=``. Polars
    propagates null, so replace those null results before later mask operations.
    This does not model pandas nullable dtypes, which can preserve missingness.
    """
    result = op(left, right)
    if isinstance(result, (pl.Series, pl.Expr)):
        if op in _COMPARISONS:
            return result.fill_null(False)
        if op is operator.ne:
            return result.fill_null(True)
    return result


class UnaryOp(Op):
    fields = ["op", "operand"]
    index_effect = IndexEffect.COMMON

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
