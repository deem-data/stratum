"""Convert skrub DataOp DAGs into logical operator DAGs.

Bind graph operands, resolve variables, and expand estimator choices during
conversion. Logical rewrites run on the resulting operator DAG.
"""
from __future__ import annotations

import logging
from collections import deque
from numbers import Number

from sklearn import clone
from sklearn.base import BaseEstimator
from skrub._data_ops._choosing import BaseChoice, Choice, DiscretizedNumericChoice, Match
from skrub._data_ops._data_ops import (
    Apply, BinOp as SkrubBinOp, Call, CallMethod, Concat, DataOp, GetAttr,
    FreezeAfterFit, GetItem, Scoring, SplitX, UnaryOp as SkrubUnaryOp, Value, Var,
)
from skrub._data_ops._subsampling import SubsamplePreviews
from skrub._data_ops._utils import NULL
from skrub._utils import PassThrough

from stratum.frontend._skrub_graph import _collect_child_data_ops, build_graph
from stratum.optimizer.logical._base import OperandRef, rebuild_dict
from stratum.optimizer.logical._ops import (
    RESPONSE_MODES, BinOp, CallOp, ChoiceOp, GetAttrOp, GetItemOp, ImplOp,
    MethodCallOp, Op, PredictorOp, SplitXOp, TransformerOp, UnaryOp, ValueOp,
    VariableOp, _getitem_key_name,
)
from stratum.utils._utils import start_time, log_time

logger = logging.getLogger(__name__)


def convert_to_ops(dag: DataOp, env: dict = None) -> Op:
    """Convert a Skrub DataOp DAG to stratum's logical IR (Op DAG).

    Single fused topological pass: ``as_op`` builds each op together with its
    de-duplicated ``inputs`` list, operand references, and output edges. Inputs
    are resolved through ``ids_to_ops`` (keyed by ``id(DataOp)``), which is
    guaranteed populated because we walk in topological, inputs-first order.
    """
    start = start_time()
    children, nodes, parents = get_dataops_graph(dag)
    check_choices_not_shared(nodes.values())
    order = topological_traverse(nodes, parents, children)
    root_id = order[-1]
    feeds_estimator = _apply_fed_nodes(nodes, parents, order)

    # id(DataOp) -> Op. Keyed by DataOp identity (not the graph's node keys) so
    # as_op's operand binder can resolve inputs found in the impl fields directly.
    ids_to_ops = {}
    # id(choice) -> ChoiceOp: one op per choice object, shared by all its uses.
    choice_ops = {}
    for node_key in order:
        skrub_op = nodes[node_key]
        impl = skrub_op._skrub_impl
        if isinstance(impl, SubsamplePreviews):
            # Drop the preview node: route its single input straight to consumers.
            # Consumers reference this node's DataOp in their fields, so mapping it
            # to the input op makes them wire to the input op directly.
            input_key = children.get(node_key, [])[0]
            ids_to_ops[id(skrub_op)] = ids_to_ops[id(nodes[input_key])]
            continue
        ids_to_ops[id(skrub_op)] = as_op(skrub_op, ids_to_ops, env,
                                         feeds_estimator=node_key in feeds_estimator,
                                         choice_ops=choice_ops)

    root = ids_to_ops[id(nodes[root_id])]
    log_time("conversion took", start)
    return root


def topological_traverse(nodes, parents, children):
    """ Compute a topological order of the DAG in skrub IR. """
    # Compute in-degree (number of children for each node)
    indegree = {n: len(children.get(n, [])) for n in nodes}

    # Initialize queue with nodes having no children
    queue = deque([n for n, deg in indegree.items() if deg == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for parent in parents.get(node, []):
            indegree[parent] -= 1
            if indegree[parent] == 0:
                queue.append(parent)

    return topo_order


def get_dataops_graph(dag: DataOp) -> tuple[dict, dict, dict]:
    start = start_time()
    g = build_graph(dag)
    nodes = g["nodes"]
    parents = g["parents"]
    children = g["children"]
    log_time("Conversion dag took", start)
    return children, nodes, parents


def _apply_fed_nodes(nodes: dict, parents: dict, order: list) -> set:
    """Return node keys with a downstream `.skb.apply()` consumer.

    An Apply feeding another Apply must transform or predict for that consumer
    during response passes. Determine this dependency from all graph edges,
    including estimator parameters. Unlike skrub's evaluation-time check, this
    rule does not depend on traversal order when Apply and non-Apply paths share
    a node.
    """
    fed = set()
    for key in reversed(order):
        if any(isinstance(nodes[p]._skrub_impl, Apply) or p in fed
               for p in parents.get(key, ())):
            fed.add(key)
    return fed


class OperandBinder:
    """Build an operator's ``inputs`` list and references to those inputs.

    ``ids_to_ops`` maps ``id(DataOp)`` to an already-converted ``Op``. Binding a
    DataOp records that op in ``inputs`` and returns ``OperandRef(k)``, where
    ``k`` is its input index. ``bind`` also handles DataOps nested in tuples,
    lists, dictionary values, and slice bounds.

    Inputs are ordered by their first binding. For ``df.join(other)``,
    :func:`as_op` binds the method receiver (``df``) before its arguments, so
    ``df`` gets ``OperandRef(0)`` and ``other`` gets ``OperandRef(1)``. Repeated
    references to the same upstream op reuse its index: for ``x + x``, both
    operands refer to the same input edge.

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
        if isinstance(value, BaseEstimator) and next(_collect_child_data_ops(value), None) is not None:
            raise NotImplementedError(
                f"DataOp parameters of {type(value).__name__} are only supported "
                "in the estimator of `.skb.apply()`.")
        if isinstance(value, tuple):
            return tuple(self.bind(v) for v in value)
        if isinstance(value, list):
            return [self.bind(v) for v in value]
        if isinstance(value, dict):
            return rebuild_dict(value, ((k, self.bind(v)) for k, v in value.items()))
        if isinstance(value, slice):
            return slice(self.bind(value.start), self.bind(value.stop), self.bind(value.step))
        return value

    def bind_seq(self, seq):
        """Bind DataOps in a tuple/list argument sequence to OperandRefs."""
        return tuple(self.bind(a) for a in seq)

    def bind_map(self, mapping):
        """Bind DataOps in a kwargs dict to OperandRefs."""
        return {k: self.bind(v) for k, v in mapping.items()}


def _bind_or_value(binder: OperandBinder, value):
    """Bind a field holding DataOps (-> OperandRefs), constants, or a mix of both.

    Delegates to :meth:`OperandBinder.bind`, which recurses the same containers
    ``_collect_child_data_ops`` does. Binding has to follow that walk exactly:
    graph extraction already reported every DataOp it finds -- including ones
    nested in a tuple, such as the row/column pair of ``df.loc[mask, cols]`` --
    as a child of this node, so a DataOp left unbound here stays in the DAG with
    no consumer. Topological traversal rejects that dangling output edge.
    """
    return binder.bind(value)


def _bind_operand(binder: OperandBinder, value) -> OperandRef:
    """Bind ``value`` as an input edge, giving a literal a ``ValueOp`` of its own."""
    bound = binder.bind(value)
    return bound if isinstance(bound, OperandRef) else binder.ref_op(ValueOp(bound))


# Method-argument groups supported during fitting or response evaluation.
# Applicable groups depend on the estimator kind (see BaseEstimatorOp subclasses).
_EXECUTABLE_KWARGS_KEYS = frozenset({"fit", "fit_transform", "transform"}) | RESPONSE_MODES
# Groups skrub routes to a method stratum never calls. Ignoring them would change
# results, so they are rejected. Searches use an explicit metric rather than
# calling the estimator's `score` method.
_UNSUPPORTED_KWARGS_KEYS = frozenset({"score"})


def _bind_apply_kwargs(binder: OperandBinder, estimator_class, estimator,
                       kwargs: dict | None) -> dict:
    """Bind the DataOps nested in an Apply's per-method kwargs to OperandRefs.

    ``.skb.apply()`` always passes all seven method groups, most of them None.
    Dropping the empty ones keeps the op's config (and therefore its CSE key)
    down to the groups that carry something.

    Bind all remaining groups, including unused groups such as a transformer's
    ``fit`` kwargs. Graph extraction includes their DataOps as dependencies;
    binding preserves edge consistency until stage-specific input pruning.
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
            # This estimator kind ignores the group in skrub too. Bind its graph
            # dependencies for consistency, but do not pass the group to a method.
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


def _has_unbound_parameter(value, named_estimators: set[int]) -> bool:
    """Find DataOps in containers (list, dict, etx), excluding estimators with named parameters.

    For example, Pipeline exposes its steps through ``get_params()`` as well as
    storing them in a list. Their parameters are bound by their flattened names.
    """
    if isinstance(value, BaseEstimator):
        return (id(value) not in named_estimators
                and next(_collect_child_data_ops(value), None) is not None)
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, (tuple, list, set, frozenset)):
        items = value
    elif isinstance(value, slice):
        items = (value.start, value.stop, value.step)
    else:
        return next(_collect_child_data_ops(value), None) is not None
    return any(_has_unbound_parameter(item, named_estimators) for item in items)


def _bind_estimator_params(binder: OperandBinder, estimator) -> dict:
    """Bind direct and sklearn named nested parameters to input references.

    Named nested parameters such as ``scale__with_mean`` are supported::

        allowed = X.skb.apply(Pipeline([
            ("scale", StandardScaler(with_mean=st.as_data_op(True))),
        ]))

    A DataOp inside a parameter container is rejected during conversion::

        rejected = X.skb.apply(FunctionTransformer(
            kw_args={"offset": st.as_data_op(1)},
        ))
    """
    params = estimator.get_params()
    named_estimators = {id(value) for value in params.values()
                        if isinstance(value, BaseEstimator)}
    refs = {}
    for name, value in params.items():
        if isinstance(value, DataOp):
            refs[name] = binder.ref(value)
        elif _has_unbound_parameter(value, named_estimators):
            # TODO: Support DataOps inside estimator parameter containers when needed.
            raise NotImplementedError(
                f"DataOps inside parameter containers are not supported "
                f"({type(estimator).__name__} parameter {name!r}). "
                "DataOp parameters must be exposed directly by get_params().")
    return refs


def _apply_estimator_op(impl: Apply, estimator, ids_to_ops: dict, feeds_estimator: bool = False,
                        choice_ops: dict | None = None) -> Op:
    """Build a TransformerOp or PredictorOp for one concrete Apply estimator.

    ``estimator`` is ``impl.estimator`` itself, or one outcome of it when the
    Apply's estimator is a ``Choice``. Each call uses its own binder so every op
    gets X as OperandRef(0) and its own de-duplicated inputs list.

    ``feeds_estimator`` says another Apply consumes this one's output. skrub then
    fit_transforms and transforms any estimator that can, a predictor included (a
    KMeans supplies distances to the next estimator), so such an estimator becomes a
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
    param_refs = _bind_estimator_params(binder, estimator)
    y = _bind_or_value(binder, impl.y)
    cols = _bind_or_value(binder, impl.cols)
    exclude_cols = _bind_or_value(binder, impl.exclude_cols)
    no_wrap = _bind_or_value(binder, impl.no_wrap)
    kwargs = _bind_apply_kwargs(binder, estimator_class, estimator, impl.kwargs)
    # Bind the original DataOp identities before cloning. Each converted op gets
    # its own estimator so rewrites can update its parameters independently.
    op = estimator_class(
        estimator=clone(estimator),
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
        # Every operand is an input edge; a literal one (an array) gets a ValueOp.
        operands = [_bind_operand(binder, o) for o in (impl.first, *impl.others)]
        axis = _bind_or_value(binder, impl.axis)
        return_op = ConcatOp(operands=operands, axis=axis)
        return_op.inputs = binder.inputs
    elif isinstance(impl, FreezeAfterFit):
        raise NotImplementedError("`.skb.freeze_after_fit()` is not supported by stratum.")
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
