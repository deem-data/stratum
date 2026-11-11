"""Score candidates from the responses already computed by their plans.

- A :class:`Metric` calls ``sign * fn(y_true, values, **kwargs)``.
- scikit-learn scorers additionally invoke an estimator response method and
  inspect estimator metadata. Stratum selects a response method during planning
  and passes its results directly to the metric through ``ScoreCandidatesOp``.
- :data:`_NATIVE` contains Stratum's metric implementations. Other names resolve
  through scikit-learn's registered scorers, extracting the function, sign,
  response methods, and keyword arguments.

Compatibility depends on both the metric function and response conversion;
native implementations need equivalence tests against scikit-learn.
"""
from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from sklearn.metrics import get_scorer

import logging
logger = logging.getLogger(__name__)

_PROBABILITY_RESPONSES = ("predict_proba", "predict_log_proba")


@dataclass(frozen=True)
class Metric:
    """A metric and everything needed to call it on a fold's values.

    ``response_method`` lists what the metric can read, best first, the way
    scikit-learn's scorers declare alternatives (``roc_auc`` takes a decision function
    or probabilities). The plan picks one of them at build time.
    """

    name: str
    fn: Callable[..., float]
    response_method: tuple[str, ...] = ("predict",)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: +1 when the metric is already a utility, -1 for a `neg_*` loss.
    sign: int = 1
    #: Where the implementation comes from: "stratum", "sklearn" or "callable".
    source: str = "sklearn"

    def __call__(self, y_true, values, **kwargs) -> float:
        # Per-fold kwargs (from `.skb.with_scoring(kwargs=...)`) extend the scorer's
        # own, as skrub merges them into a scorer's `_kwargs`.
        return self.sign * self.fn(y_true, values, **{**self.kwargs, **kwargs})

    def __repr__(self) -> str:
        return f"Metric({self.name!r}, via={self.source})"


def read_response(values, response_mode: str):
    """Convert a two-column probability response to its second column.

    This assumes the second column represents the metric's positive class.
    Other response shapes and non-probability responses are returned unchanged.
    """
    if response_mode not in _PROBABILITY_RESPONSES:
        return values
    # TODO: Select the probability column using the metric's pos_label and
    # estimator classes; column 1 is incorrect when the first class is positive.
    array = np.asarray(values)
    return array[:, 1] if array.ndim == 2 and array.shape[1] == 2 else values


# --- metrics Stratum computes itself -----------------------------------------------

def _labels(y) -> np.ndarray:
    """Labels as a flat array, whatever frame library produced them."""
    # TODO: Preserve target dimensions for multilabel and multioutput metrics;
    # flattening changes subset accuracy and per-output R² aggregation.
    return np.asarray(y).ravel()


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(_labels(y_true) == _labels(y_pred)))


def mean_squared_error(y_true, y_pred) -> float:
    return float(np.mean((_labels(y_true) - _labels(y_pred)) ** 2))


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_labels(y_true) - _labels(y_pred))))


def r2_score(y_true, y_pred) -> float:
    true, pred = _labels(y_true), _labels(y_pred)
    residual = np.sum((true - pred) ** 2)
    total = np.sum((true - np.mean(true)) ** 2)
    # For a constant target, return 1.0 for exact predictions and 0.0 otherwise.
    if total == 0:
        return 1.0 if residual == 0 else 0.0
    return float(1 - residual / total)


def _native(name, fn, sign=1) -> Metric:
    return Metric(name=name, fn=fn, sign=sign, source="stratum")


_NATIVE: dict[str, Metric] = {
    m.name: m for m in (
        _native("accuracy", accuracy),
        _native("r2", r2_score),
        _native("neg_mean_squared_error", mean_squared_error, sign=-1),
        _native("neg_root_mean_squared_error", root_mean_squared_error, sign=-1),
        _native("neg_mean_absolute_error", mean_absolute_error, sign=-1),
    )
}


# --- resolution ---------------------------------------------------------------------

def resolve_scoring(scoring) -> Metric:
    """Resolve ``scoring`` to the :class:`Metric` the plan will score with.

    A string names a metric, Stratum's own where there is one and scikit-learn's
    otherwise. A callable is a metric, ``f(y_true, y_pred)``, not a scikit-learn scorer:
    Stratum scores from the values a plan produced and has no estimator to hand one.
    ``make_scorer(...)`` is accepted and decomposed, which is how a caller asks for a
    metric that reads probabilities or carries keyword arguments.

    ``scoring`` is required: a search ranks a set of candidates, so a per-estimator
    default would compare two different metrics.
    """
    if scoring is None:
        raise ValueError(
            "scoring=None would score each candidate with its own estimator's `score`,"
            " so a batch mixing estimator kinds would rank an accuracy against an R²."
            " Name one metric: a string, a metric `f(y_true, y_pred)`, or a scorer built"
            " with `make_scorer`."
        )
    if isinstance(scoring, (list, tuple, set, dict)):
        raise ValueError(
            f"scoring={scoring!r} asks for several metrics, but a search reports a single"
            " score per candidate. Pass one."
        )
    if isinstance(scoring, str):
        metric = _NATIVE.get(scoring) or _from_sklearn_name(scoring)
    elif callable(scoring):
        metric = _decompose(scoring) if _is_sklearn_scorer(scoring) else _from_callable(scoring)
    else:
        raise ValueError(
            f"scoring={scoring!r} is not a metric: pass a string naming one, a callable"
            " `f(y_true, y_pred)`, or a scorer built with `make_scorer`."
        )
    logger.info(f"Using metric: {metric}")
    return metric


def resolve_declared_scoring(dag) -> Metric:
    """The :class:`Metric` declared on ``dag`` with ``.skb.with_scoring()``.

    Only the metric is resolved here; the scorer's ``kwargs`` may be computed by the
    plan (``sample_weight=X["w"]``), so they are bound as plan operands and evaluated
    on each test fold (see ``DeclaredScoringOp``). A declaration Stratum cannot score
    with raises ``ValueError``: several scorers (a search reports one score), a
    ``scoring`` that is itself computed by the plan, or ``None``, which means the
    estimator's own ``score`` (see ``resolve_scoring``).
    """
    from skrub._data_ops._evaluation import find_scoring_node, needs_eval

    node = find_scoring_node(dag)
    if node is None:
        raise ValueError("this DataOp declares no scorer with `.skb.with_scoring()`.")
    scorers = node._skrub_impl.scorers
    if len(scorers) != 1:
        raise ValueError(
            f"this DataOp declares {len(scorers)} scorers with `.skb.with_scoring()`, but"
            " a search reports a single score per candidate. Pass `scoring=` to pick one.")
    [info] = scorers
    scoring, kwargs = info["scoring"], info["kwargs"]
    if needs_eval(scoring):
        raise ValueError(
            "the scoring declared with `.skb.with_scoring()` is computed by the plan;"
            " Stratum resolves the metric when it builds the plan. Pass a value instead.")
    if isinstance(scoring, str) and scoring in _NATIVE and kwargs not in (None, {}):
        # Stratum's own metrics take no keyword arguments, scikit-learn's do.
        metric = _from_sklearn_name(scoring)
    else:
        metric = resolve_scoring(scoring)
    if info["name"] is not None:
        metric = dataclasses.replace(metric, name=info["name"])
    return metric


def _is_sklearn_scorer(scoring) -> bool:
    """Whether ``scoring`` is a scikit-learn scorer we can take apart."""
    return hasattr(scoring, "_score_func")


def _from_sklearn_name(name: str) -> Metric:
    """Decompose the scorer scikit-learn registers under ``name``."""
    try:
        scorer = get_scorer(name)
    except (ValueError, KeyError) as e:
        known = ", ".join(sorted(_NATIVE))
        raise ValueError(
            f"scoring={name!r} is neither a metric Stratum implements ({known}) nor a"
            f" scikit-learn scorer name. Pass one of those, a callable"
            f" `f(y_true, y_pred)`, or add it to `_NATIVE`."
        ) from e
    return _decompose(scorer, name=name)


def _decompose(scorer, name: str | None = None) -> Metric:
    """Reduce a scikit-learn scorer to the metric it wraps.

    Read the scorer's metric function, response methods, kwargs, and sign without
    invoking an estimator. These are scikit-learn internals and may change between
    versions. A missing metric function is rejected; the other fields have defaults.
    """
    fn = getattr(scorer, "_score_func", None)
    if fn is None:
        raise ValueError(
            f"scoring={name or scorer!r} resolves to {scorer!r}, which Stratum cannot"
            " reduce to a metric `f(y_true, values)`. Implement it in"
            " `stratum/optimizer/logical/_scoring.py::_NATIVE`, or pass a metric directly."
        )
    response = getattr(scorer, "_response_method", "predict") or "predict"
    if isinstance(response, str):
        response = (response,)
    return Metric(
        name=name or getattr(fn, "__name__", repr(scorer)),
        fn=fn,
        response_method=tuple(response),
        kwargs=dict(getattr(scorer, "_kwargs", {}) or {}),
        sign=int(getattr(scorer, "_sign", 1)),
    )


def _from_callable(scoring: Callable) -> Metric:
    """Accept a user metric, and refuse a scikit-learn style scorer with a reason."""
    try:
        params = list(inspect.signature(scoring).parameters.values())
    except (TypeError, ValueError):
        params = []
    required = [p for p in params
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(required) > 2 or (params and params[0].name in ("estimator", "est", "model")):
        raise ValueError(
            f"scoring={scoring.__name__!r} looks like a scikit-learn scorer"
            " `f(estimator, X, y)`. Stratum scores a candidate from the values its plan"
            " produced and has no estimator to pass one: give a metric"
            " `f(y_true, y_pred)` instead, or `make_scorer(...)` to choose the response"
            " it reads."
        )
    return Metric(name=getattr(scoring, "__name__", "callable"), fn=scoring,
                  source="callable")
