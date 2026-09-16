"""Scoring a candidate from the values its plan produced.

A metric is a pure function of the labels and what a model produced for them:
``sign * fn(y_true, values, **kwargs)``. scikit-learn wraps one in a *scorer*, a
``scorer(estimator, X, y)`` callable that additionally decides which response method to
call and how to read the result. That wrapper needs a fitted estimator object to
interrogate: it asks what the estimator *is* (classifier or regressor, what its classes
are) before it will use anything the estimator produced. A candidate is a sub-DAG of a
merged plan, not an estimator, so honouring that signature means fabricating an object
for scikit-learn to interrogate.

Stratum takes scikit-learn's scoring API as a *vocabulary* instead. ``scoring="roc_auc"``
resolves to a :class:`Metric` here, the plan's response mode already produces the values
that metric reads (see ``ScoreCandidatesOp``), and scoring is one call against them.
Nothing stands in for an estimator, because nothing has to.

:data:`_NATIVE` holds the metrics Stratum computes itself. Every other name is resolved
by decomposing scikit-learn's scorer into the same four fields, so a metric Stratum has
not implemented still works and still means exactly what it means upstream. Moving a
metric from the second group to the first is a local change with a test that pins the
two against each other.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from sklearn.metrics import get_scorer

import logging
logger = logging.getLogger(__name__)

#: Responses a metric can read. The plan produces exactly one of them per pass.
RESPONSE_METHODS = ("predict", "predict_proba", "predict_log_proba", "decision_function")

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

    def __call__(self, y_true, values) -> float:
        return self.sign * self.fn(y_true, values, **self.kwargs)

    def __repr__(self) -> str:
        return f"Metric({self.name!r}, via={self.source})"


def read_response(values, response_mode: str):
    """The values as a metric reads them for ``response_mode``.

    A two-column probability matrix is one score to every binary metric. scikit-learn's
    scorer narrows it the same way, to the column of the positive class, which is the
    last one because an estimator's ``classes_`` is sorted.
    """
    if response_mode not in _PROBABILITY_RESPONSES:
        return values
    array = np.asarray(values)
    return array[:, 1] if array.ndim == 2 and array.shape[1] == 2 else values


# --- metrics Stratum computes itself -----------------------------------------------

def _labels(y) -> np.ndarray:
    """Labels as a flat array, whatever frame library produced them."""
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
    # scikit-learn's convention for a constant target: 0.0 when the fit is exact.
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

    Stratum reads the four fields rather than calling the scorer, because calling it
    would require an estimator object to interrogate. The fields are scikit-learn
    internals, so a version bump can invalidate this; a scorer that does not carry them
    is refused rather than guessed at.
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
