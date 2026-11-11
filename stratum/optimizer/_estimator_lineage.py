"""Estimator lineage keys derived from classes and constructor parameters.

An estimator op is keyed like any other cacheable op (see
:mod:`stratum.optimizer._lineage`), with the estimator itself named by its class
and its constructor parameters. Reuse assumes fitting is deterministic for those
parameters and input data. Eligibility checks reject unsupported packages,
frozen estimators, and unset seeds except for known deterministic configurations
or packages with fixed default seeds. Rejected estimators raise :class:`Uncacheable`.

Execution parameters such as ``n_jobs`` and ``verbose`` are excluded so changes
to these settings do not invalidate cache entries.
"""
from __future__ import annotations

import fnmatch
from types import FunctionType

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.frozen import FrozenEstimator
from skrub._data_ops._data_ops import DataOp
from skrub.selectors import _base as _selectors

from stratum.optimizer.logical._base import OperandRef, Uncacheable

#: Top-level packages eligible for estimator lineage. User-defined estimators
#: are excluded because their fitting behavior is not known.
KNOWN_ESTIMATOR_PACKAGES = frozenset({"sklearn", "skrub", "stratum", "lightgbm", "xgboost",
                                      "catboost"})

#: Parameters treated as execution settings and excluded from estimator lineage.
RUNTIME_PARAMETERS = frozenset({"n_jobs", "verbose", "verbosity", "silent", "thread_count",
                                "n_threads", "nthread", "copy", "copy_X"})

#: Packages whose estimators fall back to a fixed default seed when given none.
_DEFAULT_SEEDED_PACKAGES = frozenset({"lightgbm", "xgboost", "catboost"})

#: Parameters naming an estimator's seed.
_SEED_PARAMETERS = ("random_state", "random_seed", "seed")

#: Estimators with a seed parameter that only matters in some configurations: the
#: predicate identifies configurations that are deterministic without a seed.
_SEEDLESS_WHEN = {
    "Ridge": lambda p: p.get("solver") not in ("sag", "saga"),
    "RidgeClassifier": lambda p: p.get("solver") not in ("sag", "saga"),
    "LogisticRegression": lambda p: p.get("solver") not in ("liblinear", "sag", "saga"),
    "Lasso": lambda p: p.get("selection") == "cyclic",
    "ElasticNet": lambda p: p.get("selection") == "cyclic",
}


def estimator_lineage(estimator: BaseEstimator, *, graph_inputs: tuple = ()):
    """Return the estimator class and parameters used to compute its lineage key.

    ``graph_inputs`` are already represented by operator operands.
    An unbound DataOp parameter has no input key and must be rejected.
    """
    cls = type(estimator)
    package = cls.__module__.partition(".")[0]
    if package not in KNOWN_ESTIMATOR_PACKAGES:
        raise Uncacheable(f"{cls.__qualname__} is not an estimator stratum knows")
    if isinstance(estimator, FrozenEstimator):
        raise Uncacheable("a frozen estimator depends on fitted state")
    params = estimator.get_params(deep=False)
    _check_seeded(cls, package, params)
    return ("estimator", f"{cls.__module__}.{cls.__qualname__}",
            {name: _parameter(params[name], graph_inputs) for name in sorted(params)
             if name not in RUNTIME_PARAMETERS})


def _check_seeded(cls: type, package: str, params: dict) -> None:
    seed = next((name for name in _SEED_PARAMETERS if name in params), None)
    if seed is not None and isinstance(params[seed], DataOp):
        raise Uncacheable("a graph-fed random seed cannot be checked at plan time")
    if seed is None or params[seed] is not None or package in _DEFAULT_SEEDED_PACKAGES:
        return
    for klass in cls.__mro__:
        when = _SEEDLESS_WHEN.get(klass.__name__)
        if when is not None and klass.__module__.startswith("sklearn."):
            if when(params):
                return
            break
    raise Uncacheable(f"{cls.__qualname__} has {seed}=None")


def _parameter(value, graph_inputs):
    """A parameter value as plain values: nested estimators by their own lineage."""
    if isinstance(value, BaseEstimator):
        return estimator_lineage(value, graph_inputs=graph_inputs)
    if isinstance(value, _selectors.Selector):
        return selector_lineage(value)
    if isinstance(value, DataOp):
        # Graph-fed: its value is bound through the op's `param_refs` and inputs.
        if not any(value is operand for operand in graph_inputs):
            raise Uncacheable("an estimator parameter has no graph input binding")
        return ("graph-fed",)
    if isinstance(value, type) and issubclass(value, np.generic):
        return np.dtype(value)  # `dtype=np.float32`
    if type(value) in (list, tuple):
        return type(value)(_parameter(v, graph_inputs) for v in value)
    if type(value) is dict:
        return {k: _parameter(v, graph_inputs) for k, v in value.items()}
    if isinstance(value, (np.random.RandomState, np.random.Generator)):
        raise Uncacheable("a random state object is consumed by every fit")
    return value


def selector_lineage(selector):
    """A skrub column selector (``cols=`` of ``.skb.apply()``) as plain values.

    Column names can be encoded directly. Selectors are encoded by structure and
    filters by predicate. The predicate must be skrub's or its glob matcher, since an
    arbitrary predicate may depend on mutable state or randomness.
    """
    if selector is None or isinstance(selector, (str, int, OperandRef)):
        return selector
    if type(selector) in (list, tuple):
        return type(selector)(selector_lineage(s) for s in selector)
    if type(selector) is _selectors.All:
        return ("selector", "all")
    if type(selector) is _selectors.Cols:
        return ("selector", "cols", tuple(selector.columns))
    if type(selector) is _selectors.Inv:
        return ("selector", "inv", selector_lineage(selector.complement))
    if type(selector) in (_selectors.Or, _selectors.And, _selectors.Sub, _selectors.XOr):
        return ("selector", type(selector).__name__, selector_lineage(selector.left),
                selector_lineage(selector.right))
    if type(selector) in (_selectors.Filter, _selectors.NameFilter):
        predicate = selector.predicate
        module = getattr(predicate, "__module__", "") or ""
        qualname = getattr(predicate, "__qualname__", "")
        library = (isinstance(predicate, FunctionType) and module.startswith("skrub.")) or (
            predicate is fnmatch.fnmatchcase)
        if not library or "<" in qualname:
            raise Uncacheable("a column selector filters with a user predicate")
        return ("selector", type(selector).__name__, f"{module}.{qualname}",
                selector.args, selector.kwargs)
    raise Uncacheable(f"no lineage encoding for the selector {type(selector).__qualname__}")
