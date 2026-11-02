"""Lineage of estimators: what a fit computes, named by class and parameters.

An estimator op is keyed like any other cacheable op (see
:mod:`stratum.optimizer._lineage`), with the estimator itself named by its class
and its constructor parameters. That is sound only for estimators whose fit is a
deterministic function of their parameters and data, so only estimators from the
libraries stratum knows are keyed, and an estimator that draws random numbers must
be seeded. Anything else raises :class:`Uncacheable`.

Parameters that only say *how* to run (``n_jobs``, ``verbose``) are left out, so a
fit on more cores still finds the entry a fit on fewer wrote.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.base import BaseEstimator
from skrub._data_ops._data_ops import DataOp
from skrub.selectors import _base as _selectors

from stratum.optimizer.logical._base import Uncacheable

#: Top-level packages whose estimators are keyed. A user-defined estimator may do
#: anything in ``fit``, so it is not.
KNOWN_ESTIMATOR_PACKAGES = frozenset({"sklearn", "skrub", "stratum", "lightgbm", "xgboost",
                                      "catboost"})

#: Parameters that change how an estimator runs, never what it computes.
RUNTIME_PARAMETERS = frozenset({"n_jobs", "verbose", "verbosity", "silent", "thread_count",
                                "n_threads", "nthread", "copy", "copy_X"})

#: Packages whose estimators fall back to a fixed default seed when given none.
_DEFAULT_SEEDED_PACKAGES = frozenset({"lightgbm", "xgboost", "catboost"})

#: Parameters naming an estimator's seed.
_SEED_PARAMETERS = ("random_state", "random_seed", "seed")

#: Estimators with a seed parameter that only matters in some configurations: the
#: predicate says when an unseeded one is deterministic all the same.
_SEEDLESS_WHEN = {
    "Ridge": lambda p: p.get("solver") not in ("sag", "saga"),
    "RidgeClassifier": lambda p: p.get("solver") not in ("sag", "saga"),
    "LogisticRegression": lambda p: p.get("solver") not in ("liblinear", "sag", "saga"),
    "Lasso": lambda p: p.get("selection") == "cyclic",
    "ElasticNet": lambda p: p.get("selection") == "cyclic",
}


def estimator_lineage(estimator: BaseEstimator):
    """Plain values naming ``estimator``'s fit. Raises :class:`Uncacheable`."""
    cls = type(estimator)
    package = cls.__module__.partition(".")[0]
    if package not in KNOWN_ESTIMATOR_PACKAGES:
        raise Uncacheable(f"{cls.__qualname__} is not an estimator stratum knows")
    params = estimator.get_params(deep=False)
    _check_seeded(cls, package, params)
    return ("estimator", f"{cls.__module__}.{cls.__qualname__}",
            {name: _parameter(params[name]) for name in sorted(params)
             if name not in RUNTIME_PARAMETERS})


def _check_seeded(cls: type, package: str, params: dict) -> None:
    seed = next((name for name in _SEED_PARAMETERS if name in params), None)
    if seed is None or params[seed] is not None or package in _DEFAULT_SEEDED_PACKAGES:
        return
    for klass in cls.__mro__:
        when = _SEEDLESS_WHEN.get(klass.__name__)
        if when is not None and klass.__module__.startswith("sklearn."):
            if when(params):
                return
            break
    raise Uncacheable(f"{cls.__qualname__} has {seed}=None")


def _parameter(value):
    """A parameter value as plain values: nested estimators by their own lineage."""
    if isinstance(value, BaseEstimator):
        return estimator_lineage(value)
    if isinstance(value, DataOp):
        # Graph-fed: its value is bound through the op's `param_refs` and inputs.
        return ("graph-fed",)
    if isinstance(value, type) and issubclass(value, np.generic):
        return np.dtype(value)  # `dtype=np.float32`
    if isinstance(value, (list, tuple)):
        return type(value)(_parameter(v) for v in value)
    if isinstance(value, dict):
        return {k: _parameter(v) for k, v in value.items()}
    if isinstance(value, (np.random.RandomState, np.random.Generator)):
        raise Uncacheable("a random state object is consumed by every fit")
    return value


def selector_lineage(selector):
    """A skrub column selector (``cols=`` of ``.skb.apply()``) as plain values.

    Column names are plain already. A selector is named by its structure; a filter
    by its predicate, which must be skrub's or the standard library's (``numeric()``,
    ``glob()``), since a user's predicate may compute anything.
    """
    if selector is None or isinstance(selector, (str, int)):
        return selector
    if isinstance(selector, (list, tuple)):
        return type(selector)(selector_lineage(s) for s in selector)
    if isinstance(selector, _selectors.All):
        return ("selector", "all")
    if isinstance(selector, _selectors.Cols):
        return ("selector", "cols", tuple(selector.columns))
    if isinstance(selector, _selectors.Inv):
        return ("selector", "inv", selector_lineage(selector.complement))
    if isinstance(selector, (_selectors.Or, _selectors.And, _selectors.Sub, _selectors.XOr)):
        return ("selector", type(selector).__name__, selector_lineage(selector.left),
                selector_lineage(selector.right))
    if isinstance(selector, (_selectors.Filter, _selectors.NameFilter)):
        predicate = selector.predicate
        module = getattr(predicate, "__module__", "") or ""
        qualname = getattr(predicate, "__qualname__", "")
        library = module.startswith("skrub.") or module in sys.stdlib_module_names
        if not library or "<" in qualname:
            raise Uncacheable("a column selector filters with a user predicate")
        return ("selector", type(selector).__name__, f"{module}.{qualname}",
                selector.args, selector.kwargs)
    raise Uncacheable(f"no lineage encoding for the selector {type(selector).__qualname__}")
