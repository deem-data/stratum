"""Physical predictor operators.

Lowering turns a logical :class:`~stratum.optimizer.logical._ops.PredictorOp`
wrapping a supported model into an *abstract* physical predictor op -- one per
model family (random forest, ridge, LightGBM, ...). Implementation selection then
swaps the abstract op to a concrete impl. Today every family has exactly one: the
reference impl, which runs the user's estimator unchanged through the inherited
``BaseEstimatorOp.process``.

The abstract ops exist so alternative implementations have something to register
against. A new impl subclasses the abstract op, registers under it, gates itself
with ``supports`` and swaps the estimator at plan time, the same way
:class:`~stratum.optimizer.physical._transform_execs.RustStringEncoder` does::

    @rust_impl(of=RandomForestOp)
    class RustRandomForest(RandomForestOp, RustPhysicalOp):
        is_abstract = False

        @classmethod
        def supports(cls, op, ctx):
            return isinstance(op.original_estimator, RandomForestClassifier)

        def on_impl_selected(self, ctx):
            self.estimator = ...
            self.original_estimator = ...

Every reference impl registers under the ``sklearn-skrub`` backend, including
those for LightGBM, XGBoost and CatBoost: in the registry that backend means
"run the estimator as given, through its scikit-learn API", which every selector
already treats as the backend-agnostic fallback.

Lowering is incremental. A predictor with no family below returns ``None`` from
:func:`lower_predictor`, passes through lowering unchanged, and keeps running via
the ``sklearn-skrub`` impl registered on ``PredictorOp`` itself.
"""
from __future__ import annotations

import sys

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge, RidgeClassifier,
                                  SGDClassifier, SGDRegressor)

from stratum.optimizer.logical._ops import PredictorOp
from stratum.optimizer.physical._lowering import lowering_rule
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import sklearn_skrub_impl


# --- Random forest -----------------------------------------------------------
class RandomForestOp(PredictorOp, PhysicalOp):
    """Abstract physical random forest (``RandomForestClassifier``/``Regressor``)."""
    is_abstract = True


@sklearn_skrub_impl(of=RandomForestOp)
class SklearnRandomForest(RandomForestOp):
    """Reference impl: runs the scikit-learn forest as-is."""
    is_abstract = False


# --- Linear models -----------------------------------------------------------
class LinearRegressionOp(PredictorOp, PhysicalOp):
    """Abstract physical ordinary least squares (``LinearRegression``)."""
    is_abstract = True


@sklearn_skrub_impl(of=LinearRegressionOp)
class SklearnLinearRegression(LinearRegressionOp):
    """Reference impl: runs the scikit-learn ``LinearRegression`` as-is."""
    is_abstract = False


class RidgeOp(PredictorOp, PhysicalOp):
    """Abstract physical ridge model (``Ridge``/``RidgeClassifier``)."""
    is_abstract = True


@sklearn_skrub_impl(of=RidgeOp)
class SklearnRidge(RidgeOp):
    """Reference impl: runs the scikit-learn ridge model as-is."""
    is_abstract = False


class LassoOp(PredictorOp, PhysicalOp):
    """Abstract physical lasso (``Lasso``)."""
    is_abstract = True


@sklearn_skrub_impl(of=LassoOp)
class SklearnLasso(LassoOp):
    """Reference impl: runs the scikit-learn ``Lasso`` as-is."""
    is_abstract = False


class ElasticNetOp(PredictorOp, PhysicalOp):
    """Abstract physical elastic net (``ElasticNet``)."""
    is_abstract = True


@sklearn_skrub_impl(of=ElasticNetOp)
class SklearnElasticNet(ElasticNetOp):
    """Reference impl: runs the scikit-learn ``ElasticNet`` as-is."""
    is_abstract = False


class LogisticRegressionOp(PredictorOp, PhysicalOp):
    """Abstract physical logistic regression (``LogisticRegression``)."""
    is_abstract = True


@sklearn_skrub_impl(of=LogisticRegressionOp)
class SklearnLogisticRegression(LogisticRegressionOp):
    """Reference impl: runs the scikit-learn ``LogisticRegression`` as-is."""
    is_abstract = False


class SGDOp(PredictorOp, PhysicalOp):
    """Abstract physical SGD-trained linear model (``SGDClassifier``/``Regressor``)."""
    is_abstract = True


@sklearn_skrub_impl(of=SGDOp)
class SklearnSGD(SGDOp):
    """Reference impl: runs the scikit-learn SGD model as-is."""
    is_abstract = False


# --- Gradient boosting libraries ----------------------------------------------
class LightGBMOp(PredictorOp, PhysicalOp):
    """Abstract physical LightGBM model (any ``lightgbm.LGBMModel``)."""
    is_abstract = True


@sklearn_skrub_impl(of=LightGBMOp)
class LibLightGBM(LightGBMOp):
    """Reference impl: runs the LightGBM estimator as-is."""
    is_abstract = False


class XGBoostOp(PredictorOp, PhysicalOp):
    """Abstract physical XGBoost model (any ``xgboost.XGBModel``)."""
    is_abstract = True


@sklearn_skrub_impl(of=XGBoostOp)
class LibXGBoost(XGBoostOp):
    """Reference impl: runs the XGBoost estimator as-is."""
    is_abstract = False


class CatBoostOp(PredictorOp, PhysicalOp):
    """Abstract physical CatBoost model (any ``catboost.CatBoost``)."""
    is_abstract = True


@sklearn_skrub_impl(of=CatBoostOp)
class LibCatBoost(CatBoostOp):
    """Reference impl: runs the CatBoost estimator as-is."""
    is_abstract = False


# Matched on the exact type: scikit-learn subclasses these for different models
# (``Lasso`` is an ``ElasticNet``, ``MultiTaskLasso`` a ``Lasso``,
# ``LogisticRegressionCV`` a ``LogisticRegression``), which must stay unlowered.
_SKLEARN_FAMILIES: dict[type, type[PredictorOp]] = {
    RandomForestClassifier: RandomForestOp,
    RandomForestRegressor: RandomForestOp,
    LinearRegression: LinearRegressionOp,
    Ridge: RidgeOp,
    RidgeClassifier: RidgeOp,
    Lasso: LassoOp,
    ElasticNet: ElasticNetOp,
    LogisticRegression: LogisticRegressionOp,
    SGDClassifier: SGDOp,
    SGDRegressor: SGDOp,
}

# (module, base class, family). The libraries are optional, so their base classes
# are looked up in ``sys.modules`` instead of imported: an estimator can only be an
# instance of a library the user has already imported.
_LIBRARY_FAMILIES: tuple[tuple[str, str, type[PredictorOp]], ...] = (
    ("lightgbm", "LGBMModel", LightGBMOp),
    ("xgboost", "XGBModel", XGBoostOp),
    ("catboost", "CatBoost", CatBoostOp),
)


def predictor_family(estimator) -> type[PredictorOp] | None:
    """The abstract physical op for ``estimator``, or ``None`` if it has none."""
    family = _SKLEARN_FAMILIES.get(type(estimator))
    if family is not None:
        return family
    for module_name, base_name, family in _LIBRARY_FAMILIES:
        module = sys.modules.get(module_name)
        base = getattr(module, base_name, None) if module is not None else None
        if base is not None and isinstance(estimator, base):
            return family
    return None


@lowering_rule(PredictorOp)
def lower_predictor(op: PredictorOp, ctx) -> PhysicalOp | None:
    """Lower a ``PredictorOp`` to the matching abstract physical predictor.

    Only models with a family above are lowered; anything else returns ``None``
    and stays a logical ``PredictorOp``.
    """
    family = predictor_family(op.original_estimator)
    if family is None:
        return None
    return family(
        estimator=op.estimator, y=op.y, cols=op.cols, how=op.how,
        allow_reject=op.allow_reject, unsupervised=op.unsupervised,
        kwargs=op.kwargs, param_refs=op.param_refs,
    )
