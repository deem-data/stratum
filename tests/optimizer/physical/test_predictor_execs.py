"""Physical predictor ops: lowering, reference-impl selection, and equivalence.

Each supported model family lowers to its own abstract physical op, and the
reference impl bound at plan time runs the user's estimator unchanged, so a plan
predicts exactly what skrub predicts.
"""
import importlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.linear_model import (LassoCV, LogisticRegressionCV, MultiTaskElasticNet,
                                  MultiTaskLasso, Ridge, RidgeCV)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum._api import evaluate
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._base import All
from stratum.optimizer.logical._ops import OperandRef, PredictorOp
from stratum.optimizer.physical._impl_selection import (
    DefaultImplementationSelector,
    FlagBasedSelector,
    GreedyImplementationSelector,
    select_implementations,
)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._predictor_execs import (
    CatBoostOp,
    ElasticNetOp,
    LassoOp,
    LibCatBoost,
    LibLightGBM,
    LibXGBoost,
    LightGBMOp,
    LinearRegressionOp,
    LogisticRegressionOp,
    RandomForestOp,
    RidgeOp,
    SGDOp,
    SklearnElasticNet,
    SklearnLasso,
    SklearnLinearRegression,
    SklearnLogisticRegression,
    SklearnRandomForest,
    SklearnRidge,
    SklearnSGD,
    XGBoostOp,
    lower_predictor,
)
from stratum.optimizer.physical._registry import (PhysicalImpl,
                                                  _current_process_execute,
                                                  _placeholder_cost,
                                                  _placeholder_exec_mem,
                                                  build_default_physical_registry)


def _ctx():
    return PlanContext(backend="pandas", pandas_query=False, rechunk=True,
                       parallelism=1, rust_backend=False, allow_patch=True)


def _estimator(module, name, **params):
    """Instantiate ``module.name``, skipping when an optional library is missing."""
    if module.split(".")[0] in ("lightgbm", "xgboost", "catboost"):
        mod = pytest.importorskip(module)
    else:
        mod = importlib.import_module(module)
    return getattr(mod, name)(**params)


# (module, class, abstract op, reference impl)
FAMILIES = [
    ("sklearn.ensemble", "RandomForestClassifier", RandomForestOp, SklearnRandomForest),
    ("sklearn.ensemble", "RandomForestRegressor", RandomForestOp, SklearnRandomForest),
    ("sklearn.linear_model", "LinearRegression", LinearRegressionOp, SklearnLinearRegression),
    ("sklearn.linear_model", "Ridge", RidgeOp, SklearnRidge),
    ("sklearn.linear_model", "RidgeClassifier", RidgeOp, SklearnRidge),
    ("sklearn.linear_model", "Lasso", LassoOp, SklearnLasso),
    ("sklearn.linear_model", "ElasticNet", ElasticNetOp, SklearnElasticNet),
    ("sklearn.linear_model", "LogisticRegression", LogisticRegressionOp,
     SklearnLogisticRegression),
    ("sklearn.linear_model", "SGDClassifier", SGDOp, SklearnSGD),
    ("sklearn.linear_model", "SGDRegressor", SGDOp, SklearnSGD),
    ("lightgbm", "LGBMClassifier", LightGBMOp, LibLightGBM),
    ("lightgbm", "LGBMRegressor", LightGBMOp, LibLightGBM),
    ("lightgbm", "LGBMRanker", LightGBMOp, LibLightGBM),
    ("xgboost", "XGBClassifier", XGBoostOp, LibXGBoost),
    ("xgboost", "XGBRegressor", XGBoostOp, LibXGBoost),
    # XGBoost's own random forest is still an XGBoost model, not a RandomForestOp.
    ("xgboost", "XGBRFRegressor", XGBoostOp, LibXGBoost),
    ("catboost", "CatBoostClassifier", CatBoostOp, LibCatBoost),
    ("catboost", "CatBoostRegressor", CatBoostOp, LibCatBoost),
]
FAMILY_IDS = [name for _, name, _, _ in FAMILIES]


@pytest.mark.parametrize("module, name, family, _impl", FAMILIES, ids=FAMILY_IDS)
def test_predictor_lowers_to_its_family(module, name, family, _impl):
    estimator = _estimator(module, name)
    lowered = lower_predictor(PredictorOp(estimator=estimator), _ctx())

    assert type(lowered) is family
    assert lowered.is_abstract
    # Still a PredictorOp, so response-mode and fit-pass planning see it as one.
    assert isinstance(lowered, PredictorOp)
    assert lowered.estimator is estimator


@pytest.mark.parametrize("estimator", [
    # sklearn subclasses of lowered models that are different models.
    LogisticRegressionCV(),
    MultiTaskLasso(),
    MultiTaskElasticNet(),
    # Related models without a physical op yet.
    RidgeCV(),
    LassoCV(),
    ExtraTreesClassifier(),
    HistGradientBoostingRegressor(),
    make_pipeline(StandardScaler(), Ridge()),
], ids=lambda e: type(e).__name__)
def test_unsupported_predictor_stays_logical(estimator):
    assert lower_predictor(PredictorOp(estimator=estimator), _ctx()) is None


@pytest.mark.parametrize("selector", [
    DefaultImplementationSelector(),
    GreedyImplementationSelector(),
    FlagBasedSelector(),
], ids=lambda s: type(s).__name__)
@pytest.mark.parametrize("module, name, family, impl", FAMILIES, ids=FAMILY_IDS)
def test_every_selector_binds_the_reference_impl(module, name, family, impl, selector):
    estimator = _estimator(module, name)
    op = lower_predictor(PredictorOp(estimator=estimator), _ctx())
    select_implementations(op, _ctx(), selector=selector)

    assert type(op) is impl
    assert isinstance(op, family) and not op.is_abstract
    # The reference impl runs the user's estimator as given.
    assert type(op.estimator) is type(estimator)
    assert type(op.original_estimator) is type(estimator)


def test_alternative_impl_registers_against_the_family_op():
    """An alternative random forest registers under RandomForestOp and gates
    itself with ``supports``; unsupported configs keep the reference impl."""

    class ClassifierOnlyForest(RandomForestOp):
        is_abstract = False

        @classmethod
        def supports(cls, op, ctx):
            return isinstance(op.original_estimator, RandomForestClassifier)

    registry = build_default_physical_registry()
    registry.register(PhysicalImpl(
        op_type=RandomForestOp, backend_name="stratum",
        input_format="frame", output_format="frame",
        supports=ClassifierOnlyForest.supports, cost=_placeholder_cost,
        exec_mem=_placeholder_exec_mem, execute=_current_process_execute,
        impl_class=ClassifierOnlyForest,
    ))

    def bound(estimator, selector):
        op = lower_predictor(PredictorOp(estimator=estimator), _ctx())
        select_implementations(op, _ctx(), registry=registry, selector=selector)
        return type(op)

    greedy = GreedyImplementationSelector()
    assert bound(_estimator("sklearn.ensemble", "RandomForestClassifier"),
                 greedy) is ClassifierOnlyForest
    assert bound(_estimator("sklearn.ensemble", "RandomForestRegressor"),
                 greedy) is SklearnRandomForest
    # The default policy keeps the sklearn reference even when an alternative fits.
    assert bound(_estimator("sklearn.ensemble", "RandomForestClassifier"),
                 DefaultImplementationSelector()) is SklearnRandomForest


def test_lowering_does_not_import_optional_libraries():
    code = ("import sys, stratum.optimizer._optimize; "
            "print(sorted(m for m in ('lightgbm', 'xgboost', 'catboost') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout
    assert out.strip() == "[]"


def test_graph_fed_catboost_parameter_is_only_applied_while_fitting():
    """A response pass must not mutate the fitted CatBoost estimator."""
    X, y = make_regression(n_samples=20, n_features=2, random_state=0)
    op = LibCatBoost(
        estimator=_estimator("catboost", "CatBoostRegressor", iterations=1, verbose=0),
        y=y,
        cols=All(),
        how="no_wrap",
        param_refs={"iterations": OperandRef(1)},
    )

    op.process("fit_transform", [X, 2])

    assert op.estimator.get_params()["iterations"] == 2
    assert len(op.process("predict", [X, 2])) == len(X)


# --- End-to-end: the plan predicts what skrub predicts ------------------------

def _frame(task):
    if task == "regression":
        X, y = make_regression(n_samples=200, n_features=6, noise=0.5,
                               random_state=0)
    else:
        X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["y"] = y
    return df


# (task, module, class, params, reference impl)
END_TO_END = [
    ("regression", "sklearn.ensemble", "RandomForestRegressor",
     {"n_estimators": 5, "random_state": 0}, SklearnRandomForest),
    ("classification", "sklearn.ensemble", "RandomForestClassifier",
     {"n_estimators": 5, "random_state": 0}, SklearnRandomForest),
    ("regression", "sklearn.linear_model", "LinearRegression", {},
     SklearnLinearRegression),
    ("regression", "sklearn.linear_model", "Ridge", {}, SklearnRidge),
    ("classification", "sklearn.linear_model", "RidgeClassifier", {}, SklearnRidge),
    ("regression", "sklearn.linear_model", "Lasso", {"alpha": 0.1}, SklearnLasso),
    ("regression", "sklearn.linear_model", "ElasticNet", {"alpha": 0.1},
     SklearnElasticNet),
    ("classification", "sklearn.linear_model", "LogisticRegression", {},
     SklearnLogisticRegression),
    ("classification", "sklearn.linear_model", "SGDClassifier", {"random_state": 0},
     SklearnSGD),
    ("regression", "sklearn.linear_model", "SGDRegressor", {"random_state": 0},
     SklearnSGD),
    ("regression", "lightgbm", "LGBMRegressor",
     {"n_estimators": 5, "random_state": 0, "verbose": -1}, LibLightGBM),
    ("classification", "lightgbm", "LGBMClassifier",
     {"n_estimators": 5, "random_state": 0, "verbose": -1}, LibLightGBM),
    ("regression", "xgboost", "XGBRegressor",
     {"n_estimators": 5, "random_state": 0}, LibXGBoost),
    ("classification", "xgboost", "XGBClassifier",
     {"n_estimators": 5, "random_state": 0}, LibXGBoost),
    ("regression", "catboost", "CatBoostRegressor",
     {"iterations": 5, "random_seed": 0, "verbose": 0}, LibCatBoost),
    ("classification", "catboost", "CatBoostClassifier",
     {"iterations": 5, "random_seed": 0, "verbose": 0}, LibCatBoost),
]


@pytest.mark.parametrize("task, module, name, params, impl", END_TO_END,
                         ids=[name for _, _, name, _, _ in END_TO_END])
def test_plan_predicts_like_skrub(task, module, name, params, impl):
    model = _estimator(module, name, **params)
    data = st.as_data_op(_frame(task))
    X = data.drop("y", axis=1).skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    pred = X.skb.apply(model, y=y)

    ops, *_ = optimize(pred)
    predictors = [op for op in ops if isinstance(op, PredictorOp)]
    assert len(predictors) == 1
    assert type(predictors[0]) is impl
    assert isinstance(predictors[0], PhysicalOp)

    seed, test_size = 42, 0.2
    preds = evaluate(pred, seed=seed, test_size=test_size)
    splits = pred.skb.train_test_split(random_state=seed, test_size=test_size)
    learner = pred.skb.make_learner()
    learner.fit(splits["train"])
    np.testing.assert_array_equal(learner.predict(splits["test"]), preds)
