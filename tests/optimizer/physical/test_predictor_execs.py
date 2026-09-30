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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import (LassoCV, LogisticRegressionCV, MultiTaskElasticNet,
                                  MultiTaskLasso, Ridge, RidgeCV)
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import ExtraTreeClassifier

import stratum as st
from stratum.adapters.tree_classifier import RustRandomForestClassifier
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
    DecisionTreeOp,
    ElasticNetOp,
    ExtraTreesOp,
    HistGradientBoostingOp,
    KNeighborsOp,
    LassoOp,
    LibCatBoost,
    LibLightGBM,
    LibXGBoost,
    LightGBMOp,
    LinearRegressionOp,
    LogisticRegressionOp,
    PassthroughPredictor,
    RandomForestOp,
    RidgeOp,
    SGDOp,
    SklearnDecisionTree,
    SklearnElasticNet,
    SklearnExtraTrees,
    SklearnHistGradientBoosting,
    SklearnKNeighbors,
    SklearnLasso,
    SklearnLinearRegression,
    SklearnLogisticRegression,
    SklearnRandomForest,
    RustExactRandomForestClassifier,
    RustHistogramRandomForestClassifier,
    SklearnRidge,
    SklearnSGD,
    XGBoostOp,
    lower_predictor,
)
from stratum.optimizer.physical._registry import (PhysicalImpl, PhysicalRegistry,
                                                  _current_process_execute,
                                                  _placeholder_cost,
                                                  _placeholder_exec_mem,
                                                  build_default_physical_registry)


def _ctx():
    return PlanContext(pandas_query=False, rechunk=True,
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
    ("sklearn.ensemble", "ExtraTreesClassifier", ExtraTreesOp, SklearnExtraTrees),
    ("sklearn.ensemble", "ExtraTreesRegressor", ExtraTreesOp, SklearnExtraTrees),
    ("sklearn.tree", "DecisionTreeClassifier", DecisionTreeOp, SklearnDecisionTree),
    ("sklearn.tree", "DecisionTreeRegressor", DecisionTreeOp, SklearnDecisionTree),
    ("sklearn.ensemble", "HistGradientBoostingClassifier", HistGradientBoostingOp,
     SklearnHistGradientBoosting),
    ("sklearn.ensemble", "HistGradientBoostingRegressor", HistGradientBoostingOp,
     SklearnHistGradientBoosting),
    ("sklearn.neighbors", "KNeighborsClassifier", KNeighborsOp, SklearnKNeighbors),
    ("sklearn.neighbors", "KNeighborsRegressor", KNeighborsOp, SklearnKNeighbors),
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
    ExtraTreeClassifier(),
    # Related models without a physical op yet.
    RidgeCV(),
    LassoCV(),
    RadiusNeighborsClassifier(),
    GradientBoostingRegressor(),
    make_pipeline(StandardScaler(), Ridge()),
], ids=lambda e: type(e).__name__)
def test_unsupported_predictor_stays_logical(estimator):
    assert lower_predictor(PredictorOp(estimator=estimator), _ctx()) is None


def test_pass_through_predictor_renders_its_estimator():
    """The logical op shows its family; the bound pass-through impl names the
    estimator, so per-op stats tell pass-through predictors apart."""
    op = PredictorOp(estimator=RidgeCV(alphas=[0.1, 1.0]))
    assert str(op) == "Predictor"

    select_implementations(op, _ctx())

    assert type(op) is PassthroughPredictor
    assert str(op) == "PredictorOp(RidgeCV(alphas=[0.1, 1.0]))"


class _UserDefinedPredictor(RegressorMixin, BaseEstimator):
    """A user's own estimator, written against the scikit-learn API."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_any_estimator_passes_through():
    X, y = make_regression(n_samples=20, n_features=2, random_state=0)
    op = PredictorOp(estimator=_UserDefinedPredictor(), y=y, cols=All(), how="no_wrap")
    select_implementations(op, _ctx())

    assert type(op) is PassthroughPredictor
    assert str(op) == "PredictorOp(_UserDefinedPredictor())"
    np.testing.assert_array_equal(op.process("fit_transform", [X]),
                                  np.full(len(X), np.mean(y)))


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

    expected_impl = impl
    if name == "RandomForestClassifier":
        if isinstance(selector, GreedyImplementationSelector):
            expected_impl = RustHistogramRandomForestClassifier

    assert type(op) is expected_impl
    assert isinstance(op, family) and not op.is_abstract
    if expected_impl is SklearnRandomForest or name != "RandomForestClassifier":
        # Reference implementations run the user's estimator as given.
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

    default_registry = build_default_physical_registry()
    registry = PhysicalRegistry(
        candidate for candidate in default_registry.candidates_for(RandomForestOp)
        if candidate.backend_name == "sklearn-skrub"
    )
    registry.register(PhysicalImpl(
        op_type=RandomForestOp, backend_name="stratum",
        input_format="frame", output_format="frame",
        supports=ClassifierOnlyForest.supports, cost=_placeholder_cost,
        exec_mem=_placeholder_exec_mem, execute=_current_process_execute,
        impl_class=ClassifierOnlyForest,
        implementation_name="rf_hist",
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
    # The generic default policy keeps sklearn when no family preference names
    # either candidate.
    assert bound(_estimator("sklearn.ensemble", "RandomForestClassifier"),
                 DefaultImplementationSelector()) is SklearnRandomForest


def _bound_forest(selector, estimator=None, **op_kwargs):
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=3, random_state=0)
    op = lower_predictor(PredictorOp(estimator=estimator, **op_kwargs), _ctx())
    select_implementations(op, _ctx(), selector=selector)
    return op


def test_random_forest_policy_keeps_default_on_sklearn_and_greedy_on_histogram():
    default = _bound_forest(DefaultImplementationSelector())
    hist = _bound_forest(GreedyImplementationSelector())

    assert type(default) is SklearnRandomForest
    assert type(hist) is RustHistogramRandomForestClassifier
    assert isinstance(hist.original_estimator, RustRandomForestClassifier)
    assert hist.original_estimator._stratum_forest_binding[0] == "histogram"
    assert hist.original_estimator._stratum_forest_fit_args == (128,)
    assert hist.original_estimator._stratum_worker_budget == _ctx().parallelism


def test_greedy_op_preferences_can_make_exact_primary(monkeypatch):
    monkeypatch.setitem(
        GreedyImplementationSelector._OP_PREFERENCES,
        RandomForestOp,
        ("rf_exact", "rf_hist", "sklearn_rf"),
    )
    op = _bound_forest(GreedyImplementationSelector())

    assert type(op) is RustExactRandomForestClassifier
    assert op.original_estimator._stratum_forest_binding[0] == "exact"


def test_greedy_falls_back_to_exact_when_histogram_runtime_is_unavailable(monkeypatch):
    from stratum import _rust_backend as rb

    monkeypatch.setattr(rb, "forest_fit_hist", None)
    op = _bound_forest(GreedyImplementationSelector())

    assert type(op) is RustExactRandomForestClassifier
    assert op.original_estimator._stratum_forest_binding[0] == "exact"


@pytest.mark.parametrize(
    "estimator, op_kwargs",
    [
        (RandomForestClassifier(criterion="entropy"), {}),
        (RandomForestClassifier(), {"kwargs": {"fit": {"sample_weight": [1.0]}}}),
        (RandomForestClassifier(), {"param_refs": {"max_depth": OperandRef(1)}}),
    ],
)
def test_unsupported_native_forest_configuration_retains_sklearn(
    estimator, op_kwargs
):
    op = _bound_forest(
        GreedyImplementationSelector(), estimator=estimator, **op_kwargs
    )

    assert type(op) is SklearnRandomForest


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


class _RefusesRefitParams(Ridge):
    """Rejects any `set_params` once fitted, as CatBoost does."""

    def set_params(self, **params):
        if hasattr(self, "coef_"):
            raise RuntimeError("You can't change params of fitted model.")
        return super().set_params(**params)


def test_graph_fed_parameter_is_applied_to_a_fresh_estimator_every_fold():
    """Each fit starts from an unfitted clone, so the next fold can configure it."""
    X, y = make_regression(n_samples=20, n_features=2, random_state=0)
    op = PredictorOp(estimator=_RefusesRefitParams(), y=y, cols=All(), how="no_wrap",
                     param_refs={"alpha": OperandRef(1)})

    op.process("fit_transform", [X, 2.0])
    op.process("predict", [X, 2.0])
    op.process("fit_transform", [X, 3.0])

    assert op.estimator.alpha == 3.0
    # The prototype the fits are cloned from is never fitted itself.
    assert not hasattr(op.original_estimator, "coef_")


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
    ("regression", "sklearn.ensemble", "ExtraTreesRegressor",
     {"n_estimators": 5, "random_state": 0}, SklearnExtraTrees),
    ("classification", "sklearn.ensemble", "ExtraTreesClassifier",
     {"n_estimators": 5, "random_state": 0}, SklearnExtraTrees),
    ("regression", "sklearn.tree", "DecisionTreeRegressor",
     {"random_state": 0}, SklearnDecisionTree),
    ("classification", "sklearn.tree", "DecisionTreeClassifier",
     {"random_state": 0}, SklearnDecisionTree),
    ("regression", "sklearn.neighbors", "KNeighborsRegressor", {}, SklearnKNeighbors),
    ("classification", "sklearn.neighbors", "KNeighborsClassifier", {},
     SklearnKNeighbors),
    ("regression", "sklearn.ensemble", "HistGradientBoostingRegressor",
     {"max_iter": 5, "random_state": 0}, SklearnHistGradientBoosting),
    ("classification", "sklearn.ensemble", "HistGradientBoostingClassifier",
     {"max_iter": 5, "random_state": 0}, SklearnHistGradientBoosting),
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
