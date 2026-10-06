"""Choices in Apply estimators and op arguments: conversion, choice unrolling, and evaluation."""
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from skrub import ApplyToCols
from skrub._utils import PassThrough
from stratum._api import evaluate
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer._optimize import choice_unrolling, convert_to_ops
from stratum.optimizer.logical._ops import ChoiceOp, GetItemOp, OperandRef, PredictorOp, TransformerOp, _iter_choices
import numpy as np
import pandas as pd
import stratum as st
import unittest


class TestApplyChoice(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "s": ["u", "v", "w", "x"],
            "y": [1.0, 3.0, 5.0, 7.0],
        })

    def _scaler_choice_dag(self, scalers):
        src = st.as_data_op(self.df[["a", "b", "s"]])
        return src.skb.select(st.selectors.numeric()).skb.apply(
            st.choose_from(scalers, name="scaler"))

    def test_convert_transformer_choice(self):
        scalers = [StandardScaler(), MinMaxScaler()]
        root = convert_to_ops(self._scaler_choice_dag(scalers))

        self.assertIsInstance(root, ChoiceOp)
        self.assertEqual(len(root.inputs), 2)
        for est_op, scaler in zip(root.inputs, scalers):
            self.assertIsInstance(est_op, TransformerOp)
            self.assertIs(est_op.estimator, scaler)
            self.assertIn(root, est_op.outputs)
            self.assertIn(est_op, est_op.inputs[0].outputs)
        # Both outcomes consume the same upstream (selected) frame op.
        self.assertIs(root.inputs[0].inputs[0], root.inputs[1].inputs[0])
        self.assertEqual(root.make_outcome_names(),
                         ["scaler:StandardScaler", "scaler:MinMaxScaler"])

    def test_convert_predictor_choice(self):
        data = st.as_data_op(self.df)
        pred = data[["a", "b"]].skb.apply(
            st.choose_from([Ridge(), DummyRegressor()], name="model"), y=data["y"])
        root = convert_to_ops(pred)

        self.assertIsInstance(root, ChoiceOp)
        self.assertEqual(len(root.inputs), 2)
        for est_op in root.inputs:
            self.assertIsInstance(est_op, PredictorOp)
            self.assertEqual(len(est_op.inputs), 2)  # X and the graph-fed y
        # X and y ops are shared across the outcomes.
        self.assertIs(root.inputs[0].inputs[0], root.inputs[1].inputs[0])
        self.assertIs(root.inputs[0].inputs[1], root.inputs[1].inputs[1])

    def test_convert_optional(self):
        src = st.as_data_op(self.df[["a", "b"]])
        root = convert_to_ops(src.skb.apply(st.optional(StandardScaler(), name="scale")))

        self.assertIsInstance(root, ChoiceOp)
        estimators = [est_op.estimator for est_op in root.inputs]
        self.assertEqual({type(e).__name__ for e in estimators},
                         {"StandardScaler", "PassThrough"})
        for est_op in root.inputs:
            self.assertIsInstance(est_op, TransformerOp)

    def test_convert_dataop_outcome_rejected(self):
        src = st.as_data_op(self.df[["a", "b"]])
        dag = src.skb.apply(st.choose_from(
            [st.as_data_op(StandardScaler()), MinMaxScaler()], name="scaler"))
        with self.assertRaises(NotImplementedError):
            convert_to_ops(dag)

    def test_convert_nested_choice_is_flattened(self):
        src = st.as_data_op(self.df[["a", "b"]])
        inner = st.choose_from([StandardScaler(), MinMaxScaler()], name="inner")
        outer = st.choose_from([inner, RobustScaler()], name="outer")
        root = convert_to_ops(src.skb.apply(outer))

        # Flattened to one ChoiceOp over the three leaf estimators (as skrub's grid).
        self.assertIsInstance(root, ChoiceOp)
        self.assertEqual(len(root.inputs), 3)
        self.assertEqual([type(op.estimator).__name__ for op in root.inputs],
                         ["StandardScaler", "MinMaxScaler", "RobustScaler"])
        # The named inner choice contributes to the leaf name paths.
        self.assertEqual(root.make_outcome_names(),
                         ["inner:StandardScaler", "inner:MinMaxScaler", "outer:RobustScaler"])

    def test_evaluate_nested_choice(self):
        src = st.as_data_op(self.df[["a", "b"]])
        inner = st.choose_from([StandardScaler(), MinMaxScaler()], name="inner")
        outer = st.choose_from([inner, RobustScaler()], name="outer")
        out = evaluate(src.skb.apply(outer))

        self.assertEqual(len(out), 3)
        by_id = {o["id"]: o["vals"] for o in out}
        numeric = self.df[["a", "b"]]
        np.testing.assert_allclose(by_id["inner:StandardScaler"].to_numpy(),
                                   StandardScaler().fit_transform(numeric))
        np.testing.assert_allclose(by_id["inner:MinMaxScaler"].to_numpy(),
                                   MinMaxScaler().fit_transform(numeric))
        np.testing.assert_allclose(by_id["outer:RobustScaler"].to_numpy(),
                                   RobustScaler().fit_transform(numeric))

    def test_choice_unrolling_clones_do_not_share_fitted_state(self):
        data = st.as_data_op(self.df)
        X = data[["a", "b"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        scaled = X.skb.apply(st.choose_from([StandardScaler(), MinMaxScaler()], name="scaler"))
        pred = scaled.skb.apply(Ridge(), y=y)

        root = convert_to_ops(pred)
        self.assertIsInstance(root, PredictorOp)
        root = choice_unrolling(root)

        self.assertIsInstance(root, ChoiceOp)
        self.assertEqual(root.make_outcome_names(),
                         ["scaler:StandardScaler", "scaler:MinMaxScaler"])
        ridge_ops = [op for op in topological_iterator(root) if isinstance(op, PredictorOp)]
        self.assertEqual(len(ridge_ops), 2)
        self.assertIsNot(ridge_ops[0].estimator, ridge_ops[1].estimator)
        self.assertEqual(sum(op.was_cloned for op in ridge_ops), 1)
        scaler_ops = [op for op in topological_iterator(root) if isinstance(op, TransformerOp)]
        self.assertEqual(len(scaler_ops), 2)
        self.assertIsNot(scaler_ops[0].estimator, scaler_ops[1].estimator)

    def test_evaluate_transformer_choice(self):
        out = evaluate(self._scaler_choice_dag([StandardScaler(), MinMaxScaler()]))

        self.assertEqual(len(out), 2)
        by_id = {o["id"]: o["vals"] for o in out}
        numeric = self.df[["a", "b"]]
        np.testing.assert_allclose(by_id["scaler:StandardScaler"].to_numpy(),
                                   StandardScaler().fit_transform(numeric))
        np.testing.assert_allclose(by_id["scaler:MinMaxScaler"].to_numpy(),
                                   MinMaxScaler().fit_transform(numeric))

    def test_evaluate_choice_nested_in_cols(self):
        frame = self.df[["a", "b", "s"]]
        src = st.as_data_op(frame)
        out = evaluate(src.skb.apply(
            StandardScaler(), cols=[st.choose_from(["a", "b"], name="col")]))

        by_id = {o["id"]: o["vals"] for o in out}
        self.assertEqual(set(by_id), {"col:Opt0", "col:Opt1"})
        for i, col in enumerate(("a", "b")):
            expected = ApplyToCols(StandardScaler(), cols=[col]).fit_transform(frame)
            pd.testing.assert_frame_equal(by_id[f"col:Opt{i}"], expected)

    def test_evaluate_optional(self):
        src = st.as_data_op(self.df[["a", "b"]])
        out = evaluate(src.skb.apply(st.optional(StandardScaler(), name="scale")))

        self.assertEqual(len(out), 2)
        by_id = {o["id"]: o["vals"] for o in out}
        np.testing.assert_allclose(by_id["scale:StandardScaler"].to_numpy(),
                                   StandardScaler().fit_transform(self.df[["a", "b"]]))
        pd.testing.assert_frame_equal(by_id["scale:PassThrough"], self.df[["a", "b"]])


class TestApplyParamChoice(unittest.TestCase):
    """Choices nested in the estimator's parameters (#223) expand at conversion."""

    def setUp(self):
        self.df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "y": [1.0, 3.0, 5.0, 7.0],
        })

    def _predict(self, estimator):
        data = st.as_data_op(self.df)
        return data[["a", "b"]].skb.apply(estimator, y=data["y"])

    def _assert_concrete(self, root):
        for est_op in root.inputs:
            self.assertEqual(list(_iter_choices(est_op.estimator)), [])

    def test_convert_param_choice(self):
        root = convert_to_ops(self._predict(
            Ridge(alpha=st.choose_from([0.1, 1.0], name="alpha"))))

        self.assertIsInstance(root, ChoiceOp)
        self.assertEqual([op.estimator.alpha for op in root.inputs], [0.1, 1.0])
        self._assert_concrete(root)
        self.assertEqual(root.make_outcome_names(), ["alpha:0.1", "alpha:1.0"])
        # X and y ops are shared across the outcomes.
        self.assertIs(root.inputs[0].inputs[0], root.inputs[1].inputs[0])
        self.assertIs(root.inputs[0].inputs[1], root.inputs[1].inputs[1])

    def test_convert_param_choices_are_a_cartesian_product(self):
        root = convert_to_ops(self._predict(Ridge(
            alpha=st.choose_from([0.1, 1.0], name="alpha"),
            fit_intercept=st.choose_bool(name="intercept"))))

        self.assertEqual(len(root.inputs), 4)
        self.assertEqual({(op.estimator.alpha, op.estimator.fit_intercept) for op in root.inputs},
                         {(0.1, True), (0.1, False), (1.0, True), (1.0, False)})
        self.assertEqual(root.make_outcome_names()[0], "alpha:0.1, intercept:True")

    def test_convert_estimator_choice_with_param_choice(self):
        model = st.choose_from({
            "ridge": Ridge(alpha=st.choose_from([0.1, 1.0], name="alpha")),
            "dummy": DummyRegressor(),
        }, name="model")
        root = convert_to_ops(self._predict(model))

        # The nested choice only varies under the outcome that holds it, as in skrub.
        self.assertEqual(root.make_outcome_names(),
                         ["model:ridge, alpha:0.1", "model:ridge, alpha:1.0", "model:dummy"])
        self.assertEqual([type(op.estimator).__name__ for op in root.inputs],
                         ["Ridge", "Ridge", "DummyRegressor"])
        self._assert_concrete(root)

    def test_convert_choice_in_pipeline_step(self):
        pipe = Pipeline([("scale", st.choose_from([StandardScaler(), MinMaxScaler()], name="s")),
                         ("ridge", Ridge())])
        root = convert_to_ops(self._predict(pipe))

        self.assertEqual([type(op.estimator.steps[0][1]).__name__ for op in root.inputs],
                         ["StandardScaler", "MinMaxScaler"])
        self.assertEqual(root.make_outcome_names(), ["s:StandardScaler", "s:MinMaxScaler"])
        self._assert_concrete(root)

    def test_convert_discretized_numeric_choice(self):
        root = convert_to_ops(self._predict(
            Ridge(alpha=st.choose_float(0.1, 10.0, log=True, n_steps=3, name="alpha"))))

        np.testing.assert_allclose([op.estimator.alpha for op in root.inputs], [0.1, 1.0, 10.0])
        self._assert_concrete(root)

    def test_convert_continuous_numeric_choice_rejected(self):
        with self.assertRaises(NotImplementedError):
            convert_to_ops(self._predict(Ridge(alpha=st.choose_float(0.1, 10.0, name="alpha"))))

    def test_convert_match_rejected(self):
        model = st.choose_from(["low", "high"], name="level")
        with self.assertRaises(NotImplementedError):
            convert_to_ops(self._predict(Ridge(alpha=model.match({"low": 0.1, "high": 1.0}))))

    def test_choice_used_twice_in_one_estimator_is_one_dimension(self):
        k = st.choose_from([1, 2], name="k")
        pipe = Pipeline([("pca", PCA(n_components=k)), ("knn", KNeighborsRegressor(n_neighbors=k))])
        root = convert_to_ops(self._predict(pipe))

        # skrub keys a choice by identity: both uses take the same outcome.
        self.assertEqual([(op.estimator.steps[0][1].n_components,
                           op.estimator.steps[1][1].n_neighbors) for op in root.inputs],
                         [(1, 1), (2, 2)])
        self.assertEqual(root.make_outcome_names(), ["k:1", "k:2"])

    def test_choice_shared_by_two_applies_rejected(self):
        k = st.choose_from([1, 2], name="k")
        data = st.as_data_op(self.df)
        reduced = data[["a", "b"]].skb.apply(PCA(n_components=k))
        pred = reduced.skb.apply(KNeighborsRegressor(n_neighbors=k), y=data["y"])
        with self.assertRaises(NotImplementedError) as ctx:
            convert_to_ops(pred)
        self.assertIn("more than one operator", str(ctx.exception))

    def test_estimator_without_choice_is_kept(self):
        ridge = Ridge()
        root = convert_to_ops(self._predict(ridge))

        self.assertIsInstance(root, PredictorOp)
        self.assertIs(root.estimator, ridge)


class TestNestedChoice(unittest.TestCase):
    """A choice nested in an op's arguments binds to a ChoiceOp, one per choice object."""

    def setUp(self):
        self.data = st.as_data_op(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0.0, 1.0]}))

    def test_getitem_key_binds_to_a_choice_op(self):
        root = convert_to_ops(self.data[[st.choose_from(["a", "b"], name="col")]])

        self.assertIsInstance(root, GetItemOp)
        self.assertEqual(root.key, [OperandRef(1)])
        choice = root.inputs[1]
        self.assertIsInstance(choice, ChoiceOp)
        self.assertEqual([op.value for op in choice.inputs], ["a", "b"])
        self.assertEqual(choice.make_outcome_names(), ["col:Opt0", "col:Opt1"])

    def test_choice_used_twice_is_one_choice_op(self):
        col = st.choose_from(["a", "b"], name="col")
        root = convert_to_ops(self.data[[col]] + self.data[[st.as_data_op(col)]])

        choices = [op for op in topological_iterator(root) if isinstance(op, ChoiceOp)]
        self.assertEqual(len(choices), 1)
        self.assertEqual(len(choices[0].outputs), 2)

    def test_choice_shared_with_an_estimator_rejected(self):
        alpha = st.choose_from([0.1, 1.0], name="alpha")
        pred = (self.data[["a"]] * alpha).skb.apply(Ridge(alpha=alpha), y=self.data["y"])
        with self.assertRaises(NotImplementedError) as ctx:
            convert_to_ops(pred)
        self.assertIn("more than one operator", str(ctx.exception))

    def test_unsupported_nested_choices_rejected(self):
        a = self.data[["a"]]
        choice = st.choose_from(["a", "b"], name="c")
        for dag in (a * st.choose_float(0.1, 1.0, name="f"),
                    self.data[[choice.match({"a": "a", "b": "b"})]],
                    a.skb.apply_func(lambda df, m: df, Ridge(alpha=st.choose_from([0.1, 1.0]))),
                    st.deferred(st.choose_from([abs, round]))(a)):
            with self.subTest(dag=dag), self.assertRaises(NotImplementedError):
                convert_to_ops(dag)


if __name__ == "__main__":
    unittest.main()
