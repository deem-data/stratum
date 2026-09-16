"""Apply with a Choice estimator: conversion, choice unrolling, and evaluation."""
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from skrub._utils import PassThrough
from stratum._api import evaluate
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer._optimize import choice_unrolling, convert_to_ops
from stratum.optimizer.logical._ops import ChoiceOp, PredictorOp, TransformerOp
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

    def test_evaluate_optional(self):
        src = st.as_data_op(self.df[["a", "b"]])
        out = evaluate(src.skb.apply(st.optional(StandardScaler(), name="scale")))

        self.assertEqual(len(out), 2)
        by_id = {o["id"]: o["vals"] for o in out}
        np.testing.assert_allclose(by_id["scale:StandardScaler"].to_numpy(),
                                   StandardScaler().fit_transform(self.df[["a", "b"]]))
        pd.testing.assert_frame_equal(by_id["scale:PassThrough"], self.df[["a", "b"]])


if __name__ == "__main__":
    unittest.main()
