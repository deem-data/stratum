"""Conversion of an Apply's per-method kwargs (`fit_kwargs=`, ...) to the IR.

The kwargs used to be dropped at conversion, which both ignored them at fit time
and left the DataOps nested in them (an eval set, a sample weight) in the DAG
with no consumer, so the next topological walk raised.
"""
import unittest

import numpy as np
import pandas as pd
import stratum as st
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from stratum.optimizer._op_cse import apply_op_cse
from stratum.optimizer._op_utils import topological_iterator, validate_dag
from stratum.optimizer._optimize import convert_to_ops
from stratum.optimizer.ir._ops import (ChoiceOp, OperandRef, PredictorOp,
                                       TransformerOp)


class ApplyKwargsTest(unittest.TestCase):
    def setUp(self):
        # Same setting the pipelines that hit this run under: an eager preview would
        # fit the estimator at construction time, before the kwargs are ours to bind.
        self.enterContext(st.config_context(eager_data_ops=False))
        self.df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "w": [1.0, 1.0, 2.0, 2.0],
            "y": [1.0, 3.0, 5.0, 7.0],
        })

    def _source(self):
        data = st.as_data_op(self.df)
        return data[["a", "b", "w"]], data["y"]


class TestKwargsBinding(ApplyKwargsTest):
    def test_graph_fed_fit_kwarg_becomes_an_operand(self):
        X, y = self._source()
        weights = X["w"] * 2.0
        root = convert_to_ops(X.skb.apply(Ridge(), y=y, fit_kwargs={"sample_weight": weights}))

        self.assertIsInstance(root, PredictorOp)
        ref = root.kwargs["fit"]["sample_weight"]
        self.assertIsInstance(ref, OperandRef)
        # X stays the primary operand; the weights get their own input edge.
        self.assertEqual(len(root.inputs), 3)
        self.assertNotIn(ref.k, (0, root.y.k))
        self.assertIn(root, root.inputs[ref.k].outputs)
        self.assertEqual(root.kwargs.keys(), {"fit"})

    def test_nested_eval_set_binds_every_data_op(self):
        # `eval_set=[(X_val, y_val)]` is the shape early stopping needs: two
        # DataOps nested in a tuple inside a list.
        X, y = self._source()
        X_val, y_val = X[["a"]], y * 1.0
        root = convert_to_ops(X.skb.apply(
            Ridge(), y=y, fit_kwargs={"eval_set": [(X_val, y_val)], "verbose": False}))

        (eval_pair,) = root.kwargs["fit"]["eval_set"]
        self.assertTrue(all(isinstance(r, OperandRef) for r in eval_pair))
        self.assertNotEqual(eval_pair[0].k, eval_pair[1].k)
        self.assertFalse(root.kwargs["fit"]["verbose"])  # constants pass through
        # Every ref indexes a real input, and no op is left without a consumer.
        validate_dag(root)
        for ref in eval_pair:
            self.assertIn(root, root.inputs[ref.k].outputs)

    def test_no_orphan_ops_left_in_the_dag(self):
        # The reported crash: the ops built for the kwargs' DataOps were wired as
        # outputs of their producer but were nobody's input, so the traversal hit
        # a node missing from its in-degree map.
        X, y = self._source()
        dag = X.skb.apply(Ridge(), y=y, fit_kwargs={"eval_set": [(X[["a"]], y)]})
        root = convert_to_ops(dag)

        ops = list(topological_iterator(root))
        consumed = {id(in_op) for op in ops for in_op in op.inputs}
        for op in ops:
            if op is root:
                continue
            self.assertIn(id(op), consumed, f"{op} has no consumer")

    def test_repeated_data_op_shares_one_input_edge(self):
        X, y = self._source()
        root = convert_to_ops(X.skb.apply(Ridge(), y=y, fit_kwargs={"eval_set": [(X, y)]}))

        pair = root.kwargs["fit"]["eval_set"][0]
        self.assertEqual(pair[0].k, 0)  # the same X as the primary operand
        y_ref = root.y
        self.assertEqual(pair[1].k, y_ref.k)
        self.assertEqual(len(root.inputs), 2)

    def test_constant_only_kwargs_need_no_operand(self):
        X, y = self._source()
        root = convert_to_ops(X.skb.apply(Ridge(), y=y, fit_kwargs={"sample_weight": None}))
        self.assertEqual(root.kwargs, {"fit": {"sample_weight": None}})
        self.assertEqual(len(root.inputs), 2)  # X and y only

    def test_empty_groups_are_dropped(self):
        X, y = self._source()
        root = convert_to_ops(X.skb.apply(Ridge(), y=y))
        self.assertEqual(root.kwargs, {})

    def test_transformer_groups_are_keyed_by_method(self):
        X, _ = self._source()
        root = convert_to_ops(X.skb.apply(StandardScaler(), how="no_wrap",
                                          transform_kwargs={"copy": True}))
        self.assertIsInstance(root, TransformerOp)
        self.assertEqual(root.call_kwargs_key, "transform")
        self.assertEqual(root.method_kwargs(root.call_kwargs_key, root.inputs), {"copy": True})
        self.assertEqual(root.method_kwargs(root.fit_kwargs_key, root.inputs), {})

    def test_group_dead_for_this_estimator_kind_is_bound_but_ignored(self):
        # skrub fits a transformer through fit_transform(), so its `fit` group is
        # never called. We mirror that (warn, do not raise) but still bind the
        # DataOps inside it, or they would be left without a consumer.
        X, _ = self._source()
        dag = X.skb.apply(StandardScaler(), how="no_wrap", fit_kwargs={"extra": X[["a"]]})
        with self.assertLogs("stratum", level="WARNING") as logs:
            root = convert_to_ops(dag)

        self.assertIn("fit_kwargs` is ignored", "\n".join(logs.output))
        self.assertIsInstance(root.kwargs["fit"]["extra"], OperandRef)
        self.assertEqual(root.method_kwargs(root.fit_kwargs_key, root.inputs), {})
        validate_dag(root)


class TestUnsupportedKwargs(ApplyKwargsTest):
    def test_group_for_a_method_stratum_never_calls_raises(self):
        X, y = self._source()
        for group in ("predict_proba_kwargs", "decision_function_kwargs", "score_kwargs"):
            with self.subTest(group=group):
                dag = X.skb.apply(Ridge(), y=y, **{group: {"a": 1}})
                with self.assertRaises(NotImplementedError) as ctx:
                    convert_to_ops(dag)
                self.assertIn(group.removesuffix("_kwargs"), str(ctx.exception))

    def test_choice_inside_kwargs_raises(self):
        # A choice would have to expand the parameter grid; its DataOp outcomes are
        # graph children, so keeping it silently would orphan them.
        X, y = self._source()
        dag = X.skb.apply(Ridge(), y=y,
                          fit_kwargs={"sample_weight": st.choose_from([None, 1], name="w")})
        with self.assertRaises(NotImplementedError) as ctx:
            convert_to_ops(dag)
        self.assertIn("choice", str(ctx.exception))

    def test_nested_choice_inside_kwargs_raises(self):
        X, y = self._source()
        dag = X.skb.apply(Ridge(), y=y,
                          fit_kwargs={"eval_set": [(X, st.choose_from([1, 2], name="c"))]})
        with self.assertRaises(NotImplementedError):
            convert_to_ops(dag)


class TestKwargsSurviveRewrites(ApplyKwargsTest):
    def test_cse_keeps_applies_with_different_fit_kwargs_apart(self):
        X, y = self._source()
        light, heavy = X["w"], X["w"] * 10.0
        first = X.skb.apply(Ridge(), y=y, fit_kwargs={"sample_weight": light})
        second = X.skb.apply(Ridge(), y=y, fit_kwargs={"sample_weight": heavy})
        root = apply_op_cse(convert_to_ops(first + second))

        estimators = [op for op in topological_iterator(root) if isinstance(op, PredictorOp)]
        self.assertEqual(len(estimators), 2)
        self.assertNotEqual(estimators[0].structure_key(), estimators[1].structure_key())

    def test_estimator_choice_binds_kwargs_per_outcome(self):
        X, y = self._source()
        weights = X["w"] * 2.0
        root = convert_to_ops(X.skb.apply(
            st.choose_from([Ridge(), DummyRegressor()], name="model"),
            y=y, fit_kwargs={"sample_weight": weights}))

        self.assertIsInstance(root, ChoiceOp)
        for est_op in root.inputs:
            ref = est_op.kwargs["fit"]["sample_weight"]
            self.assertIsInstance(ref, OperandRef)
            self.assertIn(est_op, est_op.inputs[ref.k].outputs)
        validate_dag(root)

    def test_clone_does_not_share_the_kwargs_dict(self):
        X, y = self._source()
        root = convert_to_ops(X.skb.apply(Ridge(), y=y,
                                          fit_kwargs={"sample_weight": X["w"]}))
        clone = root.clone()
        self.assertEqual(clone.kwargs, root.kwargs)
        self.assertIsNot(clone.kwargs, root.kwargs)
        self.assertIsNot(clone.kwargs["fit"], root.kwargs["fit"])


class TestMethodKwargs(ApplyKwargsTest):
    def test_resolves_refs_against_inputs(self):
        op = PredictorOp(estimator=Ridge(), kwargs={"fit": {"sample_weight": OperandRef(1)}})
        weights = np.array([1.0, 2.0])
        self.assertIs(op.method_kwargs("fit", [None, weights])["sample_weight"], weights)

    def test_a_whole_graph_fed_group_is_resolved(self):
        # `fit_kwargs=some_data_op` is legal in skrub: the group itself is a DataOp
        # that evaluates to a dict.
        op = PredictorOp(estimator=Ridge(), kwargs={"fit": OperandRef(1)})
        self.assertEqual(op.method_kwargs("fit", [None, {"sample_weight": 3}]),
                         {"sample_weight": 3})

    def test_a_group_that_resolves_to_a_non_dict_raises(self):
        op = PredictorOp(estimator=Ridge(), kwargs={"fit": OperandRef(1)})
        with self.assertRaises(TypeError):
            op.method_kwargs("fit", [None, "not a dict"])

    def test_missing_or_none_group_is_empty(self):
        op = PredictorOp(estimator=Ridge(), kwargs={"fit": None})
        self.assertEqual(op.method_kwargs("fit", []), {})
        self.assertEqual(op.method_kwargs("predict", []), {})
        self.assertEqual(op.method_kwargs(None, []), {})


if __name__ == "__main__":
    unittest.main()
