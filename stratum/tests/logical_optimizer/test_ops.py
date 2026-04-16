import operator
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import stratum as skrub
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from skrub._data_ops._data_ops import DataOp

from stratum.optimizer.ir._ops import (
    DATA_OP_PLACEHOLDER, BinOp, CallOp, DummyConfigManager, GetAttrOp,
    GetItemOp, ImplOp, MethodCallOp, Op, PlaceHolder, SearchEvalOp, ValueOp,
    VariableOp, check_estm_inputs, estimator_parallel_config,
    estm_supports_polars, process_estimator_task, process_transformer_task,
    remove_datops_from_args,
)
from stratum.optimizer._optimize import optimize as optimize_


def _inp(val):
    """Helper: create an Op with a preset intermediate value."""
    op = Op()
    op.intermediate = val
    return op


def _inputs_for(op):
    """Extract intermediate values from op.inputs."""
    return [in_op.intermediate for in_op in op.inputs]


class TestOpCloning(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_op_clone_basic(self):
        op = Op(outputs=[1], inputs=[2])
        op.name = "test_op"
        with self.assertRaises(NotImplementedError):
            op.clone()

    def test_op_clone_value_op(self):
        with self.assertRaises(ValueError):
            ValueOp(1).clone()

    def test_op_clone_search_eval_op(self):
        with self.assertRaises(ValueError):
            SearchEvalOp([]).clone()

    def test_clone_ops(self):
        data = skrub.as_data_op(self.df)
        data_op = data.apply(lambda x: x + 1)
        pred = data_op.skb.apply(DummyRegressor(), y=data["y"])
        pred = pred.skb.apply_func(lambda x, a, b: x, 1, b=1)
        choice = skrub.choose_from([pred], name="choice").as_data_op()
        with skrub.config(fast_dataops_convert=True):
            ops, *_ = optimize_(choice.empty)

        with self.assertRaises(ValueError):
            ops[0].clone()

        cloned = ops[1].clone()
        self.assertIsNot(cloned, ops[1])
        self.assertEqual(ops[1].args, cloned.args)
        self.assertEqual(ops[1].columns, cloned.columns)

        cloned = ops[2].clone()
        self.assertIsNot(cloned, ops[2])
        self.assertEqual(ops[2].key, cloned.key)

        cloned = ops[3].clone()
        self.assertIsNot(ops[3].estimator, cloned.estimator)

        cloned = ops[4].clone()
        self.assertEqual(ops[4].func, cloned.func)
        self.assertEqual(ops[4].args, cloned.args)

        cloned = ops[5].clone()
        self.assertEqual(ops[5].attr_name, cloned.attr_name)

    def test_replace_non_existing_input(self):
        op = Op(outputs=[1], inputs=[2])
        with self.assertRaises(ValueError):
            op.replace_input(3, 4)

    def test_replace_non_existing_output(self):
        op = Op(outputs=[1], inputs=[2])
        with self.assertRaises(ValueError):
            op.replace_output(3, 4)


class TestOpMisc(unittest.TestCase):
    def test_placeholder_str_repr(self):
        ph = PlaceHolder("test")
        self.assertEqual(str(ph), "test")
        self.assertEqual(repr(ph), "test")

    def test_update_name_noop(self):
        Op().update_name()

    def test_check_kwargs_bad_type(self):
        with self.assertRaises(TypeError):
            Op().check_kwargs("not_a_dict")

    def test_remove_datops_bad_type(self):
        with self.assertRaises(ValueError):
            remove_datops_from_args([1, 2, 3])


class TestVariableOp(unittest.TestCase):
    def test_basics(self):
        op = VariableOp(name="x")
        self.assertEqual(op.value, "EMPTY_VARIABLE")
        cloned = op.clone()
        self.assertIsNot(op, cloned)
        self.assertEqual(cloned.name, "x")
        result = op.process("fit_transform", {"x": 123}, _inputs_for(op))
        self.assertEqual(result, 123)


class TestImplOp(unittest.TestCase):
    def test_clone(self):
        cls = type("FakeImpl", (), {
            "__init__": lambda self, a=None, b=None: (setattr(self, 'a', a), setattr(self, 'b', b)) and None,
            "_fields": ["a", "b"],
        })
        op = ImplOp(name="test", skrub_impl=cls(a=1, b=2))
        cloned = op.clone()
        self.assertTrue(cloned.was_cloned)
        self.assertIsNot(op, cloned)

    def test_replace_fields_with_values(self):
        mock_dataop = MagicMock(spec=DataOp)
        cls = type("Impl", (), {"_fields": ["x", "y", "z"], "x": mock_dataop, "y": [mock_dataop, 5], "z": {"k": mock_dataop}})
        op = ImplOp(name="test", skrub_impl=cls())
        op.inputs = [_inp("vx"), _inp("vy"), _inp("vz")]
        inputs = [in_op.intermediate for in_op in op.inputs]
        ns = op.replace_fields_with_values(inputs)
        self.assertEqual(ns.x, "vx")
        self.assertEqual(ns.y[1], 5)
        self.assertEqual(ns.z["k"], "vz")

    def test_process_with_eval(self):
        mock_dataop = MagicMock(spec=DataOp)
        def fake_eval(mode, environment):
            val = yield mock_dataop
            return val * 2
        op = ImplOp(name="test", skrub_impl=SimpleNamespace(eval=fake_eval))
        op.inputs = [_inp(10)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, 20)

    def test_process_without_eval(self):
        cls = type("Impl", (), {"_fields": ["a"], "a": 42, "compute": lambda self, ns, mode, env: ns.a + 1})
        op = ImplOp(name="test", skrub_impl=cls())
        op.inputs = []
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, 43)


class TestUtilFunctions(unittest.TestCase):
    def test_dummy_config_manager(self):
        with DummyConfigManager() as d:
            self.assertIsNotNone(d)

    def test_estimator_parallel_config_none(self):
        self.assertIsInstance(estimator_parallel_config(None), DummyConfigManager)

    def test_estm_supports_polars_sklearn(self):
        self.assertTrue(estm_supports_polars(StandardScaler()))

    def test_estm_supports_polars_unknown(self):
        e = MagicMock()
        e.__class__ = type("Foo", (), {"__module__": "some_other_lib"})
        self.assertFalse(estm_supports_polars(e))

    def _fake_estimator(self):
        e = MagicMock()
        e.__class__ = type("Foo", (), {"__module__": "some_other_lib", "__name__": "Foo"})
        return e

    def test_check_estm_inputs_polars_unsupported(self):
        df, y = pl.DataFrame({"a": [1, 2, 3]}), pl.DataFrame({"b": [4, 5, 6]})
        converted, x_out, y_out = check_estm_inputs(self._fake_estimator(), "fit_transform", df, y)
        self.assertTrue(converted)
        self.assertIsInstance(x_out, pd.DataFrame)
        self.assertIsInstance(y_out, pd.DataFrame)

    def test_check_estm_inputs_polars_predict(self):
        converted, x_out, y_out = check_estm_inputs(self._fake_estimator(), "predict", pl.DataFrame({"a": [1]}), None)
        self.assertTrue(converted)
        self.assertIsNone(y_out)

    def test_check_estm_inputs_sklearn_no_convert(self):
        converted, _, _ = check_estm_inputs(StandardScaler(), "fit_transform", pl.DataFrame({"a": [1]}), None)
        self.assertFalse(converted)

    def test_process_estimator_bad_mode(self):
        with self.assertRaises(ValueError):
            process_estimator_task((StandardScaler(), pd.DataFrame({"a": [1]}), None, None, "no-wrap", False, False, {}, "bad", None))

    def test_process_transformer_bad_mode(self):
        with self.assertRaises(ValueError):
            process_transformer_task((StandardScaler(), pd.DataFrame({"a": [1]}), None, None, "no-wrap", False, False, {}, "bad", None))


class TestOpProcess(unittest.TestCase):
    def test_method_call(self):
        op = MethodCallOp("upper", args=(), kwargs={})
        op.inputs = [_inp("hello")]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, "HELLO")

    def test_method_call_with_placeholders(self):
        op = MethodCallOp("format", args=(DATA_OP_PLACEHOLDER,), kwargs={"end": DATA_OP_PLACEHOLDER})
        op.inputs = [_inp("{0} {end}"), _inp("hello"), _inp("world")]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, "hello world")

    def test_call_op(self):
        op = CallOp(func=lambda a, b: a + b, args=(DATA_OP_PLACEHOLDER, DATA_OP_PLACEHOLDER), kwargs={})
        op.inputs = [_inp(3), _inp(7)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, 10)

    def test_getattr_dataframe_op(self):
        op = GetAttrOp(attr_name=["real", "imag"])
        op.is_dataframe_op = True
        op.inputs = [_inp(1 + 2j)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, 0.0)

    def test_getattr_normal(self):
        op = GetAttrOp(attr_name="real")
        op.inputs = [_inp(3 + 4j)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, 3.0)

    def test_getitem_with_placeholder(self):
        op = GetItemOp(key="dummy")
        op.key = DATA_OP_PLACEHOLDER
        op.inputs = [_inp({"x": 42}), _inp("x")]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, 42)

    def test_binop_both_placeholders(self):
        op = BinOp(op=operator.add, left=DATA_OP_PLACEHOLDER, right=DATA_OP_PLACEHOLDER)
        op.inputs = [_inp(10), _inp(20)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, 30)

    def test_binop_left_literal(self):
        op = BinOp(op=operator.mul, left=5, right=DATA_OP_PLACEHOLDER)
        op.inputs = [_inp(3)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result, 15)
