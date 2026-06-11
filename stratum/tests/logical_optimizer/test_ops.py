import operator
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import stratum as st
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from skrub._data_ops._data_ops import DataOp

from stratum.optimizer.ir._ops import (
    OperandRef, OperandBinder, BinOp, CallOp, DummyConfigManager, GetAttrOp,
    GetItemOp, ImplOp, MethodCallOp, Op, SearchEvalOp, ValueOp,
    VariableOp, check_estm_inputs, estimator_parallel_config,
    estm_supports_polars, process_estimator_task, process_transformer_task,
)
from stratum.optimizer._optimize import optimize as optimize_


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
        data = st.as_data_op(self.df)
        data_op = data.apply(lambda x: x + 1)
        pred = data_op.skb.apply(DummyRegressor(), y=data["y"])
        pred = pred.skb.apply_func(lambda x, a, b: x, 1, b=1)
        choice = st.choose_from([pred], name="choice").as_data_op()
        with st.config(fast_dataops_convert=True):
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
    def test_operand_ref_eq_hash_repr(self):
        self.assertEqual(OperandRef(2), OperandRef(2))
        self.assertNotEqual(OperandRef(2), OperandRef(3))
        self.assertEqual(hash(OperandRef(2)), hash(OperandRef(2)))
        self.assertEqual(repr(OperandRef(1)), "OperandRef(1)")

    def test_update_name_noop(self):
        Op().update_name()

    def test_check_kwargs_bad_type(self):
        with self.assertRaises(TypeError):
            Op().check_kwargs("not_a_dict")


class TestVariableOp(unittest.TestCase):
    def test_basics(self):
        op = VariableOp(name="x")
        self.assertEqual(op.value, "EMPTY_VARIABLE")
        cloned = op.clone()
        self.assertIsNot(op, cloned)
        self.assertEqual(cloned.name, "x")
        result = op.process("fit_transform", {"x": 123}, [])
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
        # The same DataOp appearing in several fields is a single operand (dedup):
        # every occurrence resolves to the same input value.
        mock_dataop = MagicMock(spec=DataOp)
        cls = type("Impl", (), {"_fields": ["x", "y", "z"], "x": mock_dataop, "y": [mock_dataop, 5], "z": {"k": mock_dataop}})
        op = ImplOp(name="test", skrub_impl=cls())
        inputs = ["vx"]
        ns = op.replace_fields_with_values(inputs)
        self.assertEqual(ns.x, "vx")
        self.assertEqual(ns.y[1], 5)
        self.assertEqual(ns.z["k"], "vx")

    def test_process_with_eval(self):
        mock_dataop = MagicMock(spec=DataOp)
        def fake_eval(mode, environment):
            val = yield mock_dataop
            return val * 2
        op = ImplOp(name="test", skrub_impl=SimpleNamespace(eval=fake_eval))
        result = op.process("fit_transform", {}, [10])
        self.assertEqual(result, 20)

    def test_process_without_eval(self):
        cls = type("Impl", (), {"_fields": ["a"], "a": 42, "compute": lambda self, ns, mode, env: ns.a + 1})
        op = ImplOp(name="test", skrub_impl=cls())
        result = op.process("fit_transform", {}, [])
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
        result = op.process("fit_transform", {}, ["hello"])
        self.assertEqual(result, "HELLO")

    def test_method_call_with_placeholders(self):
        # input 0 is the implicit object; the format arg/kwarg reference inputs 1 and 2.
        op = MethodCallOp("format", args=(OperandRef(1),), kwargs={"end": OperandRef(2)})
        result = op.process("fit_transform", {}, ["{0} {end}", "hello", "world"])
        self.assertEqual(result, "hello world")

    def test_call_op(self):
        op = CallOp(func=lambda a, b: a + b, args=(OperandRef(0), OperandRef(1)), kwargs={})
        result = op.process("fit_transform", {}, [3, 7])
        self.assertEqual(result, 10)

    def test_getattr_dataframe_op(self):
        op = GetAttrOp(attr_name=["real", "imag"])
        op.is_dataframe_op = True
        result = op.process("fit_transform", {}, [1 + 2j])
        self.assertEqual(result, 0.0)

    def test_getattr_normal(self):
        op = GetAttrOp(attr_name="real")
        result = op.process("fit_transform", {}, [3 + 4j])
        self.assertEqual(result, 3.0)

    def test_getitem_with_placeholder(self):
        # input 0 is the container, input 1 is the graph-fed key.
        op = GetItemOp(key=OperandRef(1))
        result = op.process("fit_transform", {}, [{"x": 42}, "x"])
        self.assertEqual(result, 42)

    def test_binop_both_placeholders(self):
        op = BinOp(op=operator.add, left=OperandRef(0), right=OperandRef(1))
        result = op.process("fit_transform", {}, [10, 20])
        self.assertEqual(result, 30)

    def test_binop_left_literal(self):
        op = BinOp(op=operator.mul, left=5, right=OperandRef(0))
        result = op.process("fit_transform", {}, [3])
        self.assertEqual(result, 15)


class TestOperandRef(unittest.TestCase):
    def test_eq_and_hash(self):
        self.assertEqual(OperandRef(2), OperandRef(2))
        self.assertNotEqual(OperandRef(2), OperandRef(3))
        # not equal to a bare int with the same value
        self.assertNotEqual(OperandRef(2), 2)
        self.assertEqual(hash(OperandRef(2)), hash(OperandRef(2)))
        # usable as a dict/set key
        self.assertEqual(len({OperandRef(0), OperandRef(0), OperandRef(1)}), 2)

    def test_str_and_repr(self):
        self.assertEqual(str(OperandRef(4)), "$4")
        self.assertEqual(repr(OperandRef(1)), "OperandRef(1)")


class TestOperandBinder(unittest.TestCase):
    """Bind DataOps to OperandRefs through the id->Op lookup, deduplicating edges."""

    def _binder_for(self, *data_ops):
        ids_to_ops = {id(d): ValueOp(i) for i, d in enumerate(data_ops)}
        return OperandBinder(ids_to_ops), ids_to_ops

    def test_repeated_dataop_shares_single_edge(self):
        x = st.as_data_op(1)
        binder, ids = self._binder_for(x)
        # x + x: same upstream DataOp feeding two slots -> one input edge, two refs at 0
        left = binder.ref(x)
        right = binder.ref(x)
        self.assertEqual(left, OperandRef(0))
        self.assertEqual(right, OperandRef(0))
        self.assertEqual(len(binder.inputs), 1)
        self.assertIs(binder.inputs[0], ids[id(x)])

    def test_distinct_dataops_get_increasing_indices(self):
        a, b = st.as_data_op(1), st.as_data_op(2)
        binder, _ = self._binder_for(a, b)
        self.assertEqual(binder.ref(a), OperandRef(0))
        self.assertEqual(binder.ref(b), OperandRef(1))
        self.assertEqual(binder.ref(a), OperandRef(0))  # still deduped
        self.assertEqual(len(binder.inputs), 2)

    def test_bind_recurses_nested_containers(self):
        a, b, c = st.as_data_op(1), st.as_data_op(2), st.as_data_op(3)
        binder, _ = self._binder_for(a, b, c)
        bound = binder.bind({"xs": [a, (b, 7)], "y": c, "lit": "keep"})
        self.assertEqual(
            bound,
            {"xs": [OperandRef(0), (OperandRef(1), 7)], "y": OperandRef(2), "lit": "keep"},
        )
        self.assertEqual(len(binder.inputs), 3)

    def test_bind_seq_and_bind_map(self):
        a, b = st.as_data_op(1), st.as_data_op(2)
        binder, _ = self._binder_for(a, b)
        self.assertEqual(binder.bind_seq((a, 5)), (OperandRef(0), 5))
        self.assertEqual(binder.bind_map({"k": b, "c": 9}), {"k": OperandRef(1), "c": 9})

    def test_ref_op_binds_already_converted_op(self):
        binder = OperandBinder({})
        op0, op1 = ValueOp(1), ValueOp(2)
        self.assertEqual(binder.ref_op(op0), OperandRef(0))
        self.assertEqual(binder.ref_op(op1), OperandRef(1))
        self.assertEqual(binder.ref_op(op0), OperandRef(0))  # dedup by identity


class TestEdgeDedup(unittest.TestCase):
    def test_add_input_dedup_returns_index(self):
        op, a, b = Op(), ValueOp(1), ValueOp(2)
        self.assertEqual(op.add_input(a), 0)
        self.assertEqual(op.add_input(b), 1)
        self.assertEqual(op.add_input(a), 0)  # already present -> existing index, no dup
        self.assertEqual(op.inputs, [a, b])
        self.assertEqual(op.num_input_operands, 2)

    def test_add_output_dedup(self):
        op, out = Op(), Op()
        op.add_output(out)
        op.add_output(out)
        self.assertEqual(op.outputs, [out])
