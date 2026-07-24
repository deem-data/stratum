"""Implementation selection: registry-driven choice and plan-time binding.

Choosing an impl swaps the op to the impl's concrete ``PhysicalOp`` class and
runs its ``on_impl_selected`` at plan time -- e.g. the Rust kernels swap the op's
estimators for the Rust adapter there -- so execution is the op's ordinary
``process`` with no selection left in it.
"""
import unittest

import pandas as pd

import stratum as st
from skrub import StringEncoder

from stratum.adapters.string_encoder import (RustyStringEncoder,
                                             supports_rust_string_encoder)
from stratum.optimizer._optimize import optimize
from stratum.optimizer.ir._ops import TransformerOp
from stratum.optimizer.physical._impl_selection import (
    DefaultImplementationSelector,
    FlagBasedSelector,
    get_implementation_selector,
    select_implementations,
)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._registry import (PhysicalImpl, PhysicalRegistry,
                                                  _current_process_execute,
                                                  _placeholder_cost,
                                                  _placeholder_exec_mem)
from stratum.optimizer.physical._transform_execs import StringEncoderOp
from stratum.optimizer.ir._ops import Op, ValueOp


def _ctx(backend="pandas", rust=False):
    return PlanContext(backend=backend, pandas_query=False, rechunk=True,
                       parallelism=1, rust_backend=rust, allow_patch=True)


def _impl(op_type, backend, supports=lambda op, ctx: True, impl_class=None):
    return PhysicalImpl(op_type=op_type, backend_name=backend,
                        input_format="frame", output_format="frame",
                        supports=supports, cost=_placeholder_cost,
                        exec_mem=_placeholder_exec_mem,
                        execute=_current_process_execute,
                        impl_class=impl_class)


class DummyOp(Op):
    def process(self, mode, inputs):
        return "dummy"


class TestFlagBasedSelector(unittest.TestCase):
    def test_rust_preferred_only_when_enabled(self):
        selector = FlagBasedSelector()
        rust = _impl(DummyOp, "rust")
        generic = _impl(DummyOp, "sklearn-skrub")
        self.assertIs(generic, selector.choose(DummyOp(), [rust, generic], _ctx()))
        self.assertIs(rust, selector.choose(DummyOp(), [rust, generic], _ctx(rust=True)))

    def test_allow_patch_gates_rust(self):
        # Legacy semantics: rust runs only under allow_patch AND rust_backend.
        selector = FlagBasedSelector()
        rust = _impl(DummyOp, "rust")
        generic = _impl(DummyOp, "sklearn-skrub")
        ctx = PlanContext(backend="pandas", pandas_query=False, rechunk=True,
                          parallelism=1, rust_backend=True, allow_patch=False)
        self.assertIs(generic, selector.choose(DummyOp(), [rust, generic], ctx))

    def test_no_candidates_returns_none(self):
        self.assertIsNone(FlagBasedSelector().choose(DummyOp(), [], _ctx()))


class TestDefaultImplementationSelector(unittest.TestCase):
    def test_default_selector_is_configurable_and_snapshotted(self):
        with st.config(implementation_selector="default"):
            self.assertEqual(
                "default", PlanContext.from_flags().implementation_selector)
            self.assertIsInstance(
                get_implementation_selector("default"),
                DefaultImplementationSelector)

    def test_unknown_selector_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            with st.config(implementation_selector="unknown"):
                pass

    def test_preference_order_is_independent_of_plan_backend(self):
        selector = DefaultImplementationSelector()
        pandas = _impl(DummyOp, "pandas")
        polars = _impl(DummyOp, "polars")
        sklearn = _impl(DummyOp, "sklearn-skrub")
        numpy = _impl(DummyOp, "numpy")
        rust = _impl(DummyOp, "rust")

        self.assertIs(pandas, selector.choose(
            DummyOp(), [rust, polars, numpy, sklearn, pandas], _ctx("polars")))
        self.assertIs(sklearn, selector.choose(
            DummyOp(), [rust, polars, numpy, sklearn], _ctx()))
        self.assertIs(numpy, selector.choose(
            DummyOp(), [rust, polars, numpy], _ctx()))

    def test_falls_back_to_first_supported_candidate(self):
        selector = DefaultImplementationSelector()
        polars = _impl(DummyOp, "polars")
        rust = _impl(DummyOp, "rust")
        self.assertIs(polars, selector.choose(DummyOp(), [polars, rust], _ctx()))
        self.assertIsNone(selector.choose(DummyOp(), [], _ctx()))


class TestPlanTimeBinding(unittest.TestCase):
    def test_on_impl_selected_runs_at_plan_time(self):
        """Choosing an impl swaps the op to its class and runs on_impl_selected."""
        bound = []

        class BoundDummyOp(DummyOp, PhysicalOp):
            is_abstract = False
            def on_impl_selected(self, ctx):
                bound.append(self)

        registry = PhysicalRegistry()
        registry.register(_impl(DummyOp, "pandas", impl_class=BoundDummyOp))

        op = DummyOp()
        select_implementations(op, _ctx(), registry=registry)
        self.assertIsInstance(op, BoundDummyOp)
        self.assertEqual([op], bound)
        # Execution afterwards is the op's plain process -- nothing left to decide.
        self.assertEqual("dummy", op.process("fit_transform", []))

    def test_supports_filter_excludes_candidates(self):
        class FailDummyOp(DummyOp, PhysicalOp):
            is_abstract = False
            def on_impl_selected(self, ctx):
                raise AssertionError("unsupported impl must not be bound")

        registry = PhysicalRegistry()
        registry.register(_impl(DummyOp, "pandas", supports=lambda op, ctx: False,
                                impl_class=FailDummyOp))
        op = DummyOp()
        select_implementations(op, _ctx(), registry=registry)  # no-op, no error
        self.assertNotIsInstance(op, FailDummyOp)


class TestTransformerPlanTimeBinding(unittest.TestCase):
    """Default selection stays on skrub; explicit legacy selection can bind Rust."""

    def setUp(self):
        encoder = StringEncoder(vectorizer="tfidf", analyzer="char", n_components=2)
        supported, reason = supports_rust_string_encoder(encoder)
        if not supported:
            self.skipTest(f"Rust StringEncoder unavailable: {reason}")
        self.encoder = encoder

    def test_default_selector_prefers_skrub_over_rust(self):
        df = pd.DataFrame({"a": ["apple", "banana", "cherry", "orange"]})
        data = st.as_data_op(df).skb.apply(self.encoder, cols=["a"])
        with st.config(rust_backend=True):
            ops, *_ = optimize(data)
        transformer_ops = [op for op in ops if isinstance(op, TransformerOp)]
        self.assertEqual(1, len(transformer_ops))
        op = transformer_ops[0]
        self.assertNotIsInstance(op.estimator, RustyStringEncoder)
        self.assertNotIsInstance(op.original_estimator, RustyStringEncoder)

    def test_explicit_flag_selector_binds_rust_for_abstract_physical_op(self):
        op = StringEncoderOp(estimator=self.encoder, cols=["a"])
        select_implementations(
            op,
            _ctx(rust=True),
            selector=FlagBasedSelector(),
        )
        self.assertIsInstance(op.estimator, RustyStringEncoder)
        self.assertIsInstance(op.original_estimator, RustyStringEncoder)
        self.assertTrue(op.original_estimator._stratum_force_rust)

    def test_no_swap_when_rust_disabled(self):
        df = pd.DataFrame({"a": ["apple", "banana", "cherry", "orange"]})
        data = st.as_data_op(df).skb.apply(self.encoder, cols=["a"])
        with st.config(rust_backend=False):
            ops, *_ = optimize(data)
        op = [op for op in ops if isinstance(op, TransformerOp)][0]
        self.assertNotIsInstance(op.estimator, RustyStringEncoder)


if __name__ == "__main__":
    unittest.main()
