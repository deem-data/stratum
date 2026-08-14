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
from stratum.optimizer.ir._join_ops import JoinOp
from stratum.optimizer.ir._ops import TransformerOp
from stratum.optimizer.ir._selection_ops import SelectionKind, SelectionOp
from stratum.optimizer.physical._impl_selection import (
    DefaultImplementationSelector,
    FlagBasedSelector,
    GreedyImplementationSelector,
    get_implementation_selector,
    select_implementations,
)
from stratum.optimizer.physical._join_execs import PandasJoinOp, PolarsJoinOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._registry import (PhysicalImpl, PhysicalRegistry,
                                                  _current_process_execute,
                                                  _placeholder_cost,
                                                  _placeholder_exec_mem,
                                                  get_default_physical_registry)
from stratum.optimizer.physical._source_execs import (InMemoryFrame,
                                                       PandasInMemoryFrame,
                                                       PolarsInMemoryFrame)
from stratum.optimizer.physical._selection_execs import (
    PandasIndexSelectionOp,
    PolarsSelectionOp,
)
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


class TestGreedyImplementationSelector(unittest.TestCase):
    def test_greedy_selector_is_configurable_and_snapshotted(self):
        with st.config(implementation_selector="greedy"):
            self.assertEqual(
                "greedy", PlanContext.from_flags().implementation_selector)
            self.assertIsInstance(
                get_implementation_selector("greedy"),
                GreedyImplementationSelector)

    def test_preference_order_is_independent_of_plan_backend(self):
        selector = GreedyImplementationSelector()
        pandas = _impl(DummyOp, "pandas")
        polars = _impl(DummyOp, "polars")
        sklearn = _impl(DummyOp, "sklearn-skrub")
        numpy = _impl(DummyOp, "numpy")
        rust = _impl(DummyOp, "rust")

        self.assertIs(rust, selector.choose(
            DummyOp(), [pandas, sklearn, numpy, polars, rust], _ctx("pandas")))
        self.assertIs(polars, selector.choose(
            DummyOp(), [pandas, sklearn, numpy, polars], _ctx()))
        self.assertIs(numpy, selector.choose(
            DummyOp(), [pandas, sklearn, numpy], _ctx()))
        self.assertIs(sklearn, selector.choose(
            DummyOp(), [pandas, sklearn], _ctx()))

    def test_falls_back_to_first_supported_candidate(self):
        selector = GreedyImplementationSelector()
        other_a = _impl(DummyOp, "some-other-backend")
        other_b = _impl(DummyOp, "yet-another-backend")
        self.assertIs(other_a, selector.choose(DummyOp(), [other_a, other_b], _ctx()))
        self.assertIsNone(selector.choose(DummyOp(), [], _ctx()))

    def test_does_not_consult_impl_cost(self):
        # The greedy ranking must stay a static backend preference: it must
        # not call the placeholder PhysicalImpl.cost.
        def _boom(op, stats):
            raise AssertionError("greedy selector must not call cost()")

        rust = PhysicalImpl(op_type=DummyOp, backend_name="rust",
                            input_format="frame", output_format="frame",
                            supports=lambda op, ctx: True, cost=_boom,
                            exec_mem=_placeholder_exec_mem,
                            execute=_current_process_execute)
        selector = GreedyImplementationSelector()
        self.assertIs(rust, selector.choose(DummyOp(), [rust], _ctx()))

    def test_supports_filter_runs_before_either_policy(self):
        # bind_op filters through supports() before the selector runs; a
        # selector never sees an unsupported candidate.
        rust = _impl(DummyOp, "rust", supports=lambda op, ctx: False)
        pandas = _impl(DummyOp, "pandas")
        registry = PhysicalRegistry()
        registry.register(rust)
        registry.register(pandas)
        candidates = [c for c in registry.candidates_for(DummyOp)
                     if c.supports(DummyOp(), _ctx())]
        self.assertEqual([pandas], candidates)
        self.assertIs(pandas, GreedyImplementationSelector().choose(
            DummyOp(), candidates, _ctx()))

    def test_greedy_selector_prefers_polars_for_dataframe_source(self):
        registry = get_default_physical_registry()
        candidates = list(registry.candidates_for(InMemoryFrame))
        op = InMemoryFrame(data=None)
        chosen = GreedyImplementationSelector().choose(op, candidates, _ctx())
        self.assertIs(PolarsInMemoryFrame, chosen.impl_class)

    def test_small_dag_binds_independent_backends_per_operator(self):
        # No global backend or conversion optimization: two ops in one DAG
        # each bind to a different backend under greedy selection, purely
        # from their own candidate lists -- there is no plan-wide reasoning
        # about mixing backends.
        class UpstreamOp(DummyOp, PhysicalOp):
            is_abstract = False
            def on_impl_selected(self, ctx):
                pass

        class DownstreamOp(DummyOp, PhysicalOp):
            is_abstract = False
            def on_impl_selected(self, ctx):
                pass

        class PolarsUpstream(UpstreamOp):
            pass

        class SklearnDownstream(DownstreamOp):
            pass

        registry = PhysicalRegistry()
        # Rust is registered but unsupported here (filtered before selection
        # runs), so polars -- next in the greedy ranking -- wins for Upstream.
        registry.register(_impl(UpstreamOp, "rust", supports=lambda op, ctx: False))
        registry.register(_impl(UpstreamOp, "polars", impl_class=PolarsUpstream))
        registry.register(_impl(DownstreamOp, "sklearn-skrub",
                                impl_class=SklearnDownstream))

        upstream = UpstreamOp(inputs=[])
        downstream = DownstreamOp(inputs=[upstream])
        upstream.add_output(downstream)

        select_implementations(downstream, _ctx(), registry=registry,
                               selector=GreedyImplementationSelector())

        self.assertIsInstance(upstream, PolarsUpstream)
        self.assertIsInstance(downstream, SklearnDownstream)


class TestRelationalImplementationSelection(unittest.TestCase):
    def _relational_ops(self):
        return (
            JoinOp(
                how="left",
                left_on="low_category_0",
                right_on="low_category_0",
            ),
            SelectionOp(kind=SelectionKind.HEAD, args=(5,)),
        )

    def test_default_binds_pandas_join_and_selection(self):
        expected_impls = (PandasJoinOp, PandasIndexSelectionOp)

        for op, expected_impl in zip(self._relational_ops(), expected_impls):
            with self.subTest(op=type(op).__name__):
                select_implementations(
                    op,
                    _ctx(),
                    selector=DefaultImplementationSelector(),
                )
                self.assertIsInstance(op, expected_impl)

    def test_greedy_binds_polars_join_and_selection(self):
        expected_impls = (PolarsJoinOp, PolarsSelectionOp)

        for op, expected_impl in zip(self._relational_ops(), expected_impls):
            with self.subTest(op=type(op).__name__):
                select_implementations(
                    op,
                    _ctx(),
                    selector=GreedyImplementationSelector(),
                )
                self.assertIsInstance(op, expected_impl)


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

    def test_greedy_selector_binds_rust_for_supported_transformer(self):
        # Greedy ranks rust first; unlike the default selector it does not
        # need rust_backend/allow_patch enabled -- support already gates it.
        op = StringEncoderOp(estimator=self.encoder, cols=["a"])
        select_implementations(op, _ctx(), selector=GreedyImplementationSelector())
        self.assertIsInstance(op.estimator, RustyStringEncoder)
        self.assertIsInstance(op.original_estimator, RustyStringEncoder)

    def test_no_swap_when_rust_disabled(self):
        df = pd.DataFrame({"a": ["apple", "banana", "cherry", "orange"]})
        data = st.as_data_op(df).skb.apply(self.encoder, cols=["a"])
        with st.config(rust_backend=False):
            ops, *_ = optimize(data)
        op = [op for op in ops if isinstance(op, TransformerOp)][0]
        self.assertNotIsInstance(op.estimator, RustyStringEncoder)


if __name__ == "__main__":
    unittest.main()
