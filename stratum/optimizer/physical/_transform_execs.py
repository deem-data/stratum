"""Physical transformer operators.

Lowering turns a logical :class:`~stratum.optimizer.logical._ops.TransformerOp`
wrapping a supported estimator into an *abstract* physical transformer op -- one
per estimator kind.
Implementation selection then swaps the abstract op to a concrete backend impl:
the sklearn/skrub reference (``@physical_impl``) or the native Rust kernel
(``@rust_impl``).

Estimators without a dedicated physical operator use the generic Skrub
transformer implementation.
"""

from __future__ import annotations

from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer as _SklearnCountVectorizer
from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler
from typing import Any

from skrub import TableVectorizer as _SkrubTableVectorizer
from skrub import StringEncoder as _SkrubStringEncoder

from stratum.adapters.one_hot_encoder import (
    RustyOneHotEncoder,
    supports_rust_one_hot_encoder,
)
from stratum.adapters.string_encoder import (
    RustyStringEncoder,
    supports_rust_string_encoder,
)
from stratum.adapters.table_vectorizer import (
    StratumFusedTableVectorizer as _StratumFusedTableVectorizer,
)
from stratum.adapters.standard_scaler import (
    NumpyStandardScaler,
    RustyStandardScaler,
    supports_rust_standard_scaler,
)
from stratum.adapters.count_vectorizer import (
    RustyCountVectorizer,
    supports_rust_count_vectorizer,
)
from stratum.optimizer.logical._ops import TransformerOp
from stratum.optimizer.physical._lowering import lowering_rule
from stratum.optimizer.physical._physical_ops import (
    PhysicalOp,
    RustPhysicalOp,
    EstimatorExecution,
)
from stratum.optimizer.physical._registry import (
    physical_impl,
    rust_impl,
    numpy_impl,
    sklearn_skrub_impl,
)


class _TransformerExecution(EstimatorExecution):
    """Shared physical execution for Skrub-compatible transformer estimators."""

    # TODO: specialize transformer sizing, especially the native TF-IDF and
    # SVD/FD handles retained by RustStringEncoder and nested TableVectorizers.
    # For now the generic fallback runs once per fit; native handles remain
    # under-accounted until those kernels expose their retained byte counts.


class PassthroughTransformer(_TransformerExecution, TransformerOp, PhysicalOp):
    """Execute an estimator through the standard Skrub transformer task."""


class TableVectorizerOp(_TransformerExecution, TransformerOp, PhysicalOp):
    """Abstract physical TableVectorizer transformer.

    The reference implementation deliberately keeps Skrub's complete pipeline
    intact. The Stratum implementation uses the fused TableVectorizer adapter
    for supported configurations; unsupported configurations use the reference
    implementation.
    """

    is_abstract = True


@sklearn_skrub_impl(of=TableVectorizerOp)
class SkrubTableVectorizer(TableVectorizerOp):
    """Reference implementation that runs Skrub TableVectorizer unchanged."""

    is_abstract = False


@physical_impl(of=TableVectorizerOp, backend="stratum")
class StratumTableVectorizer(TableVectorizerOp):
    """Skrub TableVectorizer orchestration with supported Rust leaf encoders."""

    is_abstract = False

    @classmethod
    def supports(cls, op: TableVectorizerOp, ctx: Any) -> bool:
        """Select this for mixed/partially supported configs"""
        estimator = op.original_estimator
        if not isinstance(estimator, _SkrubTableVectorizer):
            return False
        if getattr(estimator, "specific_transformers", ()):
            return False
        if _StratumFusedTableVectorizer.supports(estimator, ctx):
            return False
        return any(
            supported
            for supported, _ in (
                supports_rust_one_hot_encoder(estimator.low_cardinality),
                supports_rust_string_encoder(estimator.high_cardinality),
            )
        )

    def on_impl_selected(self, ctx: Any) -> None:
        self.estimator = _bind_table_vectorizer_leaves(self.estimator)
        self.original_estimator = _bind_table_vectorizer_leaves(self.original_estimator)


def _bind_table_vectorizer_leaves(estimator: _SkrubTableVectorizer):
    """Clone a TableVectorizer and replace only supported encoder prototypes."""
    bound = clone(estimator)
    leaves = (
        (
            "low_cardinality",
            supports_rust_one_hot_encoder,
            _as_rusty_one_hot_encoder,
        ),
        (
            "high_cardinality",
            supports_rust_string_encoder,
            _as_rusty_string_encoder,
        ),
    )
    for name, supports, adapt in leaves:
        prototype = getattr(bound, name)
        supported, _ = supports(prototype)
        if supported:
            setattr(bound, name, adapt(prototype))
    return bound


@physical_impl(of=TableVectorizerOp, backend="stratum")
class StratumFusedTableVectorizer(TableVectorizerOp):
    """Multithreaded fused orchestration with eligible Rust leaf encoders."""

    is_abstract = False

    @classmethod
    def supports(cls, op: TableVectorizerOp, ctx: Any) -> bool:
        """Select for fully supported (all Rust) configs"""
        return _StratumFusedTableVectorizer.supports(
            op.original_estimator,
            ctx,
        )

    def on_impl_selected(self, ctx: Any) -> None:
        # Bind both copies before execution. The fused adapter independently
        # clones and selects eligible Rust leaves during fit.
        self.estimator = _as_stratum_fused_table_vectorizer(self.estimator)
        self.original_estimator = _as_stratum_fused_table_vectorizer(
            self.original_estimator
        )


def _as_stratum_fused_table_vectorizer(
    estimator: _SkrubTableVectorizer,
) -> _StratumFusedTableVectorizer:
    """Clone a TableVectorizer configuration into the fused runtime."""
    return _StratumFusedTableVectorizer(**estimator.get_params(deep=False))


class StringEncoderOp(_TransformerExecution, TransformerOp, PhysicalOp):
    """Abstract physical StringEncoder transformer.

    Subclasses the logical ``TransformerOp`` it lowers from, so it carries the
    same estimator state and its concrete impls inherit physical execution
    unchanged -- they differ only in *which* estimator object ``process`` runs
    (the plain skrub encoder, or the Rust adapter swapped in at plan time).
    """

    is_abstract = True


@sklearn_skrub_impl(of=StringEncoderOp)
class SkrubStringEncoder(StringEncoderOp):
    """Reference impl: runs the skrub ``StringEncoder`` as-is."""

    is_abstract = False


@rust_impl(of=StringEncoderOp)
class RustStringEncoder(StringEncoderOp, RustPhysicalOp):
    """Native Rust impl: swaps in the ``RustyStringEncoder`` adapter at plan time,
    so ``process`` runs the Rust kernel with no run-time decision left. Its Rust
    capability hints come from ``RustPhysicalOp`` (Rayon: GIL-free, data-parallel)."""

    is_abstract = False

    @classmethod
    def supports(cls, op: StringEncoderOp, ctx: Any) -> bool:
        supported, _ = supports_rust_string_encoder(op.original_estimator)
        return supported

    def on_impl_selected(self, ctx: Any) -> None:
        self.original_estimator = _as_rusty_string_encoder(self.original_estimator)
        self.estimator = _as_rusty_string_encoder(self.estimator)


def _as_rusty_string_encoder(estimator) -> RustyStringEncoder:
    """Adapt a skrub ``StringEncoder`` to the Rust drop-in, forcing the Rust path."""
    if isinstance(estimator, RustyStringEncoder):
        rusty = estimator
    else:
        rusty = RustyStringEncoder(**estimator.get_params(deep=False))
    rusty._stratum_force_rust = True
    return rusty


# --- OneHotEncoder -----------------------------------------------------------
# Not yet lowered to its own dedicated physical op, so its Rust kernel is keyed
# on the generic pass-through transformer. Selection swaps a supported op to
# this class and its on_impl_selected swaps in the Rust adapter.
@rust_impl(of=PassthroughTransformer, output_format="matrix")
class RustOneHotEncoder(_TransformerExecution, TransformerOp, RustPhysicalOp):
    """Native Rust one-hot encoder: swaps in ``RustyOneHotEncoder`` at plan time."""

    @classmethod
    def supports(cls, op: TransformerOp, ctx: Any) -> bool:
        supported, _ = supports_rust_one_hot_encoder(op.original_estimator)
        return supported

    def on_impl_selected(self, ctx: Any) -> None:
        self.original_estimator = _as_rusty_one_hot_encoder(self.original_estimator)
        self.estimator = _as_rusty_one_hot_encoder(self.estimator)


def _as_rusty_one_hot_encoder(estimator) -> RustyOneHotEncoder:
    """Adapt a sklearn ``OneHotEncoder`` to the Rust drop-in, forcing the Rust path."""
    if isinstance(estimator, RustyOneHotEncoder):
        rusty = estimator
    else:
        # sklearn exposes additional OHE parameters that the Rust kernel does
        # not implement.  They are validated by supports() above and the
        # supported defaults are equivalent to the adapter's defaults, so do
        # not pass them through and accidentally disable the Rust fast path.
        params = estimator.get_params(deep=False)
        rusty = RustyOneHotEncoder(
            drop=params["drop"],
            dtype=params["dtype"],
            handle_unknown=params["handle_unknown"],
            sparse_output=params["sparse_output"],
        )
    rusty._stratum_force_rust = True
    return rusty


# --- StandardScaler ----------------------------------------------------------
class StandardScalerOp(_TransformerExecution, TransformerOp, PhysicalOp):
    """Abstract physical standardization operator."""

    is_abstract = True


@sklearn_skrub_impl(of=StandardScalerOp)
class SklearnStandardScaler(StandardScalerOp):
    """Reference sklearn standardization."""

    is_abstract = False


@numpy_impl(of=StandardScalerOp)
class NumpyStandardScalerExec(StandardScalerOp):
    """NumPy dense statistics with reference fallbacks."""

    is_abstract = False

    @classmethod
    def supports(cls, op, ctx):
        return (
            type(op.original_estimator) is _SklearnStandardScaler and not op.param_refs
        )

    def on_impl_selected(self, ctx):
        self.estimator = NumpyStandardScaler(**self.estimator.get_params(deep=False))
        self.original_estimator = NumpyStandardScaler(
            **self.original_estimator.get_params(deep=False)
        )


@rust_impl(of=StandardScalerOp)
class RustStandardScaler(StandardScalerOp, RustPhysicalOp):
    """Rust kernels for standardization."""

    is_abstract = False

    @classmethod
    def supports(cls, op, ctx):
        return (
            not op.param_refs
            and supports_rust_standard_scaler(op.original_estimator)[0]
        )

    def on_impl_selected(self, ctx):
        for name in ("estimator", "original_estimator"):
            estimator = RustyStandardScaler(
                **getattr(self, name).get_params(deep=False),
            )
            estimator._bind_native_worker_budget(ctx.parallelism)
            estimator._stratum_force_rust = True
            setattr(self, name, estimator)


# --- CountVectorizer ---------------------------------------------------------
class CountVectorizerOp(_TransformerExecution, TransformerOp, PhysicalOp):
    """Abstract physical word-counting operator."""

    is_abstract = True


@sklearn_skrub_impl(of=CountVectorizerOp, output_format="matrix")
class SklearnCountVectorizer(CountVectorizerOp):
    """Reference sklearn text counting."""

    is_abstract = False


@rust_impl(of=CountVectorizerOp, output_format="matrix")
class RustCountVectorizer(CountVectorizerOp, RustPhysicalOp):
    """Parallel native counting with conservative sklearn fallbacks."""

    is_abstract = False

    @classmethod
    def supports(cls, op, ctx):
        return (
            not op.param_refs
            and supports_rust_count_vectorizer(op.original_estimator)[0]
        )

    def on_impl_selected(self, ctx):
        for name in ("estimator", "original_estimator"):
            estimator = RustyCountVectorizer(**getattr(self, name).get_params(deep=False))
            estimator._bind_native_worker_budget(ctx.parallelism)
            estimator._stratum_force_rust = True
            setattr(self, name, estimator)


@lowering_rule(TransformerOp)
def lower_transformer(op: TransformerOp, ctx) -> PhysicalOp | None:
    """Lower a ``TransformerOp`` to the matching abstract physical transformer.

    Dedicated estimators get their own physical op; others run through Skrub.
    """
    if type(op.original_estimator) is _SklearnCountVectorizer:
        return CountVectorizerOp(
            estimator=op.estimator,
            y=op.y,
            cols=op.cols,
            exclude_cols=op.exclude_cols,
            no_wrap=op.no_wrap,
            allow_reject=op.allow_reject,
            unsupervised=op.unsupervised,
            kwargs=op.kwargs,
            param_refs=op.param_refs,
            feeds_estimator=op.feeds_estimator,
        )
    if type(op.original_estimator) is _SklearnStandardScaler:
        return StandardScalerOp(
            estimator=op.estimator,
            y=op.y,
            cols=op.cols,
            exclude_cols=op.exclude_cols,
            no_wrap=op.no_wrap,
            allow_reject=op.allow_reject,
            unsupervised=op.unsupervised,
            kwargs=op.kwargs,
            param_refs=op.param_refs,
            feeds_estimator=op.feeds_estimator,
        )
    if isinstance(op.original_estimator, _SkrubTableVectorizer):
        return TableVectorizerOp(
            estimator=op.estimator,
            y=op.y,
            cols=op.cols,
            exclude_cols=op.exclude_cols,
            no_wrap=op.no_wrap,
            allow_reject=op.allow_reject,
            unsupervised=op.unsupervised,
            kwargs=op.kwargs,
            param_refs=op.param_refs,
            feeds_estimator=op.feeds_estimator,
        )
    if isinstance(op.original_estimator, _SkrubStringEncoder):
        return StringEncoderOp(
            estimator=op.estimator,
            y=op.y,
            cols=op.cols,
            exclude_cols=op.exclude_cols,
            no_wrap=op.no_wrap,
            allow_reject=op.allow_reject,
            unsupervised=op.unsupervised,
            kwargs=op.kwargs,
            param_refs=op.param_refs,
            feeds_estimator=op.feeds_estimator,
        )
    op.__class__ = PassthroughTransformer
    return op
