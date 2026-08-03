"""Physical transformer operators.

Lowering turns a logical :class:`~stratum.optimizer.ir._ops.TransformerOp`
wrapping a supported estimator into an *abstract* physical transformer op -- one
per estimator kind.
Implementation selection then swaps the abstract op to a concrete backend impl:
the sklearn/skrub reference (``@physical_impl``) or the native Rust kernel
(``@rust_impl``).

Lowering is incremental. Only estimators with a branch in
:func:`lower_transformer` move to a dedicated physical op; every other
``TransformerOp`` returns ``None`` from the rule, passes through lowering
unchanged, and keeps running via the ``sklearn-skrub`` impl registered on
``TransformerOp`` itself.
"""
from __future__ import annotations

from typing import Any

from sklearn.base import clone
from skrub import TableVectorizer as _SkrubTableVectorizer
from skrub import StringEncoder as _SkrubStringEncoder

from stratum.adapters.one_hot_encoder import (RustyOneHotEncoder,
                                             supports_rust_one_hot_encoder)
from stratum.adapters.string_encoder import (RustyStringEncoder,
                                             supports_rust_string_encoder)
from stratum.optimizer.ir._ops import TransformerOp
from stratum.optimizer.physical._lowering import lowering_rule
from stratum.optimizer.physical._physical_ops import PhysicalOp, RustPhysicalOp
from stratum.optimizer.physical._registry import (physical_impl, rust_impl,
                                                  sklearn_skrub_impl)


class TableVectorizerOp(TransformerOp, PhysicalOp):
    """Abstract physical TableVectorizer transformer.

    The reference implementation deliberately keeps Skrub's complete pipeline
    intact.  The hybrid implementation below only changes the nested leaf
    estimator prototypes at plan time, so unsupported TableVectorizer
    configurations retain the same execution path.
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
        estimator = op.original_estimator
        if not isinstance(estimator, _SkrubTableVectorizer):
            return False
        if getattr(estimator, "specific_transformers", ()):
            return False
        # Select if at least one encoder can use Rust
        return any(
            supported
            for supported, _ in (
                supports_rust_one_hot_encoder(estimator.low_cardinality),
                supports_rust_string_encoder(estimator.high_cardinality),
            )
        )

    def on_impl_selected(self, ctx: Any) -> None:
        # Bind both estimator copies: BaseEstimatorOp uses the original copy for
        # fitting and stores the fitted result in the other copy.
        self.estimator = _bind_table_vectorizer_leaves(self.estimator)
        self.original_estimator = _bind_table_vectorizer_leaves(
            self.original_estimator
        )


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


class StringEncoderOp(TransformerOp, PhysicalOp):
    """Abstract physical StringEncoder transformer.

    Subclasses the logical ``TransformerOp`` it lowers from, so it carries the
    same estimator state and its concrete impls inherit ``BaseEstimatorOp.process``
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
# Not yet lowered to its own physical op, so its Rust kernel is keyed on the
# logical TransformerOp (same shape as the migrated StringEncoder impls, just
# without an abstract parent op). Selection swaps a supported TransformerOp to
# this class and its on_impl_selected swaps in the Rust adapter.
@rust_impl(of=TransformerOp, output_format="matrix")
class RustOneHotEncoder(TransformerOp, RustPhysicalOp):
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


@lowering_rule(TransformerOp)
def lower_transformer(op: TransformerOp, ctx) -> PhysicalOp | None:
    """Lower a ``TransformerOp`` to the matching abstract physical transformer.

    Only estimators with a dedicated physical op are lowered; anything else
    returns ``None`` and stays a logical ``TransformerOp``.
    """
    if isinstance(op.original_estimator, _SkrubTableVectorizer):
        return TableVectorizerOp(
            estimator=op.estimator, y=op.y, cols=op.cols, how=op.how,
            allow_reject=op.allow_reject, unsupervised=op.unsupervised,
            kwargs=op.kwargs, param_refs=op.param_refs,
        )
    if isinstance(op.original_estimator, _SkrubStringEncoder):
        return StringEncoderOp(
            estimator=op.estimator, y=op.y, cols=op.cols, how=op.how,
            allow_reject=op.allow_reject, unsupervised=op.unsupervised,
            kwargs=op.kwargs, param_refs=op.param_refs,
        )
    return None
