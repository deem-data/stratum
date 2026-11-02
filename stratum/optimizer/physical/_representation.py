"""Frame library of plan edges (#226).

Every frame-typed edge carries a physical property, its frame library
(pandas or polars), fixed by the producer's implementation. Contracts:

* bound ``PhysicalOp``: ``accepts_input_lib`` / ``output_frame_lib``;
* recorded frontend calls (``MethodCallOp``, ``CallOp``, ``GetAttrOp``): pandas;
* other logical ops: pass the input library through.
"""
from __future__ import annotations

from enum import Enum

from stratum.optimizer.logical._base import FRAME_TYPES, IRNode, OutputType


class FrameLib(str, Enum):
    PANDAS = "pandas"
    POLARS = "polars"

    def __str__(self) -> str:
        return self.value

    @property
    def other(self) -> FrameLib:
        return FrameLib.POLARS if self is FrameLib.PANDAS else FrameLib.PANDAS


#: Backends bound to one frame library.
BACKEND_FRAME_LIB = {"pandas": FrameLib.PANDAS, "polars": FrameLib.POLARS}


class RepresentationError(NotImplementedError):
    """An edge carries a frame library its consumer cannot read."""


def frame_lib(op: IRNode) -> FrameLib | None:
    """Output library of ``op``; ``None`` for non-frame or unknown outputs."""
    return getattr(op, "_frame_lib", None)


def input_libs(op: IRNode) -> list[FrameLib | None]:
    return [frame_lib(i) for i in op.inputs]


def _is_opaque_call(op: IRNode) -> bool:
    # Lazy import: avoids an import cycle.
    from stratum.optimizer.logical._ops import CallOp, GetAttrOp, MethodCallOp
    return isinstance(op, (MethodCallOp, CallOp, GetAttrOp))


def _pandas_only_operand(op: IRNode) -> bool:
    """NumericOp whose constant polars cannot represent (e.g. ``DateOffset``)."""
    from stratum.optimizer.logical._numeric_ops import NumericOp
    if not isinstance(op, NumericOp) or op.constant is None:
        return False
    import polars as pl
    try:
        pl.lit(op.constant)
    except Exception:  # noqa: BLE001
        return True
    return False


def _polars_safe_estimators() -> frozenset:
    """Estimators that fit and answer on a polars frame exactly as on the pandas one.

    Each is checked by ``test_an_allowed_estimator_answers_as_on_pandas``. Others
    can differ: skrub's ``ToCategorical`` makes polars categoricals, which
    scikit-learn's HistGradientBoosting does not take as categorical.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    return frozenset({LinearRegression, LogisticRegression, MinMaxScaler, Ridge,
                      SimpleImputer, StandardScaler})


def _reads_pandas_only(impl_class, op: IRNode) -> bool:
    """An estimator run as given (the ``sklearn-skrub`` backend, or unbound) reads
    its data as the pandas run gives it, unless it is one known to answer the same
    on polars, given data it reads alike from both. Stratum's own kernels
    (``rust``, ``stratum``) read either library."""
    from stratum.optimizer.logical._ops import BaseEstimatorOp
    return (isinstance(op, BaseEstimatorOp)
            and getattr(impl_class, "backend_name", "sklearn-skrub") == "sklearn-skrub"
            and (type(op.original_estimator) not in _polars_safe_estimators()
                 or not _read_alike(op)))


def _read_alike(op: IRNode) -> bool:
    """Whether scikit-learn reads ``op``'s data alike from pandas and polars: each
    input's schema is known and holds only dtypes it converts the same way from
    both. It keeps a pandas frame's float32 columns, but widens a polars frame's
    to float64, and an estimator then computes in the wider type."""
    import polars as pl
    from stratum.optimizer.logical import _schema
    alike = (pl.Float64, pl.Int64, pl.Boolean)
    for k, producer in enumerate(op.inputs):
        if op.reads_state(k) or producer.output_type not in FRAME_TYPES:
            continue
        schema = producer.output_schema
        if not _schema.is_known(schema) or any(dtype not in alike for dtype in schema.values()):
            return False
    return True


def accepts_input(impl_class, op: IRNode, index: int, lib: FrameLib | None) -> bool:
    """Whether ``op`` run by ``impl_class`` reads input ``index`` as ``lib``."""
    if lib is None or op.reads_state(index):
        # A fitted state is read as the object it is.
        return True
    if lib is FrameLib.POLARS and _reads_pandas_only(impl_class, op):
        return False
    if impl_class is not None and hasattr(impl_class, "accepts_input_lib"):
        return impl_class.accepts_input_lib(op, index, lib)
    if _is_opaque_call(op) or _pandas_only_operand(op):
        return lib is FrameLib.PANDAS
    return True


def output_lib(impl_class, op: IRNode, libs: list[FrameLib | None]) -> FrameLib | None:
    """Output library of ``op`` run by ``impl_class``.

    UNKNOWN outputs are treated as frames: a spurious conversion is a no-op,
    a missing one is a runtime error.
    """
    if op.output_type not in FRAME_TYPES and op.output_type is not OutputType.UNKNOWN:
        return None
    if impl_class is not None and hasattr(impl_class, "output_frame_lib"):
        return impl_class.output_frame_lib(op, libs)
    if _is_opaque_call(op) or _pandas_only_operand(op):
        return FrameLib.PANDAS
    return next((lib for lib in libs if lib is not None), None)


def check_frame_representations(ops) -> None:
    """Plan validation: every edge's library is readable by its consumer."""
    problems = []
    for op in ops:
        rejected = [(k, lib) for k, lib in enumerate(input_libs(op))
                    if not accepts_input(type(op), op, k, lib)]
        if rejected:
            problems.append(
                f"{op}: " + "; ".join(f"input {k} ({op.inputs[k]}) is {lib}"
                                      for k, lib in rejected)
                + _refusal_suffix(op))
    if problems:
        raise RepresentationError(
            "Frame library mismatch on plan edges:\n  " + "\n  ".join(problems))


def _refusal_suffix(op: IRNode) -> str:
    reasons = getattr(op, "_refusals", None)
    if not reasons:
        return ""
    return " [" + "; ".join(f"{backend} refused: {why}" for backend, why in reasons) + "]"
