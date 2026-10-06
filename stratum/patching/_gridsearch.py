from skrub._data_ops._skrub_namespace import SkrubNamespace
from stratum._config import FLAGS
from stratum._api import grid_search as stratum_grid_search
from stratum.optimizer.logical._scoring import resolve_declared_scoring

# Store reference to original method before patching
_original_make_grid_search = SkrubNamespace.make_grid_search


def _stratum_make_grid_search(self, *, fitted=False, keep_subsampling=False, **kwargs):
    """Stratum adapter for skrub's make_grid_search method.
    
    When scheduler mode is enabled, uses Stratum's optimized grid search.
    Otherwise, falls back to the original skrub implementation.
    """
    # Note: We extract instead of pop to avoid mutating kwargs
    scoring = kwargs.get("scoring", None)
    # skrub's own default, scoring=None, means the estimator's `score`. Stratum refuses
    # that for a search (ADR 0004), but refusing it here would break the drop-in contract
    # (ADR 0002), so the call goes to unpatched skrub unchanged instead. The exception is
    # a metric the DataOp declares with `.skb.with_scoring()`, which skrub then uses.
    if FLAGS.scheduler and (scoring is not None or _declares_usable_scoring(self._data_op)):
        # Use Stratum's scheduler-based grid search
        return stratum_grid_search(
            dag=self._data_op,
            cv=kwargs.get("cv", None),
            scoring=scoring,
            return_predictions=kwargs.get("return_predictions", False),
        )
    # Fall back to original implementation
    return _original_make_grid_search(self, fitted=fitted, keep_subsampling=keep_subsampling, **kwargs)


def _declares_usable_scoring(data_op) -> bool:
    """Whether Stratum can score with the metric ``data_op`` declares, if any."""
    try:
        resolve_declared_scoring(data_op)
    except ValueError:
        return False
    return True


# This will be used by the patching system to replace the method
make_grid_search = _stratum_make_grid_search