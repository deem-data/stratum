from skrub._data_ops._skrub_namespace import SkrubNamespace
from stratum._config import FLAGS
from stratum.runtime import grid_search as stratum_grid_search

# Store reference to original method before patching
_original_make_grid_search = SkrubNamespace.make_grid_search


def _stratum_make_grid_search(self, *, fitted=False, keep_subsampling=False, **kwargs):
    """Stratum adapter for skrub's make_grid_search method.
    
    When scheduler mode is enabled, uses Stratum's optimized grid search.
    Otherwise, falls back to the original skrub implementation.
    """
    if FLAGS.scheduler:
        # Use Stratum's scheduler-based grid search
        # Extract kwargs that are relevant for grid_search
        # Note: We extract instead of pop to avoid mutating kwargs
        cv = kwargs.get("cv", None)
        scoring = kwargs.get("scoring", None)
        return_predictions = kwargs.get("return_predictions", False)
        show_stats = kwargs.get("show_stats", False)
        
        # Get the DataOp from the namespace instance
        dag = self._data_op
        # Call Stratum's grid_search
        return stratum_grid_search(
            dag=dag,
            cv=cv,
            scoring=scoring,
            return_predictions=return_predictions,
            show_stats=show_stats
        )
    else:
        # Fall back to original implementation
        return _original_make_grid_search(self, fitted=fitted, keep_subsampling=keep_subsampling, **kwargs)


# This will be used by the patching system to replace the method
make_grid_search = _stratum_make_grid_search