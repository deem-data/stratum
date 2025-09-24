import numbers
import os
import threading
from contextlib import contextmanager

import numpy as np

from ._reporting import _patching


def _parse_env_bool(env_variable_name, default):
    value = os.getenv(env_variable_name, default)
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        raise ValueError(
            f"{env_variable_name!r} must be either 'True' or 'False', got {value}."
        )


_global_config = {
    "use_table_report": _parse_env_bool("SKB_USE_TABLE_REPORT", False),
    "use_table_report_data_ops": _parse_env_bool("SKB_USE_TABLE_REPORT_DATA_OPS", True),
    "max_plot_columns": int(os.environ.get("SKB_MAX_PLOT_COLUMNS", 30)),
    "max_association_columns": int(os.environ.get("SKB_MAX_ASSOCIATION_COLUMNS", 30)),
    "subsampling_seed": int(os.environ.get("SKB_SUBSAMPLING_SEED", 0)),
    "enable_subsampling": os.environ.get("SKB_ENABLE_SUBSAMPLING", "default"),
    "float_precision": int(os.environ.get("SKB_FLOAT_PRECISION", 3)),
    "cardinality_threshold": int(os.environ.get("SKB_CARDINALITY_THRESHOLD", 40)),
}
_threadlocal = threading.local()


def _get_threadlocal_config():
    """Return a thread-local copy of the global configuration.

    This is used to ensure that each thread has its own configuration
    without affecting the global configuration.
    """
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global skrub configuration.
    set_config : Set global skrub configuration.

    Examples
    --------
    >>> import sklearn
    >>> config = sklearn.get_config()
    >>> config.keys()
    dict_keys([...])
    """
    return _get_threadlocal_config().copy()


def _apply_external_patches(config):
    if config["use_table_report"]:
        _patching._patch_display(
            max_plot_columns=config["max_plot_columns"],
            max_association_columns=config["max_plot_columns"],
        )
    else:
        # No-op if dispatch haven't been previously enabled
        _patching._unpatch_display()


def set_config(
    use_table_report=None,
    use_table_report_data_ops=None,
    max_plot_columns=None,
    max_association_columns=None,
    subsampling_seed=None,
    enable_subsampling=None,
    float_precision=None,
    cardinality_threshold=None,
):
    """Set global skrub configuration.

    Parameters
    ----------
    use_table_report : bool, default=None
        The type of display used for dataframes. Default is ``True``.

        - If ``True``, replace the default DataFrame HTML displays with
          :class:`~skrub.TableReport`.
        - If ``False``, the original Pandas or Polars dataframe HTML representation
          will be used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT``
        environment variable.

    use_table_report_data_ops : bool, default=None
        The type of HTML representation used for the dataframes preview in skrub
        DataOps. Default is ``False``.

        - If ``True``, :class:`~skrub.TableReport` will be used.
        - If ``False``, the original Pandas or Polars dataframe display will be
          used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT_DATA_OPS``
        environment variable.

    max_plot_columns : int, default=None
        Set the ``max_plot_columns`` argument of :class:`~skrub.TableReport`.
        Default is 30. If "all", all columns will be plotted.

        This configuration can also be set with the ``SKB_MAX_PLOT_COLUMNS``
        environment variable.

    max_association_columns : int, default=None
        Set the ``max_association_columns`` argument of :class:`~skrub.TableReport`.
        Default is 30. If "all", all columns will be plotted.

        This configuration can also be set with the ``SKB_MAX_ASSOCIATION_COLUMNS``
        environment variable.

    subsampling_seed : int, default=None
        Set the random seed of subsampling in skrub DataOps
        :func:`skrub.DataOp.skb.subsample`, when ``how="random"`` is passed.

        This configuration can also be set with the ``SKB_SUBSAMPLING_SEED`` environment
        variable.

    enable_subsampling : {'default', 'disable', 'force'}, default=None
        Control the activation of subsampling in skrub DataOps
        :func:`skrub.DataOp.skb.subsample`. Default is ``"default"``.

        - If ``"default"``, the behavior of :func:`skrub.DataOp.skb.subsample` is used.
        - If ``"disable"``, subsampling is never used, so ``skb.subsample`` becomes a
          no-op.
        - If ``"force"``, subsampling is used in all DataOps evaluation modes
          (:func:`~skrub.DataOp.skb.eval`, fit_transform, etc.).

        This configuration can also be set with the ``SKB_ENABLE_SUBSAMPLING``
        environment variable.

    float_precision : int, default=3
        Control the number of significant digits shown when formatting floats.
        Applies overall precision rather than fixed decimal places. Default is 3.

        This configuration can also be set with the ``SKB_FLOAT_PRECISION``
        environment variable.

    cardinality_threshold : int, default=40
        Set the ``cardinality_threshold`` argument of :class:`~skrub.TableVectorizer`.
        Control the threshold value used to warn user if they have
        high cardinality columns in there dataset.

        This configuration can also be set with the ``SKB_CARDINALITY_THRESHOLD``
        environment variable.

    See Also
    --------
    get_config : Retrieve current values for global configuration.
    config_context : Context manager for global skrub configuration.

    Examples
    --------
    >>> from skrub import set_config
    >>> set_config(use_table_report=True)  # doctest: +SKIP
    """
    local_config = _get_threadlocal_config()
    if use_table_report is not None:
        if not isinstance(use_table_report, bool):
            raise ValueError(
                f"'use_table_report' must be a boolean, got {use_table_report!r}."
            )
        local_config["use_table_report"] = use_table_report

    if use_table_report_data_ops is not None:
        if not isinstance(use_table_report_data_ops, bool):
            raise ValueError(
                "'use_table_report_data_ops' must be a boolean, got "
                f"{use_table_report_data_ops!r}."
            )
        local_config["use_table_report_data_ops"] = use_table_report_data_ops

    if max_plot_columns is not None:
        if not isinstance(max_plot_columns, numbers.Real) and max_plot_columns != "all":
            raise ValueError(
                "'max_plot_columns' must be a number or 'all', got "
                f"{type(max_plot_columns)!r}"
            )
        local_config["max_plot_columns"] = max_plot_columns

    if max_association_columns is not None:
        if (
            not isinstance(max_association_columns, numbers.Real)
            and max_plot_columns != "all"
        ):
            raise ValueError(
                "'max_association_columns' must be a number or 'all', got "
                f"{type(max_association_columns)!r}"
            )
        local_config["max_association_columns"] = max_association_columns

    if subsampling_seed is not None:
        np.random.RandomState(subsampling_seed)  # check seed
        local_config["subsampling_seed"] = subsampling_seed

    if enable_subsampling is not None:
        if enable_subsampling not in (options := ("default", "force", "disable")):
            raise ValueError(
                f"'enable_subsampling' options are {options!r}, got "
                f"{enable_subsampling!r}"
            )
        local_config["enable_subsampling"] = enable_subsampling

    if float_precision is not None:
        if not isinstance(float_precision, numbers.Integral) or float_precision <= 0:
            raise ValueError(
                f"'float_precision' must be a positive integer, got {float_precision!r}"
            )
        local_config["float_precision"] = float_precision

    if cardinality_threshold is not None:
        if (
            not isinstance(cardinality_threshold, numbers.Integral)
            or cardinality_threshold < 0
        ):
            raise ValueError(
                "'cardinality_threshold' must be a positive"
                f"integer, got {cardinality_threshold!r}"
            )

    _apply_external_patches(local_config)


@contextmanager
def config_context(
    *,
    use_table_report=None,
    use_table_report_data_ops=None,
    max_plot_columns=None,
    max_association_columns=None,
    subsampling_seed=None,
    enable_subsampling=None,
    float_precision=None,
    cardinality_threshold=None,
):
    """Context manager for global skrub configuration.

    Parameters
    ----------
    use_table_report : bool, default=None
        The type of display used for dataframes. Default is ``False``.

        - If ``True``, replace the default DataFrame HTML displays with
          :class:`~skrub.TableReport`.
        - If ``False``, the original Pandas or Polars dataframe HTML representation
          will be used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT``
        environment variable.

    use_table_report_data_ops : bool, default=None
        The type of HTML representation used for the dataframes preview in skrub
        DataOps. Default is ``True``.

        - If ``True``, :class:`~skrub.TableReport` will be used.
        - If ``False``, the original Pandas or Polars dataframe display will be
          used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT_DATA_OPS``
        environment variable.

    max_plot_columns : int, default=None
        Set the ``max_plot_columns`` argument of :class:`~skrub.TableReport`.
        Default is 30. If "all", all columns will be plotted.

        This configuration can also be set with the ``SKB_MAX_PLOT_COLUMNS``
        environment variable.

    max_association_columns : int, default=None
        Set the ``max_association_columns`` argument of :class:`~skrub.TableReport`.
        Default is 30. If "all", all columns will be plotted.

        This configuration can also be set with the ``SKB_MAX_ASSOCIATION_COLUMNS``
        environment variable.

    subsampling_seed : int, default=None
        Set the random seed of subsampling in skrub DataOps
        :func:`skrub.DataOp.skb.subsample`, when ``how="random"`` is passed.

        This configuration can also be set with the ``SKB_SUBSAMPLING_SEED`` environment
        variable.

    enable_subsampling : {'default', 'disable', 'force'}, default=None
        Control the activation of subsampling in skrub DataOps
        :func:`skrub.DataOp.skb.subsample`. Default is ``"default"``.

        - If ``"default"``, the behavior of :func:`skrub.DataOp.skb.subsample` is used.
        - If ``"disable"``, subsampling is never used, so ``skb.subsample`` becomes a
          no-op.
        - If ``"force"``, subsampling is used in all DataOps evaluation modes
          (preview, fit_transform, etc.).

        This configuration can also be set with the ``SKB_ENABLE_SUBSAMPLING``
        environment variable.

    float_precision : int, default=3
        Control the number of significant digits shown when formatting floats.
        Applies overall precision rather than fixed decimal places. Default is 3.

        This configuration can also be set with the ``SKB_FLOAT_PRECISION``
        environment variable.

    cardinality_threshold : int, default=40
        Set the ``cardinality_threshold`` argument of :class:`~skrub.TableVectorizer`.
        Control the threshold value used to warn user if they have
        high cardinality columns in there dataset.

        This configuration can also be set with the ``SKB_CARDINALITY_THRESHOLD``
        environment variable.

    Yields
    ------
    None.

    See Also
    --------
    get_config : Retrieve current values for global configuration.
    set_config : Set global skrub configuration.

    Examples
    --------
    >>> import skrub
    >>> with skrub.config_context(max_plot_columns=1):
    ...     ...  # doctest: +SKIP
    """
    original_config = get_config()
    set_config(
        use_table_report=use_table_report,
        use_table_report_data_ops=use_table_report_data_ops,
        max_plot_columns=max_plot_columns,
        max_association_columns=max_association_columns,
        subsampling_seed=subsampling_seed,
        enable_subsampling=enable_subsampling,
        float_precision=float_precision,
        cardinality_threshold=cardinality_threshold,
    )

    try:
        yield
    finally:
        set_config(**original_config)


# ------- Rust backend (stratum) config group ---------------

def _rust_env_bool(name, default=False):
    val = os.getenv(name)
    if val is None:
        return bool(default)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    # fall back to existing convention if someone sets "True"/"False"
    return s == "true"

def _rust_env_int(name, default):
    v = os.getenv(name)
    return int(v) if v is not None else int(default)

def _rust_env_choice(name, default, choices):
    v = os.getenv(name)
    x = (v if v is not None else default).strip().lower()
    if x not in choices:
        raise ValueError(f"{name!r} must be one of {sorted(choices)}, got {x!r}")
    return x

# Thread-local storage, similar to _get_threadlocal_config()
if not hasattr(_threadlocal, "rust_config"):
    _threadlocal.rust_config = {
        "enable_rust": _rust_env_bool("SKRUB_RUST", False),
        "num_threads": _rust_env_int("SKRUB_RUST_THREADS", 0),           # 0 => global threadpool
        "debug_timing": _rust_env_bool("SKRUB_RUST_DEBUG_TIMING", False),
        "allow_monkeypatch": _rust_env_bool("SKRUB_RUST_ALLOW_MONKEYPATCH", True),
    }

def get_rust_config():
    # Return a shallow copy of the current Rust backend config
    return _threadlocal.rust_config.copy()

def set_rust_config(**kwargs):
    """ Update Rust backend related knobs.

    Parameter:
    -----------

        enable_rust: bool, default false
            Enable/disable rust backend. It is a feature flag for the Rust backend.

        num_threads: int >= 0 (0 lets backend decide), default 0
            Set the number of threads for the multithreaded rust operations.

        debug_timing: bool, default false
            Print the timing in standard output.

        allow_monkeypatch: bool, default true
            Allows disabling runtime monkey-patch in sensitive contexts. This is a soft
            kill-switch for disabling all non-sklearn backends, even if their flags are set.
    """
    rc = _threadlocal.rust_config.copy()
    for k, v in kwargs.items():
        if k not in rc:
            raise KeyError(f"Unknown rust option {k!r}. Valid keys: {sorted(rc)}")
        if k in {"enable_rust", "debug_timing", "allow_monkeypatch"}:
            if not isinstance(v, bool):
                raise ValueError(f"rust[{k}] must be bool, got {v!r}")
        elif k in {"num_threads"}:
            if not (isinstance(v, int) and v >= 0):
                raise ValueError(f"rust[{k}] must be an int >= 0, got {v!r}")
        rc[k] = v
        _threadlocal.rust_config = rc

from contextlib import contextmanager as _ctxmgr  # reuse imported contextmanager

@_ctxmgr
def rust_config_context(**kwargs):
    """ Context manager for Rust backend config.

    Example:
        with rust_config_context(enabled=True, threads=8):
            ...
    """
    _orig = get_rust_config()
    set_rust_config(**kwargs)
    try:
        yield
    finally:
        set_rust_config(**_orig)
# ---------------------------------------------------------------------------



# Apply patching set by environment variables. Without it, setting SKB_USE_TABLE_REPORT
# or SKB_USE_TABLE_REPORT_DATA_OPS would not have an effect.
_apply_external_patches(get_config())
