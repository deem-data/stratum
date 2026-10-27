"""pandas-spelled calls translated to polars.

The logical ops and expressions keep pandas' names for string methods, datetime
fields and column methods (extraction reads pandas code). This module is where
the polars impls and the polars expression compiler turn those names and
arguments into polars calls, and where they decide that a call has no exact
polars translation, so the plan can bind pandas instead.
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl

from stratum.optimizer.logical._base import OperandRef

from stratum.optimizer.logical._column_methods import normalize_dtype


# --- .str ---------------------------------------------------------------------
#
# Each pandas ``.str`` method polars can run exactly has a translator: it takes
# the call's params (pandas' parameter names) and returns the function that runs
# the call on a polars Series or expression, or raises ``_NoTranslation`` with the
# reason. A method without one has no exact translation, even where polars has a
# method of that name: ``find`` misses to -1 in pandas but to null in polars,
# ``join`` joins each row's list in pandas but every row in polars, ``extract``
# fans out into columns, and ``encode``/``decode`` take other encodings. Shared
# by the ``StringMethodOp`` impl and the expression compiler's ``StrExpr``.


class _NoTranslation(Exception):
    """A pandas ``.str`` call polars cannot run exactly; the message says why."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _NoTranslation(reason)


def _only(params: dict, *names: str) -> None:
    extra = sorted(set(params) - set(names))
    _require(not extra, f"options {extra} are not translated")


# A regex construct the Rust regex engine lacks (lookaround, inline groups and
# flags) or that means something else there (a numbered backreference).
_UNPORTABLE_REGEX = re.compile(r"\(\?|\\[1-9]")


def _regex_unsupported(pat: str) -> str | None:
    try:
        re.compile(pat)
    except re.error as exc:
        return f"invalid regex: {exc}"
    if _UNPORTABLE_REGEX.search(pat):
        return "lookarounds, inline groups/flags and backreferences are not portable"
    return None


def _translate_without_options(polars_name: str):
    def translate(params):
        _only(params)
        return lambda obj: getattr(obj.str, polars_name)()
    return translate


def _translate_strip(polars_name: str):
    def translate(params):
        chars = params.get("to_strip")
        _require(chars is None or isinstance(chars, str), "to_strip must be a string")
        return lambda obj: getattr(obj.str, polars_name)(chars)
    return translate


def _translate_affix(polars_name: str):
    """``startswith``/``endswith``: one string; a tuple of them is pandas-only."""
    def translate(params):
        _only(params, "pat")
        pat = params.get("pat")
        _require(isinstance(pat, str), "only a single string translates")
        return lambda obj: getattr(obj.str, polars_name)(pat)
    return translate


def _translate_contains(params):
    _only(params, "pat", "case", "flags", "regex")
    pat, regex = params.get("pat"), params.get("regex", True)
    _require(isinstance(pat, str), "only a string pattern translates")
    _require(params.get("case", True) is True and not params.get("flags", 0),
             "case=/flags= are not translated")
    _require(isinstance(regex, bool), f"regex={regex!r} is not a bool")
    if regex:
        reason = _regex_unsupported(pat)
        _require(reason is None, reason)
    return lambda obj: obj.str.contains(pat, literal=not regex)


def _translate_count(params):
    # pandas counts regex matches.
    _only(params, "pat", "flags")
    pat = params.get("pat")
    _require(isinstance(pat, str), "only a string pattern translates")
    _require(not params.get("flags", 0), "flags= are not translated")
    reason = _regex_unsupported(pat)
    _require(reason is None, reason)
    return lambda obj: obj.str.count_matches(pat)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _translate_slice(params):
    """pandas slices ``[start:stop]``; polars takes an offset and a length."""
    start, stop, step = (params.get(name) for name in ("start", "stop", "step"))
    _require(step in (None, 1), "a step is not translated")
    _require(all(v is None or _is_int(v) for v in (start, stop)), "the bounds must be integers")
    offset = 0 if start is None else start
    if stop is None:
        return lambda obj: obj.str.slice(offset)
    # With a negative bound the length depends on each string's.
    _require(offset >= 0 and stop >= 0, "a negative bound next to a stop is not translated")
    return lambda obj: obj.str.slice(offset, max(stop - offset, 0))


def _translate_zfill(params):
    width = params.get("width")
    _require(_is_int(width), "the width must be an integer")
    return lambda obj: obj.str.zfill(width)


def _translate_normalize(params):
    form = params.get("form")
    _require(form in ("NFC", "NFKC", "NFD", "NFKD"), f"form={form!r} is not a Unicode form")
    return lambda obj: obj.str.normalize(form)


def _translate_split(params):
    """A literal separator into a list per row; ``expand`` and ``n`` are pandas-only."""
    _only(params, "pat", "n", "expand", "regex")
    pat, regex = params.get("pat"), params.get("regex")
    _require(isinstance(pat, str) and pat != "", "only a non-empty string separator translates")
    _require(params.get("n", -1) in (-1, 0, None) and params.get("expand", False) is False,
             "n=/expand= are not translated")
    # pandas reads a longer pattern as a regex unless told otherwise.
    _require(regex is False or regex is None and len(pat) == 1,
             "a regex separator is not translated")
    return lambda obj: obj.str.split(pat)


def polars_str_replace_unsupported(pat, repl=None, n=-1, case=None, flags=0,
                                   regex=False) -> str | None:
    """Why a pandas ``str.replace`` call has no exact polars translation, or None.

    Only literal string patterns and replacements translate. Refused:
    compiled patterns and callable replacements (pandas-only), ``case``/``flags``,
    an empty pattern, lookarounds and backreferences (unsupported by Rust regex),
    and a regex replacement containing ``\\`` or ``$`` (the two engines spell
    group references differently).

    Accepted divergence: on an *object*-dtype column, Python's ``$`` also
    matches before a trailing newline, Rust's does not. pandas 3's default
    ``str`` dtype already uses pyarrow's RE2, whose ``$`` is end-of-text like
    Rust's, so a pipeline reading strings from files sees no difference.
    """
    if not isinstance(pat, str) or not isinstance(repl, str):
        return "only string patterns and replacements translate"
    if case not in (None, True) or flags:
        return "case=/flags= are not translated"
    if not _is_int(n):
        return f"n={n!r} is not an integer"
    if regex:
        if pat == "":
            return "an empty regex pattern is not translated"
        reason = _regex_unsupported(pat)
        if reason is not None:
            return reason
        if "\\" in repl or "$" in repl:
            return "group references in the replacement are not portable"
    return None


def _translate_replace(params):
    reason = polars_str_replace_unsupported(**params)
    _require(reason is None, reason)
    return lambda obj: _polars_str_replace(obj, **params)


def _polars_str_replace(obj, pat, repl=None, n=-1, case=None, flags=0, regex=False):
    if n == 0:  # pandas makes no replacement for n=0
        return obj
    if n < 0:  # pandas replaces every match for any negative n
        return obj.str.replace_all(pat, repl, literal=not regex)
    return obj.str.replace(pat, repl, literal=not regex, n=n)


# pandas ``.str`` method -> translator of its params.
_STR_TRANSLATORS = {
    "lower": _translate_without_options("to_lowercase"),
    "upper": _translate_without_options("to_uppercase"),
    "len": _translate_without_options("len_chars"),
    "strip": _translate_strip("strip_chars"),
    "lstrip": _translate_strip("strip_chars_start"),
    "rstrip": _translate_strip("strip_chars_end"),
    "startswith": _translate_affix("starts_with"),
    "endswith": _translate_affix("ends_with"),
    "contains": _translate_contains,
    "count": _translate_count,
    "replace": _translate_replace,
    "slice": _translate_slice,
    "zfill": _translate_zfill,
    "normalize": _translate_normalize,
    "split": _translate_split,
}


def polars_str_method_unsupported(method: str, params=None) -> str | None:
    """Why a pandas ``.str.<method>`` call has no exact polars translation, or None.

    ``params`` left ``None`` checks the method alone.
    """
    translate = _STR_TRANSLATORS.get(method)
    if translate is None:
        return "no exact polars translation"
    if params is not None:
        try:
            translate(params)
        except _NoTranslation as exc:
            return str(exc)
    return None


def polars_string_method(obj, method: str, params=None):
    """Run a pandas-spelled ``.str.<method>`` call on a polars Series or expression.

    ``params`` are the call's arguments by pandas' parameter names.
    """
    try:
        run = _STR_TRANSLATORS[method](params or {})
    except _NoTranslation as exc:
        # FIXME(#227): reached only with plan-valued arguments, which supports()
        # cannot check; fall back to the recorded pandas call instead of failing.
        raise NotImplementedError(
            f"str.{method}({params}) has no exact polars translation: {exc}.") from None
    return run(obj)


# --- .dt ----------------------------------------------------------------------

# pandas `.dt.<attr>` -> polars `.dt.<method>`, for the attributes whose names
# differ between backends. Some also differ in *values*; those are corrected in
# :func:`polars_dt_attribute`.
DT_POLARS_METHODS = {
    "dayofweek": "weekday", "day_of_week": "weekday",
    "dayofyear": "ordinal_day", "day_of_year": "ordinal_day",
    "daysinmonth": "days_in_month",
}


def polars_dt_attribute(obj, attr: str):
    """Read pandas' ``.dt.<attr>`` off a polars Series or expression.

    polars exposes the attributes as methods, some under other names (see
    :data:`DT_POLARS_METHODS`), and counts two of them differently: weekdays
    from Monday=1 (ISO 8601, pandas: Monday=0) and nanoseconds past the second
    (pandas: past the microsecond, 0..999). The result carries pandas' values in
    polars' integer dtype.
    """
    if attr == "is_month_end":
        return obj.dt.month_end() == obj
    name = DT_POLARS_METHODS.get(attr, attr)
    result = getattr(obj.dt, name)()
    if name == "weekday":
        return result - 1
    if name == "nanosecond":
        return result % 1000
    return result


def polars_dt_attribute_unsupported(attr: str) -> str | None:
    """Why pandas' ``.dt.<attr>`` has no polars translation, or None."""
    if attr == "is_month_end":
        return None
    name = DT_POLARS_METHODS.get(attr, attr)
    if not hasattr(pl.col("_").dt, name):
        return f"polars has no .dt.{name}"
    return None


# --- pd.to_datetime -------------------------------------------------------------

def polars_to_datetime(obj, format=None, exact=None, cache=True, errors="raise"):
    """Parse a polars string Series or expression the way ``pd.to_datetime`` would.

    The arguments are the fields of ``DatetimeConversionOp``; ``exact=None`` keeps
    polars' default, which matches pandas'.
    """
    if errors not in ("raise", "coerce"):
        raise ValueError(f"errors must be 'raise' or 'coerce', got {errors!r}")
    kwargs = {"format": format, "cache": cache, "strict": errors == "raise"}
    if exact is not None:
        kwargs["exact"] = exact
    # TODO: Support already-datetime and numeric operands natively; the
    # Polars string namespace only accepts string input.
    return obj.str.to_datetime(**kwargs)


# --- column methods -------------------------------------------------------------
#
# Each kernel takes the receiver (a polars Series, frame or expression), the
# method's fields (those of its ``ColumnMethodOp``, resolved), and the receiver's
# dtype when it is known (``None`` for an expression whose dtype could not be
# resolved).

def _polars_astype(obj, dtype, *, obj_dtype=None):
    return obj.cast(normalize_dtype(dtype), strict=True)


def _polars_fillna(obj, value, *, obj_dtype=None):
    # FIXME(#216): pandas fills NaN as well as null, polars needs both calls, and the
    # fill_nan half is skipped whenever the dtype is unknown. A pl.Expr carries
    # no dtype, and `_expressions._polars_expr_dtype` only resolves a bare Col or
    # an astype, so any derived operand ((a / b).fillna(0), a chained
    # clip/where/fillna) silently keeps its NaNs here while pandas replaces them.
    if isinstance(obj, pl.DataFrame):
        return _polars_frame_fillna(obj, value)
    result = obj.fill_null(value)
    actual_dtype = getattr(obj, "dtype", None) or obj_dtype
    if actual_dtype is not None and actual_dtype.is_float():
        result = result.fill_nan(value)
    return result


def _fill_value_fits(value, dtype) -> bool:
    """Whether filling ``value`` keeps the column's kind, as pandas does then."""
    if isinstance(value, bool):
        return dtype == pl.Boolean
    if isinstance(value, (int, float, np.number)):
        return dtype.is_numeric()
    if isinstance(value, str):
        return dtype in (pl.String, pl.Categorical)
    return False


def polars_frame_fillna_unsupported(values: dict, schema) -> str | None:
    """Why ``DataFrame.fillna(values)`` has no exact polars translation on a frame
    with ``schema``, or ``None``.

    pandas upcasts a column to object when the fill value is of another kind,
    which polars cannot do. Only the columns whose dtype ``schema`` knows can be
    checked; the others are checked when the call runs.
    """
    if not isinstance(values, dict) or schema is None:
        return None
    for name, value in values.items():
        dtype = schema.get(name)
        if (dtype is not None and dtype != pl.Unknown and not isinstance(value, OperandRef)
                and not _fill_value_fits(value, dtype)):
            return (f"fillna({{{name!r}: {value!r}}}) on a {dtype} column: pandas would "
                    f"mix types in an object column")
    return None


def _polars_frame_fillna(frame, values: dict):
    columns = []
    for name, value in values.items():
        if name not in frame.columns:  # pandas ignores unknown columns
            continue
        dtype = frame.schema[name]
        if not _fill_value_fits(value, dtype):
            # FIXME(#227): reached only when the schema did not know this
            # column's dtype at plan time; fall back to pandas instead of failing.
            raise NotImplementedError(
                f"fillna({{{name!r}: {value!r}}}) on a {dtype} column has no exact "
                f"polars translation: pandas would mix types in an object column.")
        columns.append(_polars_fillna(pl.col(name), value, obj_dtype=dtype).alias(name))
    return frame.with_columns(columns) if columns else frame


def _polars_clip(obj, lower, upper, *, obj_dtype=None):
    return obj.clip(lower_bound=lower, upper_bound=upper)


def _polars_where(obj, cond, other, *, obj_dtype=None):
    if isinstance(obj, pl.Series):
        if not isinstance(other, pl.Series):
            # FIXME(#216): two problems on this line. Pinning `other` to the column
            # dtype raises where pandas promotes to a supertype, so an int
            # column's .where(cond, 1.5) fails here and returns floats in
            # pandas; and the list comprehension materialises one Python object
            # per row, where pl.repeat(other, len(obj), eager=True) would not.
            # The pl.Expr branch below has neither problem.
            other = pl.Series([other] * len(obj), dtype=obj.dtype)
        return obj.zip_with(cond, other)
    return pl.when(cond).then(obj).otherwise(other)


def _polars_isin(obj, values, *, obj_dtype=None):
    values = list(values) if isinstance(values, (set, frozenset)) else values
    return obj.is_in(values)


def _polars_notna(obj, *, obj_dtype=None):
    # FIXME(#216): same unknown-dtype gap as _polars_fillna. Without a dtype the
    # is_not_nan half is skipped, so (a / b).notna() reports True for a NaN
    # that pandas reports as False. _polars_isna has the mirror-image gap.
    result = obj.is_not_null()
    actual_dtype = getattr(obj, "dtype", None) or obj_dtype
    if actual_dtype is not None and actual_dtype.is_float():
        result = result & obj.is_not_nan()
    return result


def _polars_isna(obj, *, obj_dtype=None):
    # pandas counts NaN as missing, polars only null. See _polars_notna's FIXME(#216).
    result = obj.is_null()
    actual_dtype = getattr(obj, "dtype", None) or obj_dtype
    if actual_dtype is not None and actual_dtype.is_float():
        result = result | obj.is_nan()
    return result


def _polars_map(obj, mapping, na_action=None, *, obj_dtype=None):
    # Misses become missing, like pandas (unlike replace(), which keeps them).
    # A missing input misses too, so na_action="ignore" needs nothing extra.
    if {type(v) for v in mapping.values() if v is not None} == {int, float}:
        # pandas gives float64; Polars types the lookup by its first value.
        mapping = {k: None if v is None else float(v) for k, v in mapping.items()}
    return obj.replace_strict(mapping, default=None)


# pandas method name -> kernel, for every ``ColumnMethodOp`` subclass.
_POLARS_COLUMN_METHODS = {
    "astype": _polars_astype,
    "fillna": _polars_fillna,
    "clip": _polars_clip,
    "where": _polars_where,
    "isin": _polars_isin,
    "isna": _polars_isna,
    "notna": _polars_notna,
    "map": _polars_map,
}


def polars_column_method(method: str, obj, fields: dict, obj_dtype=None):
    """Run the pandas column method ``method`` with its ``fields`` (by name) on a
    polars Series, frame or expression."""
    return _POLARS_COLUMN_METHODS[method](obj, **fields, obj_dtype=obj_dtype)
