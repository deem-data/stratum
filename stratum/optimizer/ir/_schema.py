"""Schema algebra for output-schema propagation.

A *schema* is a :class:`polars.Schema` (an ordered mapping ``column name ->
dtype``). ``None`` is the *unknown* schema: it models the fallback used when an
op cannot determine its output (e.g. a UDF), and every helper here propagates it
-- any operation on an unknown schema is unknown.

For a single column (an op whose ``output_type`` is ``SERIES``) we still use a
one-entry schema, keyed by the column name, so column tracking is uniform across
frames and series.
"""
from __future__ import annotations

import polars as pl

# dtype used for columns whose name is known but whose type we cannot determine
# (e.g. a freshly assigned column built from an arbitrary expression).
UNKNOWN_DTYPE = pl.Unknown


def is_known(schema) -> bool:
    """True if ``schema`` carries column information (i.e. is not the unknown schema)."""
    return schema is not None


def as_column_list(columns) -> list[str] | None:
    """Normalize a ``str``/list/tuple of column labels to a list of names.

    Returns ``None`` when the labels are not statically known string column names
    (e.g. an :class:`OperandRef`, a slice, or anything non-string), which forces
    the caller to fall back to the unknown schema.
    """
    if isinstance(columns, str):
        return [columns]
    if isinstance(columns, (list, tuple)):
        if all(isinstance(c, str) for c in columns):
            return list(columns)
    return None


def drop_columns(schema, columns) -> pl.Schema | None:
    """Input schema with ``columns`` removed; unknown if names aren't known."""
    names = as_column_list(columns)
    if not is_known(schema) or names is None:
        return None
    drop = set(names)
    return pl.Schema({name: dt for name, dt in schema.items() if name not in drop})


def select_columns(schema, columns) -> pl.Schema | None:
    """Sub-schema holding only ``columns``, in the requested order.

    Unknown if the input schema is unknown, the labels aren't known names, or a
    requested column is absent from the input schema.
    """
    names = as_column_list(columns)
    if not is_known(schema) or names is None:
        return None
    out: dict = {}
    for name in names:
        if name not in schema:
            return None
        out[name] = schema[name]
    return pl.Schema(out)


def add_columns(schema, names, dtype=UNKNOWN_DTYPE) -> pl.Schema | None:
    """Input schema extended with ``names`` (replacing any that already exist).

    New/overwritten columns get ``dtype`` (``Unknown`` by default, since the
    value usually comes from an arbitrary expression).
    """
    if not is_known(schema) or names is None:
        return None
    out = dict(schema)
    for name in names:
        out[name] = dtype
    return pl.Schema(out)


def retype_columns(schema, dtype=UNKNOWN_DTYPE) -> pl.Schema | None:
    """Input schema with the same columns but every dtype replaced by ``dtype``.

    Models an accessor/elementwise projection (e.g. ``.dt.year``) that preserves
    the column names but produces a new dtype; callers pass ``UNKNOWN_DTYPE``
    (the default) when that resulting dtype can't be inferred."""
    if not is_known(schema):
        return None
    return pl.Schema({name: dtype for name in schema})


def rename_columns(schema, mapping) -> pl.Schema | None:
    """Input schema with column names remapped through ``mapping`` (name->name)."""
    if not is_known(schema) or not isinstance(mapping, dict):
        return None
    return pl.Schema({mapping.get(name, name): dt for name, dt in schema.items()})


def union_columns(schemas) -> pl.Schema | None:
    """Left-to-right union of several schemas (later dtypes win on a name clash).

    Models a column-wise/relaxed concat: the result holds every column that
    appears in any input. Unknown if any input schema is unknown.
    """
    out: dict = {}
    for schema in schemas:
        if not is_known(schema):
            return None
        out.update(schema)
    return pl.Schema(out)


def merge_join_schemas(left, right, keys, suffixes) -> pl.Schema | None:
    """Schema of ``left`` merged with ``right`` the way pandas ``merge`` would.

    Columns present on both sides that are *not* shared join keys get the
    ``suffixes`` appended (left first, right second); shared join keys collapse to
    a single column. Unknown if either side is unknown.
    """
    if not is_known(left) or not is_known(right):
        return None
    lsuffix, rsuffix = suffixes
    keys = set(keys or ())
    overlap = (set(left) & set(right)) - keys
    out: dict = {}
    for name, dt in left.items():
        out[f"{name}{lsuffix}" if name in overlap else name] = dt
    for name, dt in right.items():
        if name in keys:
            continue
        out[f"{name}{rsuffix}" if name in overlap else name] = dt
    return pl.Schema(out)


def aggregate_schema(schema, grouping_keys, aggregations, as_index) -> pl.Schema | None:
    """Schema of a pandas ``groupby(grouping_keys).agg(aggregations)``.

    Only the dict-spec form with scalar aggregations is statically known: its
    output columns are exactly the dict keys (dtypes left unknown, since they
    depend on the aggregation function -- ``count``->int, ``mean``->float, ...).
    Grouping keys are part of the pandas index by default and only become output
    columns when ``as_index`` is ``False``. Any other spec is unknown: a bare
    function name applies to every (numeric) non-grouping column, and a list spec
    -- including a list value inside the dict -- produces MultiIndex columns we
    can't represent in a flat schema.
    """
    keys = as_column_list(grouping_keys)
    if not is_known(schema) or keys is None or not isinstance(aggregations, dict):
        return None
    out: dict = {}
    if as_index is False:
        for key in keys:
            if key in schema:
                out[key] = schema[key]
    for col, func in aggregations.items():
        if not isinstance(col, str) or isinstance(func, (list, tuple)):
            return None
        out[col] = UNKNOWN_DTYPE
    return pl.Schema(out)
