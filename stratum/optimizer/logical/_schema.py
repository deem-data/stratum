"""Schema algebra for output-schema propagation.

A *schema* is a :class:`polars.Schema` (an ordered mapping ``column name ->
dtype``). ``None`` is the *unknown* schema: the fallback for an op that cannot
determine its output, and every helper here propagates it -- any operation on an
unknown schema is unknown. A single-column op (``output_type`` ``SERIES``) still
gets a one-entry schema keyed by the column name, so column tracking is uniform
across frames and series.

**The governing invariant:** column names are exact, and a dtype is present only
when it is statically certain -- otherwise it is :data:`UNKNOWN_DTYPE`. Never
widen the set of names to stay safe: return the unknown schema instead. Names are
what the consuming rewrites (predicate pushdown, projection pushup) reason about,
so a wrong name is a correctness bug in every consumer. Hence an operation that
can null-pad a column (an outer join, a ragged union) must drop that column's
dtype: pandas widens ``int64`` to ``float64`` there.

Op families that deliberately keep the unknown default, so that the absence of a
rule reads as intent rather than an oversight:

* ``ApplyUDFOp`` -- an arbitrary user function.
* ``BaseEstimatorOp`` -- an estimator may replace every column (a
  TableVectorizer does), and the column set is only settled at fit time.
* ``ColumnSelectorOp`` -- a deferred skrub selector. Resolvable for name- and
  dtype-based selectors, but value-dependent ones (``cardinality_below``,
  ``has_nulls``, a user ``filter``) would answer from a zero-row frame without
  erroring, i.e. silently wrong. See the TODO on ``resolve_selector_columns``.
* ``MapOp`` (the base) -- each kind opts in by overriding, as ``AssignMapOp`` does.
* ``NumericOp`` -- the numpy/MATRIX world has no columns.
* ``ChoiceOp`` -- a choice combines alternative pipelines and is the last
  operation in the DAG, so no operator downstream needs its schema. Unifying the
  outcomes' schemas would be work with no consumer.

A rule covers an *operation*, not an operation family, wherever the family is not
uniformly column preserving: see ``StringMethodOp.ONE_TO_ONE_METHODS``,
``GetAttrProjectionOp.ELEMENTWISE_ACCESSORS``, ``SelectionOp._drops_rows`` and
``DatetimeConversionOp``, each of which admits only the members that map one
input column to one output column.
"""
from __future__ import annotations

import logging
import operator

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# dtype for a column whose name is known but whose type we cannot determine
# (e.g. a freshly assigned column built from an arbitrary expression).
#
# Two traps when reading one back out of a schema:
# * `pl.Schema` *instantiates* the dtype class, so `schema["a"] is pl.Unknown` is
#   False. Compare dtypes with `==`, never `is` (and never via a set/`in` test).
# * `pl.Unknown.is_numeric()` returns False -- Unknown answers "no" rather than
#   "don't know". Test `== UNKNOWN_DTYPE` before trusting a category predicate.
UNKNOWN_DTYPE = pl.Unknown

# Comparisons produce a boolean mask whatever the operands are.
_COMPARISON_OPS = frozenset({operator.gt, operator.lt, operator.ge, operator.le,
                             operator.eq, operator.ne})
# `&`/`|`/`^`/`~` are *logical* on boolean operands but *bitwise* on integers
# (`int & int` -> Int64, `bool & bool` -> Boolean), so their result dtype follows
# the operands rather than being boolean by construction. Both `inv` and `invert`
# are listed because `operator.inv is operator.invert` is False.
_LOGICAL_OPS = frozenset({operator.and_, operator.or_, operator.xor,
                          operator.invert, operator.inv})


def elementwise_result_dtype(op, operand_dtypes) -> pl.DataType:
    """Result dtype of the elementwise operator ``op``, or ``UNKNOWN_DTYPE``.

    ``operand_dtypes`` is the dtype every operand contributes for one column, or
    ``None`` when some operand's dtype isn't known (a scalar constant, a non-frame
    input). Only two cases are statically certain: a comparison always yields a
    boolean mask, and a logical operator yields a boolean only when *every*
    operand is already boolean (on integers it is bitwise, and yields an integer).
    Everything else depends on the operands in ways this doesn't model
    (``int / int`` is float, ``int ** -1`` is float, mixed operands upcast).
    """
    if op in _COMPARISON_OPS:
        return pl.Boolean
    if (op in _LOGICAL_OPS and operand_dtypes
            and all(dt == pl.Boolean for dt in operand_dtypes)):
        return pl.Boolean
    return UNKNOWN_DTYPE


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


def project_columns(schema, columns) -> pl.Schema | None:
    """Sub-schema holding only ``columns``, in the requested order.

    Unknown if the input schema is unknown, the labels aren't known names, or a
    requested column is absent from the input schema.
    """
    names = as_column_list(columns)
    if not is_known(schema) or names is None:
        return None
    if len(set(names)) != len(names):
        # `df[["a", "a"]]` really does produce two `a` columns, which a Schema
        # cannot express -- narrowing it to one would understate the output.
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


def cast_columns(schema, dtype=UNKNOWN_DTYPE) -> pl.Schema | None:
    """Input schema with the same columns but every dtype replaced by ``dtype``.

    Models an accessor/elementwise projection (e.g. ``.dt.year``) that preserves
    the column names but produces a new dtype.
    """
    if not is_known(schema):
        return None
    return pl.Schema({name: dtype for name in schema})


def rename_columns(schema, mapping) -> pl.Schema | None:
    """Input schema with column names remapped through ``mapping`` (name->name).

    Unknown when two columns would be renamed onto the same label: pandas keeps
    both, and a Schema would silently collapse them to the last one.
    """
    if not is_known(schema) or not isinstance(mapping, dict):
        return None
    out = {mapping.get(name, name): dt for name, dt in schema.items()}
    if len(out) != len(schema):
        return None
    return pl.Schema(out)


def schema_of_frame(frame) -> pl.Schema | None:
    """Schema of an already-materialised frame, or ``None`` if it can't be read.

    A pandas ``object`` column has no column-level element type -- inferring one
    means scanning values, which can be confidently wrong (ints in the early rows,
    strings later) -- so it keeps its name with an ``UNKNOWN_DTYPE``. Anything
    that is not a pandas/polars frame (e.g. the ndarray behind an npy source) has
    no column schema at all.
    """
    if isinstance(frame, pl.DataFrame):
        # polars frames already carry exact dtypes.
        return frame.schema
    if not isinstance(frame, pd.DataFrame):
        return None
    # head(0) converts names + typed dtypes without copying data or risking a
    # mixed-object conversion error. An exotic/extension dtype can still fail.
    try:
        schema = pl.from_pandas(frame.head(0)).schema
        return pl.Schema({
            name: (UNKNOWN_DTYPE if pd.api.types.is_object_dtype(frame[name]) else dt)
            for name, dt in schema.items()
        })
    except Exception:
        logger.debug("Could not derive schema for in-memory frame; falling back to unknown.")
        return None


def concat_schemas(schemas, axis) -> pl.Schema | None:
    """Schema of a pandas ``concat`` of ``schemas`` along ``axis``.

    The names are the left-to-right union in both directions. The two axes are
    different operators, and only the dtype rule differs here:

    * ``axis=0`` is a bag union: a column absent from one operand is null-padded
      for that operand's rows, widening e.g. ``int64`` -> ``float64``. A dtype
      survives only for a column present in *every* operand with the *same* dtype.
    * ``axis=1`` is an index-aligned join: any misalignment null-pads *every*
      column, and the alignment isn't statically knowable, so no dtype survives. A
      name shared by two operands is unknown outright -- pandas keeps both columns
      and a ``Schema`` cannot express a duplicate name.

    Unknown if there are no operands, any operand's schema is unknown, or ``axis``
    is not 0/1.
    """
    if not schemas or any(not is_known(s) for s in schemas) or axis not in (0, 1):
        return None

    names: list[str] = []  # ordered left-to-right union
    for schema in schemas:
        for name in schema:
            if name not in names:
                names.append(name)

    if axis == 1:
        seen: set = set()
        for schema in schemas:
            if seen & set(schema):
                return None  # duplicate name: pandas keeps both columns
            seen |= set(schema)
        return pl.Schema({name: UNKNOWN_DTYPE for name in names})

    def row_concat_dtype(name):
        if not all(name in schema for schema in schemas):
            return UNKNOWN_DTYPE  # null-padded where absent
        dtypes = {schema[name] for schema in schemas}
        return dtypes.pop() if len(dtypes) == 1 else UNKNOWN_DTYPE
    return pl.Schema({name: row_concat_dtype(name) for name in names})


# Join kinds that can leave a side unmatched, mapped to which side's non-key
# columns become nullable. Measured against pandas: a `how="left"` merge widens
# the *right* side's `int64` to `float64` for unmatched rows (and vice versa); an
# outer join widens both. The collapsed join key never gains nulls.
_NULLABLE_JOIN_SIDES = {
    "left": ("right",),
    "right": ("left",),
    "outer": ("left", "right"),
    "full": ("left", "right"),
}


def join_schema(left, right, keys, suffixes, how="inner") -> pl.Schema | None:
    """Schema of ``left`` joined with ``right`` the way pandas ``merge`` would.

    Columns present on both sides that are *not* shared join keys get the
    ``suffixes`` appended (left first, right second); shared join keys collapse to
    a single column. ``how`` decides which dtypes survive: an unmatched row is
    null-padded, which widens an integer column to float, so the nullable side's
    non-key columns are reduced to ``UNKNOWN_DTYPE`` (see
    :data:`_NULLABLE_JOIN_SIDES`).

    ``keys`` are the names that collapse, as the caller worked them out from the
    join's key arguments. Unknown if either side is unknown, if ``keys`` is
    ``None`` (the caller could not determine them), or if two output columns would
    land on the same label -- pandas rejects that at runtime, and a Schema cannot
    express it either way.
    """
    if not is_known(left) or not is_known(right) or keys is None:
        return None
    lsuffix, rsuffix = suffixes
    keys = set(keys)
    overlap = (set(left) & set(right)) - keys
    nullable = _NULLABLE_JOIN_SIDES.get(how, ())

    def dtype_for(side, name, dt):
        if name in keys:
            # A collapsed key never gains nulls, but it *is* the two sides' key
            # columns merged into one, and pandas upcasts when they disagree
            # (int32 + float64 -> float64). So the dtype only survives when both
            # sides already carry it.
            return dt if left.get(name) == right.get(name) else UNKNOWN_DTYPE
        return UNKNOWN_DTYPE if side in nullable else dt

    out: dict = {}

    def put(name, dt):
        """Record one output column; False if that label is already taken."""
        if name in out:
            return False
        out[name] = dt
        return True

    for name, dt in left.items():
        if not put(f"{name}{lsuffix}" if name in overlap else name,
                   dtype_for("left", name, dt)):
            return None
    for name, dt in right.items():
        if name in keys:
            continue
        if not put(f"{name}{rsuffix}" if name in overlap else name,
                   dtype_for("right", name, dt)):
            return None
    return pl.Schema(out)


def aggregate_schema(schema, grouping_keys, output_columns, as_index) -> pl.Schema | None:
    """Schema of a grouped aggregation, given its output column names.

    ``output_columns`` holds one name per aggregation entry, in order, with
    dtypes left unknown (they depend on the aggregation function -- ``count`` ->
    int, ``mean`` -> float). Grouping keys are part of the pandas index by
    default and become output columns only when ``as_index`` is ``False``; only
    then do their names matter. Unknown when the output names are not static
    (``None``: a wildcard entry, or a computed child with no name of its own), or
    when two entries land on the same label, which a Schema cannot express.
    """
    if not is_known(schema) or output_columns is None:
        return None
    out: dict = {}
    if as_index is False:
        keys = as_column_list(grouping_keys)
        if keys is None:
            return None
        for key in keys:
            if key in schema:
                out[key] = schema[key]
    for col in output_columns:
        if not isinstance(col, str) or col in out:
            return None
        out[col] = UNKNOWN_DTYPE
    return pl.Schema(out)
