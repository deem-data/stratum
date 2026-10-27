"""The logical frame operators describe an operation; running it is the physical layer's job.

Logical ops and column expressions carry backend-neutral fields only. Each
backend's impl (in ``stratum/optimizer/physical``) reads those fields and runs
them. These checks keep execution code from creeping back into the logical
layer. Estimator ops and the generic and control ops (``_ops.py``,
``_numeric_ops.py``, ``_split_ops.py``, ``_candidate_ops.py``) still run
themselves and are deliberately not covered.
"""
import ast
import importlib
import inspect
from pathlib import Path

import pytest

from stratum.optimizer.logical._column_expr import ColumnExpr
from stratum.optimizer.logical._ops import Op

FRAME_MODULES = [
    "_aggregation_ops", "_column_expr", "_column_methods", "_dataframe_ops",
    "_expr_folding", "_index_ops", "_join_ops", "_map_ops", "_metadata_ops",
    "_projection_ops", "_selection_ops", "_sort_ops", "_source_ops",
]

_EXPR_EXECUTION = {"to_pandas", "to_polars", "to_pandas_query", "polars_unsupported"}


def _module(name):
    return importlib.import_module(f"stratum.optimizer.logical.{name}")


def _own_classes(module, base):
    return [cls for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, base) and cls.__module__ == module.__name__]


@pytest.mark.parametrize("name", FRAME_MODULES)
def test_frame_ops_define_no_process(name):
    offenders = [cls.__name__ for cls in _own_classes(_module(name), Op)
                 if "process" in vars(cls)]
    assert not offenders, f"logical ops that execute themselves: {offenders}"


def test_column_expressions_are_pure_data():
    offenders = {cls.__name__: sorted(_EXPR_EXECUTION & set(vars(cls)))
                 for cls in _own_classes(_module("_column_expr"), ColumnExpr)
                 if _EXPR_EXECUTION & set(vars(cls))}
    assert not offenders, f"expression nodes that evaluate themselves: {offenders}"


@pytest.mark.parametrize("name", FRAME_MODULES)
def test_frame_modules_do_not_import_the_physical_layer(name):
    tree = ast.parse(Path(_module(name).__file__).read_text())
    imported = [node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module]
    imported += [alias.name for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names]
    physical = [m for m in imported if m.startswith("stratum.optimizer.physical")]
    assert not physical, f"{name} imports {physical}"
