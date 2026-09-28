"""NYC housing risk plan on a small, generated three-table lake.

The fixture keeps the task's important shapes: three dated PLUTO releases,
missing BBLs resolved from borough/block/lot, categorical event counts, a
point-in-time label, and expanding-window CV. It exercises plan extraction
without requiring the real GCS lake.
"""

import numpy as np
import pandas as pd

import stratum as st
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer._optimize import OptConfig, logical_optimize
from stratum.optimizer.logical._column_expr import ColumnExpr, OperandLeaf
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._ops import MethodCallOp
from tests.application._nyc_task import (
    apply_hgb, base_features, load_xy, model_features,
)


def make_nyc_lake(root, n_lots=80, n_events=800):
    """Write deterministic parquet tables that cover both BBL resolution paths."""
    rng = np.random.default_rng(0)
    boro = rng.integers(1, 6, n_lots)
    block = rng.integers(1, 99999, n_lots)
    lot = rng.integers(1, 9999, n_lots)
    bbl = boro * 10**9 + block * 10**4 + lot

    pluto = []
    for release in ("19v2", "20v7", "21v4", "22v3"):
        pluto.append(pd.DataFrame({
            "release": release, "bbl": bbl.astype(float), "borocode": boro,
            "unitsres": rng.integers(0, 40, n_lots),
            "unitstotal": rng.integers(0, 45, n_lots),
            "numbldgs": rng.integers(1, 3, n_lots),
            "numfloors": rng.uniform(1, 20, n_lots),
            "yearbuilt": rng.integers(1890, 2010, n_lots),
            "yearalter1": rng.integers(0, 2010, n_lots),
            "bldgarea": rng.uniform(1e3, 1e5, n_lots),
            "resarea": rng.uniform(1e3, 1e5, n_lots),
            "lotarea": rng.uniform(1e3, 1e5, n_lots),
            "assesstot": rng.uniform(1e4, 1e7, n_lots),
            "bldgclass": rng.choice(["C1", "D4", "R4"], n_lots),
            "ownertype": rng.choice(["P", "C", None], n_lots),
            "cd": rng.integers(101, 500, n_lots),
        }))
    (root / "pluto").mkdir(parents=True)
    pd.concat(pluto, ignore_index=True).to_parquet(
        root / "pluto" / "part-0.parquet", index=False)

    def events(date_col, extra):
        idx = rng.integers(0, n_lots, n_events)
        dates = (pd.Timestamp("2017-01-01")
                 + pd.to_timedelta(rng.integers(0, 6 * 365, n_events), unit="D"))
        bbl_text = pd.Series(bbl[idx]).astype("string")
        bbl_text[rng.random(n_events) < 0.2] = pd.NA
        return pd.DataFrame({
            date_col: dates, "bbl": bbl_text,
            "block": block[idx].astype(str), "lot": lot[idx].astype(str),
            "year": dates.year, **extra(idx),
        })

    violations = events("inspectiondate", lambda idx: {
        "boroid": boro[idx].astype(str),
        "class": rng.choice(["A", "B", "C"], len(idx)),
    })
    (root / "hpd_violations").mkdir()
    violations.to_parquet(root / "hpd_violations" / "part-0.parquet", index=False)

    boro_names = np.array([
        "MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND"])
    complaints = events("received_date", lambda idx: {
        "borough": boro_names[boro[idx] - 1],
        "major_category": rng.choice(
            ["HEAT/HOT WATER", "PLUMBING", "PAINT/PLASTER"], len(idx)),
    })
    (root / "hpd_complaints").mkdir()
    complaints.to_parquet(root / "hpd_complaints" / "part-0.parquet", index=False)
    return root


def build_pipeline(lake):
    with st.config_context(eager_data_ops=False):
        X, y, violations = load_xy(str(lake), subsample=None)
        features = base_features(X, violations, str(lake))
        return apply_hgb(model_features(features), y)


def _leaves(expr):
    if isinstance(expr, OperandLeaf):
        yield expr
    for slot in getattr(type(expr), "__slots__", ()):
        value = getattr(expr, slot, None)
        values = (value.values() if isinstance(value, dict)
                  else value if isinstance(value, (list, tuple)) else (value,))
        for item in values:
            if isinstance(item, ColumnExpr):
                yield from _leaves(item)


def test_nyc_plan_folds_assigns(tmp_path):
    pipeline = build_pipeline(make_nyc_lake(tmp_path / "lake"))
    root = logical_optimize(pipeline, OptConfig(), None)
    ops = list(topological_iterator(root))
    maps = [op for op in ops if isinstance(op, AssignMapOp)]
    generic_assigns = [op for op in ops if isinstance(op, MethodCallOp)
                       and op.method_name == "assign"]

    assert len(maps) == 16
    assert not generic_assigns
    # Three maps legitimately read from a different frame (event assembly and
    # label alignment); every other assignment is row-local to its source.
    assert sum(any(leaf.ref.k != 0 for entry in op.entries.values()
                   for leaf in _leaves(entry)) for op in maps) == 3


def test_nyc_pipeline_scores_two_future_cutoffs(tmp_path):
    pipeline = build_pipeline(make_nyc_lake(tmp_path / "lake"))
    with st.config(scheduler=True, debug_graph=True):
        search = pipeline.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy")
    scores = search.results_["scores"]
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0
