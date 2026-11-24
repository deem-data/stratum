"""Replay the NYC housing trajectory on generated local data with a shared cache.

The 32 pipelines come from stratum-bench,
``trajectories/nyc-housing/mle-claude-run``, in their original order. Generated
tables preserve the columns and partitions used by the pipelines. Grid search
uses Stratum's scheduler and the task's average-precision scorer.

The tests check reused reads, per-fold cache hits, and scores against uncached
runs for selected pipelines. Opaque operations can prevent downstream reuse.

The default selector replays on a fresh in-memory cache. Use
``--selector=greedy`` to exercise the greedy selector instead.
"""
from functools import wraps
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet
import pytest
import skrub
from threadpoolctl import threadpool_limits

import stratum as st
from stratum._config import config
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer._optimize import OptConfig, logical_optimize
from stratum.optimizer.logical._combine_ops import AssignOp
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._ops import MethodCallOp
from stratum.runtime import _reuse_cache
from stratum.runtime._scheduler import SequentialScheduler

TRAJECTORY = Path(__file__).parent / "pipelines"
SCORING = "average_precision"
PIPELINES = [f"pipeline_{i:02d}" for i in range(1, 33)]


# --- the lake -----------------------------------------------------------------------

def make_lake(root: Path, n_lots=60, n_events=400, seed=0) -> Path:
    """Every table the trajectory reads, with the columns its specs name.

    Events fall between 2008 and the lake's end (2023), a fifth of them without a
    BBL, so keys are also built from borough/block/lot or bridged from a BIN.
    Partitioned tables carry the ``year`` the reads filter on.
    """
    rng = np.random.default_rng(seed)
    boro = rng.integers(1, 6, n_lots)
    block = rng.integers(1, 99999, n_lots)
    lot = rng.integers(1, 9999, n_lots)
    bbl = boro * 10**9 + block * 10**4 + lot
    bins = (boro * 10**6 + rng.choice(10**6, n_lots, replace=False)).astype(str)
    boro_names = np.array(["MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND"])

    def write(name, frame):
        (root / name).mkdir(parents=True)
        frame.to_parquet(root / name / "part-0.parquet", index=False)

    write("pluto", pd.concat([pd.DataFrame({
        "release": release, "bbl": bbl.astype(float), "borocode": boro,
        "unitsres": rng.integers(0, 40, n_lots), "unitstotal": rng.integers(0, 45, n_lots),
        "numbldgs": rng.integers(1, 3, n_lots), "numfloors": rng.uniform(1, 20, n_lots),
        "yearbuilt": rng.integers(1890, 2010, n_lots),
        "yearalter1": rng.integers(0, 2010, n_lots),
        "bldgarea": rng.uniform(1e3, 1e5, n_lots), "resarea": rng.uniform(1e3, 1e5, n_lots),
        "lotarea": rng.uniform(1e3, 1e5, n_lots), "assesstot": rng.uniform(1e4, 1e7, n_lots),
        "bldgclass": rng.choice(["C1", "D4", "R4"], n_lots),
        "ownertype": rng.choice(["P", "C", None], n_lots),
        "cd": rng.integers(101, 500, n_lots),
    }) for release in ("19v2", "20v7", "21v4", "22v3")], ignore_index=True))
    write("building_footprints", pd.DataFrame({"bin": bins, "base_bbl": bbl.astype(float)}))

    def events(date, *, key=("bbl",), boro_as=None, date_format=None, partitioned=True,
               **columns):
        """``n_events`` rows on random lots: the date, the key columns in ``key``
        (bbl, block/lot with the borough as ``boro_as``, bin), and ``columns``,
        each drawn from the values it lists."""
        idx = rng.integers(0, n_lots, n_events)
        when = (pd.Timestamp("2008-01-01")
                + pd.to_timedelta(rng.integers(0, 15 * 365, n_events), unit="D"))
        frame = {date: when.strftime(date_format) if date_format else when}
        if "bbl" in key:
            frame["bbl"] = pd.Series(bbl[idx]).astype("string").mask(
                rng.random(n_events) < 0.2)
        if boro_as is not None:
            frame[boro_as[0]] = (boro_names[boro[idx] - 1] if boro_as[1] == "name"
                                 else boro[idx].astype(str))
            frame["block"], frame["lot"] = block[idx].astype(str), lot[idx].astype(str)
        if "bin" in key:
            frame["bin"] = bins[idx]
        for name, values in columns.items():
            frame[name] = (rng.choice(values, n_events) if isinstance(values, list)
                           else rng.uniform(0, values, n_events))
        if partitioned:
            frame["year"] = when.year
        return pd.DataFrame(frame)

    boroid = ("boroid", "code")
    write("hpd_violations", events(
        "inspectiondate", boro_as=boroid, **{"class": ["A", "B", "C"]},
        ordernumber=[f"{n}" for n in range(500, 520)]))
    write("hpd_complaints", events(
        "received_date", boro_as=("borough", "name"),
        major_category=["HEAT/HOT WATER", "PLUMBING", "PAINT/PLASTER", "DOOR/WINDOW",
                        "ELECTRIC", "PESTS", "WATER LEAK", "GENERAL", "UNSANITARY",
                        "FLOORING", "SAFETY"]))
    write("hpd_litigations", events(
        "caseopendate", boro_as=boroid, partitioned=False,
        casetype=["Tenant Action", "Heat and Hot Water", "Comprehensive", "False Cert"]))
    write("hpd_vacate_orders", events(
        "vacate_effective_date", partitioned=False,
        primary_vacate_reason=["Fire Damage", "Illegal Occupancy"],
        number_of_vacated_units=10.0))
    write("hpd_omo_charges", events(
        "omocreatedate", boro_as=("boro_id", "code"), partitioned=False,
        worktypegeneral=["PLUMBING", "HEAT", "ELECTRIC", "GENERAL", "PAINT", "DOOR"],
        omoawardamount=5000.0))
    write("hpd_hwo_charges", events(
        "hwocreatedate", boro_as=boroid, partitioned=False,
        worktypegeneral=["PLUMBING", "HEAT", "ELECTRIC", "GENERAL"],
        hwoapprovedamount=2000.0))
    write("hpd_bedbug_reports", events(
        "filing_date", partitioned=False, infested_dwelling_unit_count=5.0))
    write("hpd_aep_buildings", events("aep_start_date", partitioned=False))
    write("dob_violations", events(
        "issue_date", key=("bin",), boro_as=("boro", "code"), date_format="%Y%m%d",
        violation_type_code=["E", "LL6291", "ACC1", "IMEGNCY", "C", "P", "HBLVIO"]))
    write("dob_ecb_violations", events(
        "issue_date", key=("bin",), boro_as=("boro", "code"), date_format="%Y%m%d",
        severity=["CLASS - 1", "CLASS - 2", "CLASS - 3", "Hazardous"]))
    write("dob_complaints", events(
        "date_entered", key=("bin",), date_format="%m/%d/%Y",
        complaint_category=["05", "45", "83", "71", "63", "4G", "23"]))
    write("dohmh_rodent_inspections", events(
        "inspection_date", boro_as=("boro_code", "code"),
        result=["Passed", "Rat Activity", "Failed for Other R", "Bait applied"]))
    write("evictions", events(
        "executed_date", partitioned=False,
        residential_commercial_ind=["Residential", "Commercial"]))
    for table in ("service_requests_311_2010_2019", "service_requests_311_2020_present"):
        write(table, events(
            "created_date", complaint_type=["Noise", "Street Condition", "Rodent",
                                            "Illegal Parking", "Water System", "Sewer"],
            agency=["NYPD", "DOT", "DOHMH", "DEP", "HPD"]))

    documents = [f"D{n:05d}" for n in range(n_events)]
    master = events("recorded_datetime",
                    doc_type=["DEED", "MTGE", "AL&R", "SAT", "AGMT", "ASST", "RPTT&RET",
                              "MISC"], document_amt=1e6)
    write("acris_master", master.drop(columns="bbl").assign(document_id=documents))
    legals = events("x", boro_as=("borough", "code"), partitioned=False)
    write("acris_legals", legals[["borough", "block", "lot"]].assign(document_id=documents))
    write("acris_parties", pd.DataFrame({
        "document_id": documents * 2, "party_type": ["1"] * n_events + ["2"] * n_events,
        "name": rng.choice(["ACME REALTY LLC", "Smith, John", "BIG HOLDINGS INC.",
                            "THE LANDLORD CO", "Doe Jane"], 2 * n_events)}))
    return root


# --- the replay ----------------------------------------------------------------------

def _xgboost_fits_pandas_categories() -> bool:
    """Check whether the installed XGBoost accepts pandas string categories."""
    from xgboost import XGBClassifier
    X = pd.DataFrame({"c": pd.Series(["a", "b"] * 4, dtype="category")})
    try:
        XGBClassifier(enable_categorical=True, n_estimators=1).fit(X, [0, 1] * 4)
    except ValueError:
        return False
    return True


def _skip_reason(name: str) -> str | None:
    """Return a dependency or compatibility reason to skip this pipeline, if any."""
    if name == "pipeline_17" and not _xgboost_fits_pandas_categories():
        return "this xgboost cannot fit pandas 3 categories"
    if name == "pipeline_20" and not all(
        importlib.util.find_spec(m) for m in ("skorch", "torch")
    ):
        return "skorch or torch is not installed"
    return None


def load_pipeline(name: str):
    """Import a pipeline with eager DataOps evaluation disabled."""
    spec = importlib.util.spec_from_file_location(name, TRAJECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with skrub.config_context(eager_data_ops=False):
        spec.loader.exec_module(module)
    return module.pred


class Replay:
    """Score pipelines, recording table reads and per-fold cache hits."""

    def __init__(self, lake: Path, monkeypatch):
        self.reads: list[str] = []
        self.taken: list[list[str]] = []

        def track_reads(reader):
            @wraps(reader)
            def reading(source, *args, **kwargs):
                self.reads.append(Path(source).relative_to(lake).parts[0])
                return reader(source, *args, **kwargs)
            return reading

        run_fold = SequentialScheduler.run_fold

        def recording(sched, cv_id):
            out = run_fold(sched, cv_id)
            self.taken.append(sorted(str(op) for op in sched.fold_ops
                                     if op.cache_hit is not None))
            return out

        monkeypatch.setattr(pyarrow.parquet, "read_table", track_reads(pyarrow.parquet.read_table))
        monkeypatch.setattr(pl, "read_parquet", track_reads(pl.read_parquet))
        monkeypatch.setattr(pl, "scan_parquet", track_reads(pl.scan_parquet))
        monkeypatch.setattr(SequentialScheduler, "run_fold", recording)
        monkeypatch.syspath_prepend(str(TRAJECTORY))
        monkeypatch.delitem(sys.modules, "common", raising=False)
        import common
        monkeypatch.setattr(common, "LAKE", str(lake))

    def score(self, name: str):
        """Return search results, tables read, and per-fold cache hits for ``name``."""
        self.reads, self.taken = [], []
        pred = load_pipeline(name)
        with st.config(scheduler=True), threadpool_limits(1):
            search = pred.skb.make_grid_search(
                n_jobs=1, fitted=True, refit=False, scoring=SCORING)
        return search.results_, set(self.reads), self.taken


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    return make_lake(tmp_path_factory.mktemp("nyc") / "lake")


@pytest.fixture(scope="module")
def trajectory(lake, trajectory_selector):
    """Replay pipelines in order with one cache; record results or skip reasons."""
    replayed = {}
    with pytest.MonkeyPatch.context() as monkeypatch, config(
            reuse_cache=True):
        monkeypatch.setattr(_reuse_cache, "_active", None)
        replay = Replay(lake, monkeypatch)
        for name in PIPELINES:
            replayed[name] = _skip_reason(name) or replay.score(name)
        monkeypatch.setattr(_reuse_cache, "_active", None)
    return replayed


def ran(trajectory):
    return {name: run for name, run in trajectory.items() if not isinstance(run, str)}


# --- what the replay shows -------------------------------------------------------------

# Expected table reads when earlier pipelines' entries remain cached.
# Unlisted pipelines read no tables; repeated reads use new columns or windows.
READS = {
    "pipeline_01": {"pluto", "hpd_violations", "hpd_complaints"},
    "pipeline_02": {"hpd_litigations"},
    "pipeline_03": {"hpd_omo_charges"},
    "pipeline_04": {"hpd_hwo_charges"},
    "pipeline_05": {"hpd_vacate_orders", "hpd_aep_buildings"},
    "pipeline_06": {"hpd_bedbug_reports"},
    "pipeline_07": {"service_requests_311_2010_2019", "service_requests_311_2020_present"},
    # One BIN -> BBL bridge for both DOB tables; pipeline_09 finds it cached.
    "pipeline_08": {"building_footprints", "dob_violations", "dob_ecb_violations"},
    "pipeline_09": {"dob_complaints"},
    "pipeline_10": {"dohmh_rodent_inspections"},
    "pipeline_11": {"evictions"},
    "pipeline_24": {"acris_master", "acris_legals"},
    # The violations again, with the order number column.
    "pipeline_25": {"hpd_violations"},
    # Violations and complaints again, since 2010 and since 2015.
    "pipeline_26": {"hpd_violations", "hpd_complaints"},
    "pipeline_27": {"hpd_violations", "hpd_complaints"},
    "pipeline_30": {"acris_parties"},
}


def test_each_pipeline_reads_only_what_no_earlier_one_read(trajectory):
    reads = {name: run[1] for name, run in ran(trajectory).items()}
    assert reads == {name: READS.get(name, set()) for name in reads}


def test_each_pipeline_takes_in_every_fold_what_it_shares(trajectory, trajectory_selector):
    # Later pipelines share base features and folds with pipeline_01. The UDF
    # computing top-k categories prevents reuse of its downstream operations,
    # so require at least one hit per fold, not reuse of every feature block.
    taken = {name: run[2] for name, run in ran(trajectory).items() if name != "pipeline_01"}
    if trajectory_selector == "greedy":
        # The first neighborhood UDF uses pandas row labels. Its split reads
        # pandas and its conversions back to polars retain the index, unlike
        # the earlier pipelines' polars splits. Those are different physical
        # lineages, so only the work before the split can be reused here.
        assert not any(taken.pop("pipeline_12"))
    assert {name: all(folds) for name, folds in taken.items()} == dict.fromkeys(taken, True)


@pytest.mark.parametrize("name", ["pipeline_09", "pipeline_25", "pipeline_29"])
def test_reuse_does_not_change_scores(trajectory, lake, name, monkeypatch):
    # A bridge, a re-read table and a whole feature set taken from the cache.
    if isinstance(trajectory[name], str):
        pytest.skip(trajectory[name])
    with config(reuse_cache=False):
        uncached, _, _ = Replay(lake, monkeypatch).score(name)
    cached = trajectory[name][0]
    np.testing.assert_allclose(cached.sort("id")["scores"], uncached.sort("id")["scores"])


# --- the plan --------------------------------------------------------------------------

def test_the_anchor_plan_exposes_assignments_and_expressions(lake, monkeypatch):
    Replay(lake, monkeypatch)
    root = logical_optimize(load_pipeline("pipeline_01"), OptConfig(), None)
    ops = list(topological_iterator(root))
    # Exclude clones introduced by fit/predict unrolling.
    assigns = [op for op in ops if isinstance(op, AssignOp) and not op.was_cloned]
    assert len(assigns) == 16
    assert not [op for op in ops if isinstance(op, MethodCallOp) and op.method_name == "assign"]
    # Value computations remain explicit producers; physical planning can fuse
    # private chains while keeping shared expressions available for reuse.
    assert any(isinstance(source, ExpressionMapOp)
               for op in assigns for source in op.inputs[1:])
