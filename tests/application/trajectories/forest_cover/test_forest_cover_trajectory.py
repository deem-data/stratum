"""Replay the December 2021 forest-cover MLE-STAR trajectory on a shared cache.

The 68 pipelines come from mle-trajectories,
``tab_playground_dec_21/mle_star/skrubify_openai``. Their order follows the
execution entries in that run's ``final_state.json``: initial solutions, merges,
then each training round's ablation and four improvements, and the ensembles.
The CSV is generated locally, with all 54 features and the singleton class 5
that exercises the original rare-class handling. Model parameters, ablation
choices and CV schemes are preserved.

Only train9_improve0's scorer is adapted: its prepared target is passed through
``with_scoring(kwargs=...)`` instead of read by an estimator-style scorer.
Both selectors replay independently, and CI discovers one job for each.
"""
from functools import wraps
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import skrub
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

import stratum as st
from stratum.optimizer.physical._source_execs import PandasReadCSV, PolarsReadCSV
from stratum.runtime import _reuse_cache
from stratum.runtime._scheduler import SequentialScheduler


PIPELINE_DIR = Path(__file__).parent / "pipelines"
PIPELINES = ["init_code_1", "init_code_2", "train0_0", "train0_1"]
for _round in range(10):
    PIPELINES.extend([f"train{_round}", f"ablation_{_round}"])
    PIPELINES.extend(f"train{_round}_improve{i}" for i in range(4))
PIPELINES.extend(["train10", "ensemble/ensemble0", "ensemble/ensemble1",
                  "ensemble/final_solution"])


def make_data(root: Path) -> Path:
    """Small but complete terrain data, including one unstratifiable class."""
    rng = np.random.default_rng(42)
    y = np.concatenate([np.full(1 if label == 5 else 24, label)
                        for label in range(1, 8)])
    rng.shuffle(y)
    n = len(y)
    data = {
        "Id": np.arange(n) + 1,
        "Elevation": 1800 + y * 120 + rng.integers(-90, 91, n),
        "Aspect": rng.integers(0, 360, n),
        "Slope": rng.integers(0, 60, n),
        "Horizontal_Distance_To_Hydrology": rng.integers(1, 1000, n),
        "Vertical_Distance_To_Hydrology": rng.integers(-100, 300, n),
        "Horizontal_Distance_To_Roadways": rng.integers(1, 6000, n),
        "Hillshade_9am": rng.integers(0, 256, n),
        "Hillshade_Noon": rng.integers(0, 256, n),
        "Hillshade_3pm": rng.integers(0, 256, n),
        "Horizontal_Distance_To_Fire_Points": rng.integers(1, 6000, n),
    }
    for prefix, width in [("Wilderness_Area", 4), ("Soil_Type", 40)]:
        which = rng.integers(1, width + 1, n)
        data.update({f"{prefix}{i}": (which == i).astype(int)
                     for i in range(1, width + 1)})
    data["Cover_Type"] = y
    (root / "input").mkdir()
    pd.DataFrame(data).to_csv(root / "input/train.csv", index=False)
    return root


def load_pipeline(name, monkeypatch):
    # Custom estimators need an importable module while sklearn clones them.
    module_name = "forest_cover_" + name.replace("/", "_")
    spec = importlib.util.spec_from_file_location(module_name, PIPELINE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    with skrub.config_context(eager_data_ops=False):
        spec.loader.exec_module(module)
    return module


class Replay:
    def __init__(self, data_dir, monkeypatch):
        monkeypatch.chdir(data_dir)
        self.monkeypatch = monkeypatch
        self.reads = []
        self.taken = []

        def track_reads(process):
            @wraps(process)
            def reading(op, mode, inputs):
                self.reads.append(Path(op.file_path).name)
                return process(op, mode, inputs)
            return reading

        # Keep pd.read_csv's identity intact: extraction recognizes that callable
        # as a source, so wrapping it would silently disable source optimization.
        for reader in (PandasReadCSV, PolarsReadCSV):
            monkeypatch.setattr(reader, "process", track_reads(reader.process))
        run_fold = SequentialScheduler.run_fold

        def recording(sched, indices, profile=None):
            result = run_fold(sched, indices, profile)
            self.taken.append([str(op) for op in sched.fold_ops if op.cache_hit is not None])
            return result

        monkeypatch.setattr(SequentialScheduler, "run_fold", recording)

    def score(self, name):
        self.reads, self.taken = [], []
        module = load_pipeline(name, self.monkeypatch)
        scoring = None if name == "train9_improve0" else "accuracy"
        with st.config(scheduler=True), threadpool_limits(1):
            search = module.pred.skb.make_grid_search(
                n_jobs=1, fitted=True, refit=False, scoring=scoring)
        return search.results_, self.reads, self.taken


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return make_data(tmp_path_factory.mktemp("forest_cover"))


@pytest.fixture(scope="module")
def trajectory(data_dir, tmp_path_factory, trajectory_selector):
    with pytest.MonkeyPatch.context() as monkeypatch, st.config(
            reuse_cache=True, reuse_cache_dir=str(tmp_path_factory.mktemp("forest_cache"))):
        monkeypatch.setattr(_reuse_cache, "_active", None)
        replay = Replay(data_dir, monkeypatch)
        return {name: replay.score(name) for name in PIPELINES}


def test_replay_includes_every_pipeline():
    assert len(PIPELINES) == len(set(PIPELINES)) == 68
    assert set(PIPELINES) == {p.relative_to(PIPELINE_DIR).with_suffix("").as_posix()
                              for p in PIPELINE_DIR.rglob("*.py")}


@pytest.mark.parametrize("name", PIPELINES)
def test_pipeline_scores_all_variants(trajectory, name):
    results, _, folds = trajectory[name]
    # Ablation 0 has three candidates; the remaining studies each have four.
    candidates = 3 if name == "ablation_0" else 4 if name.startswith("ablation_") else 1
    assert len(results) == candidates
    scores = np.asarray(results["scores"])
    assert np.isfinite(scores).all()
    assert ((0 <= scores) & (scores <= 1)).all()
    expected_folds = {"ensemble/ensemble0": 15, "ensemble/ensemble1": 9,
                      "ensemble/final_solution": 9}.get(name, 3)
    assert len(folds) == expected_folds


def test_csv_is_reused_across_pipelines(trajectory):
    reads = {name: run[1] for name, run in trajectory.items()}
    assert reads == {name: ["train.csv"] if name == PIPELINES[0] else []
                     for name in PIPELINES}


def test_shared_work_is_reused_in_folds(trajectory):
    # These retain keyed feature work from earlier pipelines. Opaque GroupBy
    # transforms and map calls in other pipelines prevent reuse below their
    # row filter; those pipelines still reuse the CSV above it.
    for name in ("ablation_0", "train1", "train2"):
        assert all(trajectory[name][2]), name


@pytest.mark.parametrize("name", ["ablation_4", "train1", "train5", "train10",
                                  "ensemble/final_solution"])
def test_reuse_does_not_change_scores(trajectory, data_dir, name, monkeypatch):
    with st.config(reuse_cache=False):
        fresh, _, _ = Replay(data_dir, monkeypatch).score(name)
    np.testing.assert_allclose(trajectory[name][0].sort("id")["scores"],
                               fresh.sort("id")["scores"])


def test_prepared_target_metric_matches_original(trajectory, data_dir, monkeypatch):
    """Check the adapted metric against the corpus's estimator-style scorer."""
    monkeypatch.chdir(data_dir)
    module = load_pipeline("train9_improve0", monkeypatch)

    def original_scorer(estimator, X, y_true):
        return accuracy_score(np.asarray(X["__prepared_cover_type"]).reshape(-1),
                              estimator.predict(X))

    original = module.features.skb.apply(module.model, y=module.y)
    with st.config(scheduler=False), threadpool_limits(1):
        upstream = original.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring=original_scorer)
    np.testing.assert_allclose(trajectory["train9_improve0"][0]["scores"],
                               upstream.results_["mean_test_score"])
