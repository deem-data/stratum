"""Small generated Forest Cover classification pipeline with three-fold CV."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.physical._predictor_execs import SklearnRandomForestClassifier
from stratum.optimizer.physical._source_execs import PolarsReadCSV


# Keep this preprocessing test on the same sklearn model under both selectors.
# The Rust forest currently supports only the gini criterion.
FOREST_PARAMS = dict(n_estimators=8, max_depth=5, criterion="entropy",
                     random_state=42, n_jobs=1)


def make_cover_data() -> pd.DataFrame:
    """Six eligible classes and one two-row class removed before splitting."""
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(1, 7), 12).tolist() + [7, 7]
    labels = np.asarray(labels)
    rng.shuffle(labels)
    return pd.DataFrame({
        "Id": np.arange(len(labels)) + 1,
        "Elevation": labels * 100 + rng.normal(0, 15, len(labels)),
        "Slope": rng.integers(0, 45, len(labels)),
        "Soil_Type": rng.integers(0, 4, len(labels)),
        "Cover_Type": labels,
    })


def restore_original_labels(predictions, mode):
    """The scheduler's fit_transform and predict passes both produce predictions."""
    return predictions + 1


def build_pipeline(path):
    source = st.as_data_op(str(path)).skb.apply_func(pd.read_csv)
    raw_target = source["Cover_Type"]
    counts = raw_target.value_counts()
    eligible = counts[counts >= 3].index
    filtered = source[raw_target.isin(eligible)].reset_index(drop=True)

    y = filtered["Cover_Type"].skb.mark_as_y()
    X = filtered.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )
    model = RandomForestClassifier(**FOREST_PARAMS)
    predictions = X.skb.apply(model, y=y - 1)
    return predictions.skb.apply_func(restore_original_labels, st.eval_mode())


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_cover_pipeline_scores_generated_csv(tmp_path, selector):
    path = tmp_path / "train.csv"
    frame = make_cover_data()
    frame.to_csv(path, index=False)
    filtered = frame.loc[frame["Cover_Type"] != 7].reset_index(drop=True)
    expected = cross_val_score(
        RandomForestClassifier(**FOREST_PARAMS),
        filtered.drop(columns=["Id", "Cover_Type"]),
        filtered["Cover_Type"],
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="accuracy",
    ).mean()

    with st.config_context(eager_data_ops=False), st.config(
            scheduler=True, implementation_selector=selector, debug_graph=False):
        predictions = build_pipeline(path)
        ops, *_ = optimize(predictions, OptConfig(dataframe_ops=True))
        if selector == "greedy":
            assert any(isinstance(op, PolarsReadCSV) for op in ops)
            assert any(isinstance(op, SklearnRandomForestClassifier) for op in ops)
        search = predictions.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy")

    assert search.results_ is not None
    assert len(search.results_) == 1
    assert search.results_["scores"][0] == pytest.approx(expected)
