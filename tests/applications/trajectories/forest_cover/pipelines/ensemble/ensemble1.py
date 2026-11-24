import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class RepeatedSeedStratifiedKFold(BaseCrossValidator):
    """Reproduce the original three 3-fold runs with seeds 42, 43, and 44."""

    def __init__(self, n_splits=3, random_states=(42, 43, 44)):
        self.n_splits = n_splits
        self.random_states = random_states

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * len(self.random_states)

    def split(self, X, y, groups=None):
        y_array = np.asarray(y).reshape(-1)
        for random_state in self.random_states:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=random_state,
            )
            yield from splitter.split(X, y_array)


class SoftVotingRandomForestEnsemble(ClassifierMixin, BaseEstimator):
    """Fit five random forests and average their predicted probabilities."""

    def __init__(
        self,
        n_sub_models=5,
        n_estimators=100,
        random_state_base=100,
        n_jobs=-1,
    ):
        self.n_sub_models = n_sub_models
        self.n_estimators = n_estimators
        self.random_state_base = random_state_base
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y_array = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y_array)
        self.models_ = []

        for model_index in range(self.n_sub_models):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state_base + model_index,
                n_jobs=self.n_jobs,
            )
            model.fit(X, y_array)
            self.models_.append(model)

        return self

    def predict_proba(self, X):
        probabilities = [model.predict_proba(X) for model in self.models_]
        return np.mean(np.asarray(probabilities), axis=0)

    def predict(self, X):
        # This intentionally returns argmax column indices rather than indexing
        # model.classes_, exactly matching the original script.
        return np.argmax(self.predict_proba(X), axis=1)


def restore_original_target_domain(predictions, mode):
    """Map predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode the prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except only customized
    #    the missing-file message and is not part of producing the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Faithfully remove target classes with fewer than three
    #    rows before marking X and y, since this content-dependent filtering
    #    changes which rows are scored. Counting raw labels is equivalent to
    #    counting the original labels shifted by -1.
    raw_target = data["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[
        ~raw_target.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target, then perform the original 1-7 to 0-6 transformation
    # downstream for model fitting. Predictions are mapped back to 1-7 below so
    # accuracy is evaluated in the raw target domain.
    y_raw = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The custom splitter emits the original nine folds: three 3-fold
    # StratifiedKFold runs using random states 42, 43, and 44.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=RepeatedSeedStratifiedKFold(
            n_splits=3,
            random_states=(42, 43, 44),
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. Derived columns are appended in
    #    the same order as in the original, followed by the same column drops.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    # The wrapper preserves the original per-fold five-model soft-voting
    # ensemble, including RandomForest seeds 100 through 104.
    model = SoftVotingRandomForestEnsemble(
        n_sub_models=5,
        n_estimators=100,
        random_state_base=100,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Since mark_as_y holds the raw 1-7 labels, restore predictions to that
    # domain. The operation is gated because in fit mode pred_transformed is the
    # fitted estimator rather than an array of predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_domain,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here: the repeated stratified splitter on mark_as_X
    #    drives validation. Averaging all nine fold accuracies equals the
    #    original mean of its three per-run, three-fold means. Progress printing
    #    from the manual loops is intentionally omitted.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="accuracy",
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
