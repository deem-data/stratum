import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


class ThreeSeedStratifiedKFold:
    """Reproduce the original three 3-fold runs with seeds 42, 43, and 44."""

    def __init__(self, n_splits=3, random_states=(42, 43, 44)):
        self.n_splits = n_splits
        self.random_states = random_states

    def split(self, X, y=None, groups=None):
        for random_state in self.random_states:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=random_state,
            )
            yield from splitter.split(X, y)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * len(self.random_states)


class SoftVotingRandomForestClassifier(ClassifierMixin, BaseEstimator):
    """The original five-Random-Forest probability-averaging ensemble."""

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
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = []

        # This loop belongs inside the estimator so all five forests are fitted
        # independently on each CV fold's training rows.
        for i in range(self.n_sub_models):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state_base + i,
                n_jobs=self.n_jobs,
            )
            model.fit(X, y)
            self.models_.append(model)

        return self

    def predict_proba(self, X):
        probabilities = [model.predict_proba(X) for model in self.models_]
        return np.mean(np.asarray(probabilities), axis=0)

    def predict(self, X):
        # Preserve the original implementation, which selected the probability
        # column index directly after training on zero-based class labels.
        return np.argmax(self.predict_proba(X), axis=1)


def restore_original_labels(predictions, mode):
    """Map predictions from 0-6 back to the raw target domain 1-7."""
    if mode == "fit":
        # In fit mode a prediction node contains the fitted estimator.
        return predictions
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record only the provided training-table read.
    #    The original's dummy-file creation, test.csv processing, final-model fit,
    #    submission prediction, directory creation, and CSV output are dropped
    #    because they do not contribute to the cross-validated score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded every class represented fewer than
    #    three times before cross-validation. Row filtering must remain before
    #    mark_as_X/mark_as_y because it changes the number of samples.
    class_sizes = data.groupby("Cover_Type")["Cover_Type"].transform("size")
    cv_data = data[class_sizes >= 3].reset_index(drop=True)

    # Mark the RAW 1-7 target. Its zero-based transform is performed only after
    # mark_as_y, and predictions are converted back before accuracy is scored.
    y_raw = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The custom splitter exactly reproduces the original three outer ensemble
    # runs: 3-fold shuffled StratifiedKFold with seeds 42, 43, and 44. Thus the
    # grid-search score averages the same nine fold accuracies.
    X = cv_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=ThreeSeedStratifiedKFold(
            n_splits=3,
            random_states=(42, 43, 44),
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — the same two derived features are added,
    #    followed by the same removal of Aspect and the three source hillshade
    #    columns.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    elevation_at_hydrology = (
        X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    )

    features = X.assign(
        Hillshade_composite=hillshade_composite,
        Elevation_at_Hydrology=elevation_at_hydrology,
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    # The wrapper preserves the original soft-voting ensemble: five
    # RandomForestClassifier models with 100 trees and random states 100-104.
    ensemble = SoftVotingRandomForestClassifier(
        n_sub_models=5,
        n_estimators=100,
        random_state_base=100,
        n_jobs=-1,
    )
    pred_zero_based = features.skb.apply(ensemble, y=y_transformed)

    # Convert zero-based predictions back to the raw target domain. This is gated
    # on eval_mode because a prediction node contains an estimator during fit.
    pred = pred_zero_based.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is passed here; the splitter on mark_as_X drives all nine
    #    folds, and their mean accuracy equals the original nested mean.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="accuracy",
        )
        print(search.results_)
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
