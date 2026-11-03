import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class TrainOnlyProblematicClasses(BaseCrossValidator):
    """Stratify eligible rows while keeping rare-class rows training-only."""

    def __init__(self, n_splits=3, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups=None):
        y = np.asarray(y).reshape(-1)
        classes, counts = np.unique(y, return_counts=True)
        problematic = np.isin(y, classes[counts < self.n_splits])

        problematic_idx = np.flatnonzero(problematic)
        eligible_idx = np.flatnonzero(~problematic)

        inner_cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        for train_idx, validation_idx in inner_cv.split(
            np.zeros((len(eligible_idx), 1)), y[eligible_idx]
        ):
            # The rare-class rows are always in training and never in validation.
            yield (
                np.concatenate([eligible_idx[train_idx], problematic_idx]),
                eligible_idx[validation_idx],
            )


class AveragedRareClassForests(ClassifierMixin, BaseEstimator):
    """Reproduce the original two-forest probability ensemble per outer fold."""

    def __init__(
        self,
        n_estimators=100,
        model1_random_state=42,
        model2_random_state=43,
        n_jobs=-1,
        n_classes=7,
        marker_column="_problematic_class",
    ):
        self.n_estimators = n_estimators
        self.model1_random_state = model1_random_state
        self.model2_random_state = model2_random_state
        self.n_jobs = n_jobs
        self.n_classes = n_classes
        self.marker_column = marker_column

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y).reshape(-1).astype(int)

        problematic = X[self.marker_column].to_numpy(dtype=bool)
        model_features = X.drop(columns=[self.marker_column])

        # Model 1 excludes all problematic samples, as in the original script.
        self.model1_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.model1_random_state,
            n_jobs=self.n_jobs,
        )
        self.model1_.fit(model_features.loc[~problematic], y[~problematic])

        # Model 2 includes the problematic samples that the custom outer splitter
        # placed in every training fold.
        self.model2_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.model2_random_state,
            n_jobs=self.n_jobs,
        )
        self.model2_.fit(model_features, y)

        # The dataset has Cover_Type labels 1-7, hence seven transformed classes.
        self.classes_ = np.arange(self.n_classes)
        return self

    def _full_probabilities(self, model, X):
        raw_probabilities = model.predict_proba(X)
        full_probabilities = np.zeros(
            (len(X), self.n_classes), dtype=float
        )
        for source_column, class_label in enumerate(model.classes_):
            full_probabilities[:, int(class_label)] = raw_probabilities[
                :, source_column
            ]
        return full_probabilities

    def predict_proba(self, X):
        X = pd.DataFrame(X).drop(columns=[self.marker_column])
        probabilities1 = self._full_probabilities(self.model1_, X)
        probabilities2 = self._full_probabilities(self.model2_, X)
        return (probabilities1 + probabilities2) / 2.0

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def restore_original_labels(value, mode):
    # During fit, a prediction node evaluates to the fitted estimator.
    if mode == "fit":
        return value
    return np.asarray(value) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only constructs
    #    the lazy plan and does not read the file.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The problematic-class mask is computed on the full target
    #    exactly where the original computed value_counts before its fold loop.
    #    It is retained only as wrapper metadata and is removed before either
    #    forest sees the model features.
    target_counts = train_df["Cover_Type"].value_counts()
    problematic_mask = train_df["Cover_Type"].map(target_counts) < 3
    marked_df = train_df.assign(_problematic_class=problematic_mask)

    # Mark the RAW 1-7 target first, then perform the original 0-6 transform.
    y = marked_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # This custom splitter reproduces the original StratifiedKFold over eligible
    # rows while putting rare-class rows in every training fold and no test fold.
    X = marked_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=TrainOnlyProblematicClasses(
            n_splits=3, shuffle=True, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Model. The wrapper preserves the original per-fold behavior: model 1
    #    excludes problematic rows, model 2 includes them, their seven-column
    #    probability matrices are averaged, and argmax supplies the prediction.
    #    n_classes=7 is the concrete value inferred from Cover_Type classes 1-7.
    ensemble = AveragedRareClassForests(
        n_estimators=100,
        model1_random_state=42,
        model2_random_state=43,
        n_jobs=-1,
        n_classes=7,
    )
    pred_transformed = X.skb.apply(ensemble, y=y_transformed)

    # Convert predictions from 0-6 back to the raw target's 1-7 scoring domain.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    # 4. Score. The custom three-fold splitter on mark_as_X drives validation.
    #    OOF arrays and progress printing are omitted because skrub performs the
    #    same fold-level fitting and accuracy evaluation.
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
