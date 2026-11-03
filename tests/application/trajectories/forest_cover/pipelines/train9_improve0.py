import numpy as np
import pandas as pd
import skrub
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class PrepareGroupedTarget(TransformerMixin, BaseEstimator):
    """Reproduce the original full-table rare-class grouping."""

    def __init__(
        self,
        target_column="Cover_Type",
        prepared_target_column="__prepared_cover_type",
        n_splits=3,
    ):
        self.target_column = target_column
        self.prepared_target_column = prepared_target_column
        self.n_splits = n_splits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        transformed = result[self.target_column] - 1

        class_counts = transformed.value_counts()
        rare_classes = class_counts[class_counts < self.n_splits].index

        if not rare_classes.empty:
            other_category_value = transformed.max() + 1
            transformed = transformed.replace(
                rare_classes.tolist(), other_category_value
            )

        result[self.prepared_target_column] = transformed.astype(int)
        return result


class ExcludeProblematicClassesCV(BaseCrossValidator):
    """Reproduce the original conditional exclusion before StratifiedKFold."""

    def __init__(
        self,
        n_splits=3,
        shuffle=True,
        random_state=42,
        prepared_target_column="__prepared_cover_type",
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.prepared_target_column = prepared_target_column

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        prepared_y = np.asarray(X[self.prepared_target_column]).reshape(-1)

        classes, counts = np.unique(prepared_y, return_counts=True)
        problematic_classes = classes[counts < self.n_splits]
        excluded = np.isin(prepared_y, problematic_classes)
        included_indices = np.flatnonzero(~excluded)
        included_y = prepared_y[included_indices]

        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        splits = splitter.split(
            np.zeros((len(included_indices), 1)),
            included_y,
        )

        # This maps the delegated splitter's indices back to the original table.
        # It is the implementation of a BaseCrossValidator, not an outer fold
        # loop in the DataOps plan.
        return map(
            lambda indices: (
                included_indices[indices[0]],
                included_indices[indices[1]],
            ),
            splits,
        )


class PreparedTargetRandomForest(ClassifierMixin, BaseEstimator):
    """Fit the original random forest using the prepared target metadata."""

    def __init__(
        self,
        prepared_target_column="__prepared_cover_type",
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ):
        self.prepared_target_column = prepared_target_column
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        # `y` is deliberately the marked raw Cover_Type target, which keeps the
        # target mark connected to the prediction graph. The original model was
        # fitted on the separately prepared zero-indexed/grouped target.
        del y
        prepared_y = np.asarray(
            X[self.prepared_target_column]
        ).reshape(-1)
        model_X = X.drop(columns=[self.prepared_target_column])

        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(model_X, prepared_y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        model_X = X.drop(columns=[self.prepared_target_column])
        return self.model_.predict(model_X)

    def predict_proba(self, X):
        model_X = X.drop(columns=[self.prepared_target_column])
        return self.model_.predict_proba(model_X)


def prepared_target_accuracy(y_true, predictions, *, prepared_target):
    """Score predictions against the target used by the original script."""
    del y_true
    prepared_target = np.asarray(prepared_target).reshape(-1)
    return accuracy_score(prepared_target, predictions)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read. The original's custom FileNotFoundError
    #    message is omitted because it does not produce the validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # The original computes class frequencies over the complete table before
    # cross-validation. This content-dependent preparation therefore remains
    # before the marks. It adds metadata without modifying the raw target.
    prepared_data = data.skb.apply(
        PrepareGroupedTarget(
            target_column="Cover_Type",
            prepared_target_column="__prepared_cover_type",
            n_splits=3,
        )
    )

    # 2. Prepare data: mark the RAW target and the design matrix. The custom
    #    splitter reproduces the conditional exclusion of classes that remain
    #    too small after grouping, followed by the original StratifiedKFold.
    y = prepared_data["Cover_Type"].skb.mark_as_y()
    X = prepared_data.drop(
        columns=["Cover_Type", "Id", "Aspect"]
    ).skb.mark_as_X(
        cv=ExcludeProblematicClassesCV(
            n_splits=3,
            shuffle=True,
            random_state=42,
            prepared_target_column="__prepared_cover_type",
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. The prepared-target metadata remains in
    #    the table until the wrapper estimator so it stays aligned through every
    #    split; the wrapper removes it before fitting the random forest.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"]
            + X["Hillshade_Noon"]
            + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"]
            - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    # Same model family and every hyperparameter value as the original. The
    # wrapper is needed only to fit and score against the original script's
    # data-dependent prepared target while keeping the raw marked y connected.
    model = PreparedTargetRandomForest(
        prepared_target_column="__prepared_cover_type",
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    # The corpus uses an estimator-style scorer that reads this metadata from
    # X. Declare it as a metric with a graph-fed argument instead: Stratum scores
    # predictions directly, and skrub keeps the argument aligned with each fold.
    pred = features.skb.apply(model, y=y).skb.with_scoring(
        make_scorer(prepared_target_accuracy),
        kwargs={"prepared_target": features["__prepared_cover_type"]},
    )

    # 4. Score. No cv= here -- the splitter on mark_as_X drives. The callable
    #    scorer compares predictions with the grouped, zero-indexed target used
    #    by the original accuracy calculation.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
