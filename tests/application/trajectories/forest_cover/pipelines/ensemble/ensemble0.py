import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class ExplicitRepeatedStratifiedKFold(BaseCrossValidator):
    """Reproduce StratifiedKFold runs using consecutive explicit seeds."""

    def __init__(self, n_splits=3, n_repeats=5, first_random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.first_random_state = first_random_state

    def split(self, X, y=None, groups=None):
        for repeat in range(self.n_repeats):
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.first_random_state + repeat,
            )
            yield from splitter.split(X, y)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


def restore_original_labels(predictions, mode):
    """Map predictions from 0-6 back to the raw target domain, 1-7."""
    if mode == "fit":
        # In fit mode a prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except and progress
    #    messages are omitted because they do not contribute to the CV score.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded target classes represented fewer
    #    than three times before cross-validation. Row filtering must remain
    #    before the marks because it changes the number of samples.
    class_sizes = train_df.groupby("Cover_Type")["Cover_Type"].transform("size")
    cv_df = train_df[class_sizes >= 3].reset_index(drop=True)

    # Mark the raw 1-7 target first. The original model was fitted on labels
    # shifted to 0-6, so that transformation occurs only after mark_as_y.
    y_raw = cv_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The custom splitter exactly combines the original five StratifiedKFold
    # runs, whose random states were 42, 43, 44, 45, and 46. Averaging its 15
    # fold scores is equivalent to averaging the five three-fold run means.
    X = cv_df.drop(columns=["Cover_Type"]).skb.mark_as_X(
        cv=ExplicitRepeatedStratifiedKFold(
            n_splits=3,
            n_repeats=5,
            first_random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering with the same formulas and feature set.
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
            "Id",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert 0-6 model predictions back to the raw 1-7 target domain used by
    # mark_as_y. The eval-mode gate leaves the fitted estimator untouched.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is passed here; the exact repeated splitter attached to
    #    mark_as_X drives validation. Per-fold progress printing is omitted.
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
