import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


class AddSoilInteractions(TransformerMixin, BaseEstimator):
    """Create the original dynamically discovered interaction columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        soil_type_cols = [col for col in X.columns if "Soil_Type" in col]
        wilderness_area_cols = [
            col for col in X.columns if "Wilderness_Area" in col
        ]

        # Keep the original loops so the generated names and column order are exact.
        if "Elevation" in X.columns:
            for soil_col in soil_type_cols:
                X[f"{soil_col}_x_Elevation"] = X[soil_col] * X["Elevation"]

        for soil_col in soil_type_cols:
            for wilderness_col in wilderness_area_cols:
                X[f"{soil_col}_x_{wilderness_col}"] = (
                    X[soil_col] * X[wilderness_col]
                )

        return X


def restore_original_target_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        return predictions
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded, so importing this module does not access
    #    the file. The original progress messages and explicit try/except are not
    #    part of producing the cross-validated score and are omitted.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — row filtering must happen before the marks because it
    #    changes the number of samples. Counting raw labels 1-7 is equivalent to
    #    counting the original transformed labels 0-6. Written as recorded ops
    #    (value_counts / isin / boolean mask), mirroring the original line for
    #    line, so every step is its own node in the plan instead of one opaque
    #    function call.
    raw_target = data["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[~raw_target.isin(problematic_classes)].reset_index(
        drop=True
    )

    # Mark the RAW target as required. Its 0-6 transformation is recorded only
    # after mark_as_y, and predictions are converted back to 1-7 below.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. A transformer is used because
    #    the original discovers columns dynamically and loops to create new named
    #    columns; this preserves the exact feature names and insertion order.
    features = X.skb.apply(AddSoilInteractions())

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Accuracy is unchanged by the label shift, but predictions must be restored
    # to the raw target domain used by mark_as_y. The operation is eval-mode gated
    # because the prediction node contains the fitted estimator during fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the StratifiedKFold on mark_as_X drives validation.
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
