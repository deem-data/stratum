import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_target_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode, the prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    lazy plan; the file is not read until the guarded scoring block runs.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes any class having fewer than three
    #    samples from cross-validation. Because this changes the number of rows,
    #    the same data-dependent filtering must happen before X and y are marked.
    target_zero_for_filter = data["Cover_Type"] - 1
    class_counts = target_zero_for_filter.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~target_zero_for_filter.isin(problematic_classes)
    cv_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target, then apply the original 1-7 to 0-6 transformation for
    # model fitting. Predictions are mapped back to 1-7 below before scoring.
    y_raw = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Keep the Hillshade columns until their composite feature has been created.
    # The original manual three-fold loop becomes this identical splitter.
    X = cv_data.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original derived feature
    #    names, values, insertion order, and final feature set.
    euclidean_hydrology = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    ).skb.apply_func(np.sqrt)

    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Euclidean_Distance_To_Hydrology=euclidean_hydrology,
        Hydro_Road_Proximity_Ratio=(
            X["Horizontal_Distance_To_Hydrology"]
            / (X["Horizontal_Distance_To_Roadways"] + 1e-6)
        ),
    ).drop(
        ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"],
        axis=1,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Restore predictions to the raw 1-7 target domain used by mark_as_y. The
    # eval-mode guard leaves the fitted estimator untouched during fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is passed here; mark_as_X's StratifiedKFold drives the
    #    search, and mean_test_score matches the original mean fold accuracy.
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
