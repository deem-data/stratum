import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except is omitted:
    #    importing this module only builds the lazy plan, while a missing file is
    #    naturally reported when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes target classes represented by fewer
    #    than three rows. Because this changes the number of rows, the filtering
    #    must happen before mark_as_X and mark_as_y. Computing counts from the
    #    target before CV is intentional here because it defines which rows belong
    #    to the original evaluation dataset rather than fitting a preprocessing
    #    parameter.
    n_splits = 3
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index
    data_cv = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target. The original's 1-7 to 0-6 target conversion is recorded
    # only after mark_as_y, and predictions are converted back below for scoring.
    y_raw = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the raw design matrix. The manual three-fold loop becomes the same
    # StratifiedKFold splitter attached to mark_as_X.
    X = data_cv.drop(columns=["Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original feature values,
    #    names, append order, and dropped columns.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    elevation_at_hydrology = (
        X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    )
    hydrology_distance_squared = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    )
    total_distance_to_hydrology = hydrology_distance_squared.skb.apply_func(
        np.sqrt
    )

    features = X.assign(
        Hillshade_composite=hillshade_composite,
        Elevation_at_Hydrology=elevation_at_hydrology,
        Total_Distance_To_Hydrology=total_distance_to_hydrology,
        Slope_squared=X["Slope"] ** 2,
    ).drop(
        columns=[
            "Id",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert 0-6 predictions back to the raw 1-7 target domain. This must be
    # gated because a prediction node evaluates to the fitted estimator in fit
    # mode rather than to an array of predictions.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is passed here; mark_as_X's StratifiedKFold drives the
    #    evaluation. Test/submission work is absent, as in the original.
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
