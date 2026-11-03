import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_cover_type_labels(predictions, mode):
    """Map zero-based model predictions back to the raw 1-based target domain."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except is omitted;
    #    a missing file naturally raises FileNotFoundError when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Faithfully remove target classes containing fewer than
    #    three rows before marking X and y, because the original excluded those
    #    rows from every scored fold. Subtracting one does not change frequencies.
    raw_target = data["Cover_Type"]
    transformed_target = raw_target - 1
    class_counts = transformed_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[
        ~transformed_target.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target as required, then perform the original 1-based to
    # 0-based label transform downstream of the mark.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Keep the raw hillshade and Aspect columns temporarily because feature
    # engineering uses the hillshade columns. Each variant drops the same raw
    # columns before fitting, matching base_drop_cols from the original.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    model_drop_cols = [
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Aspect",
    ]

    # 3. Recorded preprocessing and feature engineering. Shared expressions are
    #    recorded once, while assignments below preserve each original variant's
    #    generated-column order, which matters for a randomized forest.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    elevation_at_hydrology = (
        X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    )
    hillshade_noon_to_3pm_diff = (
        X["Hillshade_Noon"] - X["Hillshade_3pm"]
    )

    hydrology_distance_log = (
        X["Horizontal_Distance_To_Hydrology"]
        .skb.apply_func(np.maximum, 0)
        .skb.apply_func(np.log1p)
    )
    roadway_distance_log = (
        X["Horizontal_Distance_To_Roadways"]
        .skb.apply_func(np.maximum, 0)
        .skb.apply_func(np.log1p)
    )
    fire_points_distance_log = (
        X["Horizontal_Distance_To_Fire_Points"]
        .skb.apply_func(np.maximum, 0)
        .skb.apply_func(np.log1p)
    )
    total_horizontal_distance = (
        hydrology_distance_log
        + roadway_distance_log
        + fire_points_distance_log
    )
    slope_x_vertical_distance = (
        X["Slope"] * X["Vertical_Distance_To_Hydrology"]
    )

    # Baseline: all five engineered features, appended in the original order.
    baseline_features = X.assign(
        Hillshade_composite=hillshade_composite,
    ).assign(
        Elevation_at_Hydrology=elevation_at_hydrology,
    ).assign(
        Hillshade_Noon_to_3pm_Diff=hillshade_noon_to_3pm_diff,
    ).assign(
        Total_Horizontal_Distance=total_horizontal_distance,
    ).assign(
        Slope_x_Vertical_Distance_To_Hydrology=slope_x_vertical_distance,
    ).drop(columns=model_drop_cols)

    baseline_pred_zero_based = baseline_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    baseline_pred = baseline_pred_zero_based.skb.apply_func(
        restore_cover_type_labels,
        skrub.eval_mode(),
    )

    # Ablation 1: omit Hillshade_Noon_to_3pm_Diff.
    without_hillshade_diff_features = X.assign(
        Hillshade_composite=hillshade_composite,
    ).assign(
        Elevation_at_Hydrology=elevation_at_hydrology,
    ).assign(
        Total_Horizontal_Distance=total_horizontal_distance,
    ).assign(
        Slope_x_Vertical_Distance_To_Hydrology=slope_x_vertical_distance,
    ).drop(columns=model_drop_cols)

    without_hillshade_diff_pred_zero_based = (
        without_hillshade_diff_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        )
    )
    without_hillshade_diff_pred = (
        without_hillshade_diff_pred_zero_based.skb.apply_func(
            restore_cover_type_labels,
            skrub.eval_mode(),
        )
    )

    # Ablation 2: omit Total_Horizontal_Distance.
    without_total_distance_features = X.assign(
        Hillshade_composite=hillshade_composite,
    ).assign(
        Elevation_at_Hydrology=elevation_at_hydrology,
    ).assign(
        Hillshade_Noon_to_3pm_Diff=hillshade_noon_to_3pm_diff,
    ).assign(
        Slope_x_Vertical_Distance_To_Hydrology=slope_x_vertical_distance,
    ).drop(columns=model_drop_cols)

    without_total_distance_pred_zero_based = (
        without_total_distance_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        )
    )
    without_total_distance_pred = (
        without_total_distance_pred_zero_based.skb.apply_func(
            restore_cover_type_labels,
            skrub.eval_mode(),
        )
    )

    # Ablation 3: omit Slope_x_Vertical_Distance_To_Hydrology.
    without_slope_interaction_features = X.assign(
        Hillshade_composite=hillshade_composite,
    ).assign(
        Elevation_at_Hydrology=elevation_at_hydrology,
    ).assign(
        Hillshade_Noon_to_3pm_Diff=hillshade_noon_to_3pm_diff,
    ).assign(
        Total_Horizontal_Distance=total_horizontal_distance,
    ).drop(columns=model_drop_cols)

    without_slope_interaction_pred_zero_based = (
        without_slope_interaction_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        )
    )
    without_slope_interaction_pred = (
        without_slope_interaction_pred_zero_based.skb.apply_func(
            restore_cover_type_labels,
            skrub.eval_mode(),
        )
    )

    # The original scores exactly these four ablation-study variants. Fuse them
    # into one discrete choice so grid search returns one row per variant.
    # Predictions are restored to the raw 1-based Cover_Type domain before
    # scoring against the raw mark_as_y node.
    pred = skrub.choose_from(
        {
            "Baseline": baseline_pred,
            "Hillshade_Noon_to_3pm_Diff": without_hillshade_diff_pred,
            "Total_Horizontal_Distance": without_total_distance_pred,
            "Slope_x_Vertical_Distance_To_Hydrology": (
                without_slope_interaction_pred
            ),
        },
        name="variant",
    ).as_data_op()

    # 4. Score. No cv= here: mark_as_X's StratifiedKFold drives validation.
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
