import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_target_labels(predictions, mode):
    """Shift predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded, so importing this module does not read
    #    the file. The original FileNotFoundError message is omitted because it
    #    does not contribute to the cross-validated score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — faithfully remove classes having fewer than three samples.
    #    This content-dependent row filtering must happen before marking X and y.
    #    Applying the filter unconditionally reproduces both branches of the
    #    original: when there are no problematic classes, `problematic_classes`
    #    is empty and every row is retained.
    target_for_filtering = data["Cover_Type"] - 1
    class_counts = target_for_filtering.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~target_for_filtering.isin(problematic_classes)
    filtered_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target as required. The original trained and scored on labels
    # shifted from 1-7 to 0-6, so that transform is recorded after mark_as_y and
    # predictions are shifted back to the raw domain below.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The original manual three-fold loop becomes the same StratifiedKFold
    # splitter attached to mark_as_X.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — the new columns are appended in the same
    #    order as in the original script.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    elevation_at_hydrology = (
        X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    )
    hillshade_daily_difference = (
        X["Hillshade_9am"] - X["Hillshade_3pm"]
    ).skb.apply_func(np.abs)
    total_horizontal_distance = (
        X["Horizontal_Distance_To_Hydrology"]
        + X["Horizontal_Distance_To_Roadways"]
        + X["Horizontal_Distance_To_Fire_Points"]
    )
    log1p_total_horizontal_distance = total_horizontal_distance.skb.apply_func(
        np.log1p
    )
    elevation_x_vertical_hydrology = (
        X["Elevation"] * X["Vertical_Distance_To_Hydrology"]
    )

    features = X.assign(
        Hillshade_composite=hillshade_composite,
        Elevation_at_Hydrology=elevation_at_hydrology,
        Hillshade_Daily_Difference=hillshade_daily_difference,
        log1p_Total_Horizontal_Distance=log1p_total_horizontal_distance,
        Elevation_x_Vertical_Hydrology=elevation_x_vertical_hydrology,
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
    )

    # Same model family and every hyperparameter explicitly set by the original.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # The marked target is raw (1-7), so restore predictions to that domain.
    # This preserves exactly the original accuracy because shifting both true
    # labels and predictions by one does not change equality.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives the
    #    evaluation. mean_test_score is the original mean fold accuracy.
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
