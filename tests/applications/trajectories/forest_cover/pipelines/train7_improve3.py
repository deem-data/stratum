import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Shift model predictions from 0-6 back to the raw 1-7 label domain."""
    if mode == "fit":
        # In fit mode, the prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original error message and progress
    #    printing are omitted because they do not contribute to the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Faithfully remove classes with fewer than three rows before
    #    marking X and y, since the original excluded those rows from scoring.
    #    Computing this from raw Cover_Type is equivalent to computing it after
    #    subtracting one because the shift does not change class frequencies.
    raw_target = data["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[
        ~raw_target.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target. The original's 0-6 target transformation is applied
    # only after marking, and predictions are shifted back before scoring.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    X = filtered_data.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )
    y_transformed = y - 1

    # 3. Recorded preprocessing and feature engineering, preserving the original
    #    feature values, names, and creation order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    )
    features = features.assign(
        Elevation_at_Hydrology=(
            features["Elevation"]
            - features["Vertical_Distance_To_Hydrology"]
        )
    )

    transformed_road_distance = (
        features["Horizontal_Distance_To_Roadways"]
        .clip(0, None)
        .skb.apply_func(np.log1p)
    )
    features = features.assign(
        Horizontal_Distance_To_Roadways=transformed_road_distance
    )
    features = features.assign(
        Hydro_Road_Interaction=(
            features["Horizontal_Distance_To_Hydrology"]
            * features["Horizontal_Distance_To_Roadways"]
        ),
        Elevation_x_Fire_Points=(
            features["Elevation"]
            * features["Horizontal_Distance_To_Fire_Points"]
        ),
    )
    features = features.drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # The original model predicts 0-6 labels and scores those against the shifted
    # target. Shift predictions back by one so scoring against the marked raw
    # 1-7 target remains exactly equivalent.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    # 4. Score. No cv= here—the splitter attached to mark_as_X drives validation.
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
