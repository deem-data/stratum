import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    lazy plan and does not read the file.
    train_df = (
        skrub.as_data_op("./input/train.csv")
        .skb.apply_func(pd.read_csv)
    )

    # 2. Prepare Data — reproduce the original removal of classes containing
    #    fewer than three samples. Row filtering must happen before the marks
    #    because it changes the number of rows participating in cross-validation.
    target_for_filtering = train_df["Cover_Type"] - 1
    class_counts = target_for_filtering.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~target_for_filtering.isin(problematic_classes)
    filtered_df = train_df[keep_rows].reset_index(drop=True)

    # Mark the RAW target. The original's 1-7 to 0-6 transformation is applied
    # only after mark_as_y; predictions are converted back below for raw-domain
    # accuracy scoring.
    y_raw = filtered_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the raw design matrix before feature engineering. The manual
    # StratifiedKFold loop is represented by the splitter on mark_as_X.
    X = filtered_df.drop(columns=["Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — the formulas and resulting column order
    #    match the original script.
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)
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
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
        Euclidean_Distance_To_Hydrology=euclidean_hydrology,
        Slope_x_Euclidean_Distance_To_Hydrology=(
            X["Slope"] * euclidean_hydrology
        ),
    ).drop(
        columns=[
            "Id",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Accuracy is unchanged by shifting both true and predicted labels by one.
    # Convert predictions back to the raw 1-7 target domain because y_raw is the
    # marked scoring target. The operation is gated because prediction nodes
    # contain the fitted estimator while skrub is in fit mode.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives the
    #    three-fold validation.
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
