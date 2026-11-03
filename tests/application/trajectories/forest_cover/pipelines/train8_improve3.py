import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's eager FileNotFoundError
    #    message is omitted because importing this module must not read the file;
    #    a missing file will naturally raise when the plan is scored.
    train_df = (
        skrub.as_data_op("./input/train.csv")
        .skb.apply_func(pd.read_csv)
    )

    # 2. Prepare Data — reproduce the original exclusion of target classes with
    #    fewer than three samples. Row filtering must happen before the marks
    #    because it changes the number of rows participating in cross-validation.
    class_counts = train_df["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    train_df = train_df[
        ~train_df["Cover_Type"].isin(problematic_classes)
    ]

    # Mark the RAW 1-7 target. The model target is shifted only after marking so
    # scoring remains defined against the raw target.
    y = train_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Mark the raw design matrix before feature engineering. The manual
    # StratifiedKFold loop becomes the splitter attached to mark_as_X.
    X = train_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — the derived columns are appended in the
    # same order as in the original script, which preserves the feature order
    # seen by the randomized forest.
    hydrology_nonnegative = X[
        "Horizontal_Distance_To_Hydrology"
    ].skb.apply_func(np.maximum, 0)
    roadways_nonnegative = X[
        "Horizontal_Distance_To_Roadways"
    ].skb.apply_func(np.maximum, 0)
    fire_points_nonnegative = X[
        "Horizontal_Distance_To_Fire_Points"
    ].skb.apply_func(np.maximum, 0)

    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Hillshade_Noon_to_3pm_Diff=(
            X["Hillshade_Noon"] - X["Hillshade_3pm"]
        ),
        Total_Horizontal_Distance=(
            hydrology_nonnegative.skb.apply_func(np.log1p)
            + roadways_nonnegative.skb.apply_func(np.log1p)
            + fire_points_nonnegative.skb.apply_func(np.log1p)
        ),
        Slope_x_Vertical_Distance_To_Hydrology=(
            X["Slope"] * X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(
        columns=[
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

    # Convert the model's 0-6 predictions back to the raw target's 1-7 domain.
    # This leaves accuracy unchanged from the original transformed-label score.
    # Prediction arithmetic is gated because in fit mode this node contains the
    # fitted estimator rather than an array of predictions.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the StratifiedKFold on mark_as_X drives the
    # end-to-end cross-validation.
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
