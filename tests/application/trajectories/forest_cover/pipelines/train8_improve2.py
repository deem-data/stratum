import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    plan; the file is read when the guarded scoring block executes.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded every class with fewer than three
    #    samples before cross-validation. Row filtering changes the number of
    #    samples, so it must remain before mark_as_X/mark_as_y.
    class_sizes = data.groupby("Cover_Type")["Cover_Type"].transform("size")
    data = data[class_sizes >= 3].reset_index(drop=True)

    # Mark the RAW 1-7 target. The model's 0-6 target transformation happens
    # downstream of mark_as_y, and predictions are shifted back before scoring.
    y_raw = data["Cover_Type"].skb.mark_as_y()
    y = y_raw - 1

    # Feature engineering is moved after the mark so every fold reruns it. This
    # is equivalent to the original because all transformations are row-local.
    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    elevation_at_hydrology = (
        X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    )
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)

    distance_columns = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    log_distances = (
        X[distance_columns]
        .clip(lower=0)
        .skb.apply_func(np.log1p)
    )

    features = X.assign(
        Hillshade_composite=hillshade_composite,
        Elevation_at_Hydrology=elevation_at_hydrology,
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
        Horizontal_Distance_To_Hydrology=log_distances[
            "Horizontal_Distance_To_Hydrology"
        ],
        Horizontal_Distance_To_Roadways=log_distances[
            "Horizontal_Distance_To_Roadways"
        ],
        Horizontal_Distance_To_Fire_Points=log_distances[
            "Horizontal_Distance_To_Fire_Points"
        ],
        Elevation_Times_Slope=X["Elevation"] * X["Slope"],
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
    pred_zero_indexed = features.skb.apply(model, y=y)

    # Accuracy is unchanged by shifting both true and predicted labels, but skrub
    # scores against the raw marked target. Shift predictions back from 0-6 to
    # 1-7, gated because a prediction node contains the fitted estimator in fit
    # mode rather than an array of predictions.
    def restore_original_labels(values, mode):
        if mode == "fit":
            return None
        return np.asarray(values) + 1

    pred = pred_zero_indexed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here—the splitter attached to mark_as_X drives the
    #    original three-fold stratified validation scheme.
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
