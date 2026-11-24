import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map model predictions from 0–6 back to the raw 1–7 target domain."""
    if mode == "fit":
        # In fit mode, predictions is the fitted estimator rather than an array.
        return predictions
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    plan; the file is not read until the guarded scoring block runs.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes having fewer
    #    than three samples. This row filtering must occur before marking X and y
    #    because it changes their number of rows. The counts are recorded from the
    #    complete input table, matching the original script.
    n_splits = 3
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index
    keep_for_cv = ~data["Cover_Type"].isin(problematic_classes)
    cv_data = data[keep_for_cv].reset_index(drop=True)

    # Mark the RAW 1–7 target. The model target transformation is deliberately
    # downstream of mark_as_y, and predictions are converted back before scoring.
    y = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The original manual three-fold loop becomes the splitter on mark_as_X.
    X = cv_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original feature values,
    #    names, insertion order, and dropped source columns.
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)
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
        Elevation_x_Slope=X["Elevation"] * X["Slope"],
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

    # Convert 0–6 predictions back to the marked raw 1–7 target domain. This is
    # gated because prediction nodes hold fitted estimators during fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the StratifiedKFold attached to X drives validation.
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
