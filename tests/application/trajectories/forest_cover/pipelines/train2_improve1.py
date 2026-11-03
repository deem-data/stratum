import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    lazy plan and does not access the file. The original try/except and
    #    progress messages are omitted because they do not produce the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original exclusion of target classes with
    #    fewer than three samples. This target-dependent row filtering must happen
    #    before marking X and y because it changes their number of rows.
    n_splits = 3
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index
    keep_rows = ~data["Cover_Type"].isin(problematic_classes)
    data_cv = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target. The 1-7 to 0-6 target transformation is performed only
    # after mark_as_y, and predictions are shifted back below before scoring.
    y = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Mark the raw design matrix before feature engineering. StratifiedKFold sees
    # the raw 1-7 labels, which produces exactly the same strata as labels 0-6.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original feature values,
    #    names, insertion order, and removal of the three Hillshade columns.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3

    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)
    sin_aspect = aspect_radians.skb.apply_func(np.sin)
    cos_aspect = aspect_radians.skb.apply_func(np.cos)

    X_features = X.assign(
        Hillshade_composite=hillshade_composite,
        sin_Aspect=sin_aspect,
        cos_Aspect=cos_aspect,
        Elevation_Slope_Interaction=X["Elevation"] * X["Slope"],
        Total_Horizontal_Proximity=(
            X["Horizontal_Distance_To_Hydrology"]
            + X["Horizontal_Distance_To_Roadways"]
            + X["Horizontal_Distance_To_Fire_Points"]
        ),
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    # Same model family and hyperparameters as the original script.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = X_features.skb.apply(model, y=y_transformed)

    # Convert predictions from 0-6 back to the raw target's 1-7 domain. This is
    # gated because in fit mode the prediction node contains the fitted estimator.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives the same
    #    three-fold validation as the original manual loop.
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
