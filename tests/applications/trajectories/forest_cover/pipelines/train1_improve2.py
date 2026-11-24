import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read so importing this module does not read it.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — faithfully exclude classes represented by fewer than three
    #    samples, as the original did before cross-validation. Row filtering must
    #    happen before marking X and y because it changes their number of rows.
    #    Counting raw labels is equivalent to counting labels after subtracting 1.
    n_splits = 3
    cover_type = data["Cover_Type"]
    class_counts = cover_type.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index
    data_cv = data[~cover_type.isin(problematic_classes)].reset_index(drop=True)

    # Mark the RAW target. The 1-7 to 0-6 transformation is recorded only after
    # mark_as_y; predictions are shifted back below before scoring.
    y_raw = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the raw design matrix before feature engineering. The splitter exactly
    # replaces the original manual three-fold StratifiedKFold loop.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — same features, values, and final dropped
    #    columns as the original script.
    aspect_radians = X["Aspect"].skb.apply_func(np.radians)
    features = X.assign(
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_Slope_Interaction=X["Elevation"] * X["Slope"],
    ).drop(
        columns=[
            "Aspect",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    # Same model family and all explicitly configured hyperparameters.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predicted labels from 0-6 back to the raw 1-7 target domain. This
    # preserves the original accuracy while obeying the raw-target marking rule.
    # Prediction arithmetic is gated because fit mode returns the fitted estimator.
    def restore_original_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return np.asarray(prediction) + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; mark_as_X's StratifiedKFold drives validation.
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
