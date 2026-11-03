import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_target_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw target domain, 1-7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except is omitted
    # because importing this module must not read the file; a missing file is
    # reported when the guarded scoring block evaluates the recorded read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes classes represented by fewer than
    # three rows before cross-validation. Row filtering must occur before the
    # marks because it changes the number of rows. Computing each row's class
    # size with groupby/transform is equivalent to the original value_counts,
    # problematic-class list, and index-dropping branch.
    class_sizes = data.groupby("Cover_Type")["Cover_Type"].transform("size")
    cv_data = data[class_sizes >= 3].reset_index(drop=True)

    # Mark the RAW 1-7 target, then apply the original 0-6 transformation inside
    # the recorded plan. Predictions are mapped back to 1-7 below so accuracy is
    # scored against this raw mark_as_y node in the correct target domain.
    y_raw = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the unengineered design matrix. The splitter replaces the original
    # manual three-fold loop and is declared nowhere else.
    X = cv_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original calculations and
    # column-creation order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    )

    roadway = (
        features["Horizontal_Distance_To_Roadways"]
        .clip(lower=0)
        .skb.apply_func(np.log1p)
    )
    aspect_radians = features["Aspect"].skb.apply_func(np.radians)

    features = features.assign(
        Horizontal_Distance_To_Roadways=roadway,
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
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

    # Restore predictions to the raw 1-7 target domain. This must be gated on
    # eval_mode because during fitting the prediction node contains the fitted
    # estimator rather than predicted labels.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. The StratifiedKFold attached to mark_as_X drives validation, so
    # no cv= is passed here. Per-fold progress printing is intentionally dropped.
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
