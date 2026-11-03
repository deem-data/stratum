import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Invert the target's 1-to-0 label shift only when producing predictions."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded, so importing this module does not access
    #    the file. The original FileNotFoundError printing is omitted because it is
    #    not part of producing the cross-validated score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — faithfully remove target classes having fewer than three
    #    rows before marking X and y. This data-dependent filtering cannot be
    #    omitted, even when the problematic-class list happens to be empty.
    raw_target = data["Cover_Type"]
    shifted_target_for_filtering = raw_target - 1
    class_counts = shifted_target_for_filtering.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~shifted_target_for_filtering.isin(problematic_classes)
    filtered_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target. The 1-7 to 0-6 transformation used for fitting is
    # applied only after mark_as_y and is inverted on predictions below.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The manual three-fold loop becomes the identical StratifiedKFold splitter
    # attached to mark_as_X. Stratifying raw labels is equivalent to stratifying
    # their one-to-one shifted labels.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original creation order:
    #    Hillshade_composite, temporary Aspect_rad, Aspect_sin, then Aspect_cos.
    X_features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    )
    aspect_rad = X_features["Aspect"].skb.apply_func(np.deg2rad)
    X_features = X_features.assign(Aspect_rad=aspect_rad)
    X_features = X_features.assign(
        Aspect_sin=X_features["Aspect_rad"].skb.apply_func(np.sin)
    )
    X_features = X_features.assign(
        Aspect_cos=X_features["Aspect_rad"].skb.apply_func(np.cos)
    )
    X_features = X_features.drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Aspect_rad",
        ]
    )

    # Same model family and every hyperparameter from the original script.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_shifted = X_features.skb.apply(model, y=y_transformed)

    # Restore predictions from 0-6 to the raw target's 1-7 domain. This preserves
    # the original transformed-label accuracy while satisfying raw-target scoring.
    # The operation is gated because prediction nodes hold an estimator in fit mode.
    pred = pred_shifted.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives.
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
