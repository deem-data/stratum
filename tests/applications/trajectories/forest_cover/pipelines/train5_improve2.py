import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read so importing this module does not access the
    #    file. The original eager FileNotFoundError handling is omitted because
    #    the lazy read occurs only when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — faithfully remove classes with fewer than three samples
    #    before marking X and y. The original data-dependent if/else is expressed
    #    as recorded operations, so the filter also remains present when no class
    #    ultimately satisfies the condition.
    target_for_filtering = data["Cover_Type"] - 1
    class_counts = target_for_filtering.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~target_for_filtering.isin(problematic_classes)
    filtered_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target as required. Its 0-based transform is applied only after
    # mark_as_y. Stratifying on labels 1-7 is equivalent to stratifying on 0-6.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Mark the raw design matrix before feature engineering. The splitter exactly
    # reproduces the original manual three-fold StratifiedKFold loop.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — the generated columns are appended in the
    #    same order as in the original script.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    )
    features = features.assign(
        Elevation_at_Hydrology=(
            features["Elevation"] - features["Vertical_Distance_To_Hydrology"]
        )
    )

    aspect_radians = features["Aspect"].skb.apply_func(np.radians)
    aspect_sin = aspect_radians.skb.apply_func(np.sin)
    aspect_cos = aspect_radians.skb.apply_func(np.cos)

    features = features.assign(
        Slope_x_Aspect_sin=features["Slope"] * aspect_sin
    )
    features = features.assign(
        Slope_x_Aspect_cos=features["Slope"] * aspect_cos
    )
    features = features.drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    # Same model family and every hyperparameter explicitly set by the original.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # The original scored predictions in the transformed 0-6 label space. Because
    # the plan marks the raw 1-7 target, add one back to predictions; accuracy is
    # therefore exactly the same. Prediction arithmetic is gated because in fit
    # mode this node contains the fitted estimator rather than predicted labels.
    def restore_original_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives the
    #    cross-validation. Test/submission work is absent, as in the original.
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
