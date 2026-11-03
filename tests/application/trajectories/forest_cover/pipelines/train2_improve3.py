import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    plan; a missing file is reported naturally when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — preserve the original exclusion of target classes having
    #    fewer than three samples. Row filtering must occur before the marks
    #    because it changes the number of rows participating in cross-validation.
    target_zero_based = data["Cover_Type"] - 1
    class_counts = target_zero_based.value_counts()
    problematic_classes = class_counts[class_counts < 3].index.tolist()
    keep_rows = ~target_zero_based.isin(problematic_classes)
    filtered_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target, then perform the original 1-7 to 0-6 transformation
    # downstream. Stratification is unchanged by this one-to-one label shift.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The original manual three-fold loop becomes the splitter on mark_as_X.
    # Feature engineering remains downstream so it is recorded per fold.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. The assignments are kept in the
    #    original order because RandomForest feature subsampling can depend on
    #    column order.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    features = X.assign(Hillshade_composite=hillshade_composite)

    aspect_rad = features["Aspect"].skb.apply_func(np.deg2rad)
    features = features.assign(Aspect_rad=aspect_rad)

    aspect_sin = features["Aspect_rad"].skb.apply_func(np.sin)
    features = features.assign(Aspect_sin=aspect_sin)

    aspect_cos = features["Aspect_rad"].skb.apply_func(np.cos)
    features = features.assign(Aspect_cos=aspect_cos)

    features = features.assign(
        Slope_Aspect_sin=features["Slope"] * features["Aspect_sin"]
    )
    features = features.assign(
        Slope_Aspect_cos=features["Slope"] * features["Aspect_cos"]
    )

    features = features.drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Aspect_rad",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_zero_based = features.skb.apply(model, y=y_transformed)

    # The target was marked in its raw 1-7 domain, so map predictions back to
    # that domain. Prediction arithmetic must be gated because in fit mode the
    # node contains the fitted estimator rather than predicted labels.
    def restore_original_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_zero_based.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the StratifiedKFold on mark_as_X drives.
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
