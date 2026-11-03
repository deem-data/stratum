import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except is omitted because
    #    importing this module must not read data; a missing file is reported when
    #    the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Faithfully remove classes containing fewer than three
    #    samples before marking X and y. This vectorized recorded filter reproduces
    #    both branches of the original data-dependent `if problematic_classes`.
    transformed_target = data["Cover_Type"] - 1
    class_counts = transformed_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[
        ~transformed_target.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target, then perform the original 1-7 to 0-6 transformation
    # downstream. The StratifiedKFold splitter replaces the manual outer loop.
    y_raw = filtered_data["Cover_Type"].skb.mark_as_y()
    y = y_raw - 1
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. The engineered columns are
    #    appended in the same order as in the original before the source columns
    #    are dropped, preserving the feature order seen by the random forest.
    aspect_radians = X["Aspect"].skb.apply_func(np.radians)
    features = X.assign(
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
    ).drop(
        columns=[
            "Aspect",
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
    pred_transformed = features.skb.apply(model, y=y)

    # Predictions are shifted back to the raw 1-7 target domain for scoring.
    # Accuracy is unchanged from scoring both y and predictions in the 0-6 domain.
    # The operation is gated because the prediction node contains the fitted
    # estimator rather than predictions while skrub is in fit mode.
    def restore_raw_labels(values, mode):
        if mode == "fit":
            return values
        return values + 1

    pred = pred_transformed.skb.apply_func(
        restore_raw_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here—the splitter attached to mark_as_X drives the
    #    three-fold validation and mean_test_score is the original mean accuracy.
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
