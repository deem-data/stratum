import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_target(predictions, mode):
    """Map predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's FileNotFoundError
    #    message wrapper is omitted; a missing file still raises naturally when
    #    the lazily recorded plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Faithfully remove classes having fewer than three rows
    #    before marking X and y, because this changes which rows are scored.
    #    Counting raw Cover_Type values selects exactly the same rows as counting
    #    Cover_Type - 1 because subtracting one is a one-to-one relabeling.
    raw_target = data["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[
        ~raw_target.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target, then perform the original 1-7 to 0-6 transformation
    # downstream. Predictions are mapped back to 1-7 before scoring below.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Keep the feature-engineering source columns until after mark_as_X so all
    # preprocessing is recorded and rerun within each fold.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. The new columns are appended in the same
    #    order as in the original script, after which the original source
    #    Hillshade and Aspect columns are removed.
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
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

    # Restore the original 1-7 target domain for scoring against the raw marked
    # target. In fit mode, the prediction node contains the fitted estimator, so
    # it must pass through unchanged.
    pred = pred_transformed.skb.apply_func(
        restore_original_target,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here—the StratifiedKFold attached to mark_as_X drives
    #    validation, and mean_test_score is the original mean fold accuracy.
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
