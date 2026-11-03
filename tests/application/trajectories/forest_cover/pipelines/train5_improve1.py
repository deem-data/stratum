import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's try/except and progress
    #    messages are omitted because they do not contribute to the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Faithfully remove classes with fewer than three samples
    #    before marking X and y, because this changes which rows are scored.
    #    The recorded frequency logic also covers the original branch where no
    #    problematic classes exist: in that case the mask retains every row.
    target_for_counts = data["Cover_Type"] - 1
    class_counts = target_for_counts.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[
        ~target_for_counts.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW 1-7 target. The model target is shifted only after the mark,
    # and predictions are shifted back before scoring, preserving the original
    # accuracy computed in the 0-6 label domain.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original formulas and
    #    appended-column order.
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)
    horizontal_hydrology = X["Horizontal_Distance_To_Hydrology"]
    vertical_hydrology = X["Vertical_Distance_To_Hydrology"]

    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Euclidean_Distance_To_Hydrology=(
            horizontal_hydrology**2 + vertical_hydrology**2
        ).skb.apply_func(np.sqrt),
        Elevation_at_Hydrology=X["Elevation"] - vertical_hydrology,
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

    # The original scored predictions in the shifted 0-6 domain. Since skrub
    # scores against the marked raw 1-7 target, add one to predictions outside
    # fit mode; this produces exactly the same per-row accuracy.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here: the StratifiedKFold on mark_as_X drives CV.
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
