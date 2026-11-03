import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except is omitted because
    #    importing this module must not access the file; a missing file is reported
    #    when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded classes having fewer than three
    #    observations from cross-validation. This row filtering is retained before
    #    the marks because it changes the number of rows.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    data_cv = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target, as required for scoring in the original target domain.
    # The model is trained on labels shifted from 1-7 to 0-6, matching the original.
    y_raw = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the unprocessed design matrix. The original manual three-fold loop is
    # represented by the identical StratifiedKFold splitter here.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Derived columns are appended in exactly the
    #    same order as in the original script before their source columns are
    #    removed.
    aspect_radians = X["Aspect"].skb.apply_func(np.radians)
    aspect_sin = aspect_radians.skb.apply_func(np.sin)
    aspect_cos = aspect_radians.skb.apply_func(np.cos)

    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3

    hydrology_squared_distance = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    )
    hydrology_euclidean = hydrology_squared_distance.skb.apply_func(np.sqrt)

    features = X.assign(
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
        Hillshade_composite=hillshade_composite,
        Distance_To_Hydrology_Euclidean=hydrology_euclidean,
    ).drop(
        columns=[
            "Aspect",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predicted labels from 0-6 back to the raw 1-7 domain used by the
    # marked target. In fit mode the prediction node contains the fitted estimator,
    # so it must be returned unchanged rather than used in arithmetic.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is passed here; the splitter on mark_as_X drives the
    #    validation and mean_test_score is the original mean fold accuracy.
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
