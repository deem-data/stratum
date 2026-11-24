import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original custom FileNotFoundError
    #    message is omitted because it does not contribute to the CV score.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data
    #    The original excludes target classes represented by fewer than three
    #    samples. This data-dependent row filtering must occur before marking X
    #    and y because it changes their number of rows.
    n_splits = 3
    class_counts = train_df["Cover_Type"].value_counts()
    valid_classes = class_counts[class_counts >= n_splits].index
    train_df = train_df[
        train_df["Cover_Type"].isin(valid_classes)
    ].reset_index(drop=True)

    # Mark the RAW target, as required for scoring. The model is trained on the
    # original script's transformed labels (1-7 shifted to 0-6), and predictions
    # are mapped back to the raw label domain below.
    y_raw = train_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the unprocessed design matrix. The original manual three-fold loop is
    # represented by the same StratifiedKFold splitter on mark_as_X.
    X = train_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original calculations,
    #    resulting column names, and column order. Newly assigned features are
    #    appended before the original Hillshade and Aspect columns are dropped.
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)

    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
        Slope_Hydro_Interaction=(
            X["Slope"] * X["Horizontal_Distance_To_Hydrology"]
        ),
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

    # Map predicted labels from 0-6 back to the raw 1-7 scoring domain. In fit
    # mode a prediction node contains the fitted estimator, so it must pass
    # through unchanged rather than having arithmetic applied to it.
    def restore_original_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives the
    #    validation. mean_test_score is the original mean fold accuracy.
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
