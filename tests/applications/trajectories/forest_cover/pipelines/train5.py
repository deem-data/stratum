import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError message is
    #    omitted because it does not contribute to the cross-validated score; a
    #    missing file will still raise naturally when the plan is scored.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — classes represented by fewer than three rows are removed
    #    before marking X and y, matching the original. Row filtering is necessarily
    #    before the marks because it changes the number of samples.
    #    groupby(...).transform("size") records the same class-count test without
    #    eagerly materializing a runtime list of problematic classes.
    class_sizes = train_df.groupby("Cover_Type")["Cover_Type"].transform("size")
    filtered_df = train_df[class_sizes >= 3].reset_index(drop=True)

    # Mark the RAW target. The original's 1-7 to 0-6 transformation is applied
    # afterward for model fitting; predictions are mapped back to 1-7 below so
    # accuracy is computed in the raw target domain.
    y = filtered_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Mark the raw design matrix before feature engineering. The StratifiedKFold
    # replaces the original manual three-fold loop and lives only on mark_as_X.
    X = filtered_df.drop(columns=["Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original three derived
    #    features, then remove Id and the individual Hillshade columns.
    hydrology_squared_distance = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    )
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Euclidean_Distance_To_Hydrology=hydrology_squared_distance.skb.apply_func(
            np.sqrt
        ),
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(
        columns=[
            "Id",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    # Same model family and hyperparameters as the original.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predicted labels from 0-6 back to the raw 1-7 target domain. In fit
    # mode a prediction node contains the fitted estimator, so it must pass through
    # unchanged rather than being used in arithmetic.
    def restore_original_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here: the splitter declared on mark_as_X drives the search.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="accuracy",
        )
        print(search.results_)
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
