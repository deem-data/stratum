import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's eager FileNotFoundError
    #    message is omitted because importing this module must not access the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — exclude classes with fewer than three samples before the
    #    marks, exactly as the original does for cross-validation. Row filtering
    #    must occur before marking because it changes the number of samples.
    class_size = data.groupby("Cover_Type")["Cover_Type"].transform("size")
    cv_data = data[class_size >= 3].reset_index(drop=True)

    # Mark the RAW target first, then perform the original 1-7 to 0-6 transform
    # downstream for model fitting.
    y_raw = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the unengineered design matrix. The original manual three-fold
    # StratifiedKFold loop is represented by the splitter on mark_as_X.
    X = cv_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original feature values,
    #    names, and order, then remove the three source Hillshade columns.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"])

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predicted labels from 0-6 back to the raw 1-7 target domain used
    # for scoring. In fit mode the prediction node contains the fitted estimator,
    # so it must pass through unchanged.
    def restore_original_labels(values, mode):
        if mode == "fit":
            return values
        return np.asarray(values) + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — the splitter on mark_as_X drives validation, so no cv= is passed
    #    here. mean_test_score is the original mean of the three fold accuracies.
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
