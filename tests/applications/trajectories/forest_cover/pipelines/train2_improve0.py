import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError message
    #    is omitted because importing this module must not read the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes classes having fewer than three
    #    samples from cross-validation. This row filtering must occur before the
    #    marks because it changes the number of rows.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    data_cv = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target first. The model's 0-6 target transformation is recorded
    # only after mark_as_y; predictions are converted back to 1-7 below so scoring
    # remains in the raw target domain while preserving the original accuracy.
    y = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Mark the unengineered design matrix. The original manual three-fold
    # StratifiedKFold loop is represented by this splitter and nowhere else.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. assign appends the three new columns in
    #    the same order as the original script, after which the source hillshade
    #    columns are removed while the original Aspect column is retained.
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
        Roadways_Elevation_Interaction=(
            X["Horizontal_Distance_To_Roadways"] * X["Elevation"]
        ),
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert 0-6 predictions back to the marked raw target's 1-7 domain.
    # The operation is eval-mode gated because during fit the prediction node
    # contains the fitted estimator rather than an array of predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    # 4. Score. No cv= here: mark_as_X's StratifiedKFold drives validation.
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
