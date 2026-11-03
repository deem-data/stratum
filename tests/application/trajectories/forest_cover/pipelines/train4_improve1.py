import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    lazy plan; the file is read when the guarded scoring block is executed.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes having fewer
    #    than three samples. Row filtering must occur before marking X and y
    #    because it changes their number of rows. Fold-progress and warning
    #    printing from the eager script are omitted because they do not produce
    #    the cross-validated score.
    cover_type_counts = data["Cover_Type"].value_counts()
    rare_cover_types = cover_type_counts[cover_type_counts < 3].index
    data_cv = data[
        ~data["Cover_Type"].isin(rare_cover_types)
    ].reset_index(drop=True)

    # Mark the RAW target. The original's 1-7 to 0-6 target transformation is
    # recorded after the mark, and predictions are mapped back to 1-7 below so
    # scoring remains in the raw target domain.
    y_raw = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Keep the Hillshade columns at the mark because they are inputs to recorded
    # feature engineering below. Id and the target are not model features.
    # Stratifying raw labels is equivalent to stratifying labels shifted by one.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model — preserve the original feature
    #    values, names, insertion order, dropped columns, model family, and all
    #    explicitly supplied hyperparameters.
    hydrology_distance_squared = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    )

    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Euclidean_Distance_To_Hydrology=hydrology_distance_squared.skb.apply_func(
            np.sqrt
        ),
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
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

    # Convert predicted labels from 0-6 back to the raw 1-7 target domain.
    # In fit mode, a prediction node contains the fitted estimator rather than
    # predictions, so it must pass through unchanged.
    def restore_cover_type_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_cover_type_labels,
        skrub.eval_mode(),
    )

    # 4. Score — the StratifiedKFold attached to mark_as_X drives validation.
    #    Accuracy and the arithmetic mean over the three folds match the original.
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
