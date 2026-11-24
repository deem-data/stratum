import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map predictions from 0–6 back to the raw target domain 1–7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's custom FileNotFoundError
    #    message is omitted; the recorded read naturally raises if the file is absent.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — faithfully remove classes with fewer than three samples
    #    before marking X and y. Applying this recorded mask unconditionally is
    #    equivalent to the original data-dependent `if problematic_classes` branch:
    #    when there are no problematic classes, the mask retains every row.
    transformed_target = data["Cover_Type"] - 1
    class_counts = transformed_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[
        ~transformed_target.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target, then apply the original 1–7 to 0–6 transformation for
    # model fitting. Predictions are mapped back to the raw domain below before
    # scoring, preserving the original accuracy exactly.
    y_raw = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the unengineered design matrix. Stratifying on raw labels 1–7 gives
    # the same folds as stratifying on their one-to-one shifted labels 0–6.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Assigning the new columns before dropping
    # the three source hillshade columns preserves the original feature values,
    # names, and column order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Elevation_times_Slope=X["Elevation"] * X["Slope"],
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"])

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # The estimator predicts transformed labels 0–6, while scoring uses the raw
    # mark_as_y target 1–7. Restore the original label domain only during
    # prediction; in fit mode a prediction node contains the fitted estimator.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score using the splitter attached to mark_as_X. No cv= is passed here.
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
