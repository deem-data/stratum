import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    plan; the file is not read until the guarded scoring block is executed.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — faithfully remove classes having fewer than three samples
    #    before marking X and y, because this data-dependent filtering changes
    #    which rows are cross-validated. Writing the mask unconditionally is
    #    equivalent to the original `if problematic_classes` branch: when there
    #    are no problematic classes, every row is retained.
    target_for_filtering = train_df["Cover_Type"] - 1
    class_counts = target_for_filtering.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~target_for_filtering.isin(problematic_classes)
    filtered_df = train_df[keep_rows].reset_index(drop=True)

    # Mark the RAW target. The 1-7 to 0-6 transform used for model fitting is
    # applied only after mark_as_y; predictions are converted back below so
    # accuracy is evaluated in the raw target domain.
    y = filtered_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The manual three-fold loop becomes the same StratifiedKFold splitter on
    # mark_as_X. Per-fold progress and excluded-class messages are omitted because
    # they do not contribute to the cross-validated score.
    X = filtered_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original feature names,
    #    formulas, and append order.
    features = X.assign(
        Hillshade_Std=X[
            ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
        ].std(axis=1)
    )
    features = features.assign(
        Elevation_Slope_Interaction=features["Elevation"] * features["Slope"]
    )
    features = features.assign(Elevation_sq=features["Elevation"] ** 2)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert 0-6 predictions back to the raw 1-7 labels. This is gated because
    # in fit mode the prediction node contains the fitted estimator itself.
    def restore_original_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    # 4. Score — no cv= here; the splitter declared on mark_as_X drives.
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
