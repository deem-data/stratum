import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's custom missing-file
    #    message is omitted because it is not part of producing the CV score;
    #    pd.read_csv will still raise FileNotFoundError when scoring if absent.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original data-dependent removal of classes
    #    having fewer than three samples. Row filtering must happen before the
    #    X/y marks because it changes the number of rows.
    raw_target = data["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~raw_target.isin(problematic_classes)
    filtered_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW 1-7 target first. Its 0-6 transform is downstream of the mark,
    # and predictions are converted back to 1-7 before accuracy is scored.
    y_raw = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The original manual three-fold loop becomes the splitter on mark_as_X.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — append the composite at the end, then
    #    remove the three source columns, preserving the original feature values,
    #    names, and column order.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    features = X.assign(Hillshade_composite=hillshade_composite).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predicted labels from 0-6 back to the raw target's 1-7 domain.
    # Prediction arithmetic is gated because in fit mode this node contains the
    # fitted estimator rather than an array of predictions.
    def restore_original_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the StratifiedKFold on mark_as_X drives.
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
