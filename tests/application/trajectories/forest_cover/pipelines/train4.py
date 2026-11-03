import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read without reading anything at import time.
    #    The original eager try/except is omitted; a missing file is reported when
    #    the plan is scored, because importing this module must not access the data.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes having fewer
    #    than three samples. Row filtering must happen before the marks because it
    #    changes the number of rows. Counting raw Cover_Type values is equivalent
    #    to counting the shifted labels used by the original.
    n_splits = 3
    class_counts = data["Cover_Type"].value_counts()
    keep_rows = data["Cover_Type"].map(class_counts) >= n_splits
    data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target. The 1-to-7 -> 0-to-6 transformation is performed only
    # after mark_as_y, and predictions are shifted back before scoring.
    y_raw = data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The manual StratifiedKFold loop becomes the splitter on mark_as_X.
    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — create the same composite feature, retain
    #    Aspect, and remove the three original Hillshade columns.
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

    # Accuracy is evaluated against the raw marked target, so restore predictions
    # from 0-to-6 to 1-to-7. Prediction post-processing must be gated because in
    # fit mode the prediction node contains the fitted estimator.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return None
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives CV.
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
