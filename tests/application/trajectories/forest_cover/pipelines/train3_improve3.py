import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except only added an
    #    error message; the recorded read will still raise FileNotFoundError.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — faithfully remove classes having fewer than three rows
    #    before marking X and y, since this changes which rows are scored.
    #    Applying the filter unconditionally also reproduces the original
    #    `if problematic_classes` branch when that set is empty.
    target_for_filter = data["Cover_Type"] - 1
    class_counts = target_for_filter.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[
        ~target_for_filter.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the raw 1-7 target, then perform the original 0-6 transformation
    # downstream of the mark.
    y_raw = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Retain the Hillshade columns at the mark so their feature engineering is
    # fitted and evaluated inside each fold. The manual three-fold loop becomes
    # this identical StratifiedKFold splitter.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model — create the composite at the
    #    end of the table, then drop the three source columns exactly as before.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    features = X.assign(
        Hillshade_composite=hillshade_composite
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    model = LGBMClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        num_leaves=25,
        min_child_samples=30,
        colsample_bytree=0.8,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Accuracy was originally computed in the transformed 0-6 label domain.
    # Restore predictions to the raw 1-7 domain because the raw target is marked
    # for scoring. This leaves the accuracy numerically unchanged.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
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
