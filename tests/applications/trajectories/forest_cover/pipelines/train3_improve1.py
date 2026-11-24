import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError message
    #    wrapper is omitted because it is not part of producing the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes having fewer
    #    than three samples. Row filtering must occur before marking X and y
    #    because it changes their number of rows.
    n_splits = 3
    raw_target = data["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index
    keep_rows = ~raw_target.isin(problematic_classes)
    cv_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target, as required for leakage-free scoring. The label shift
    # from 1-7 to 0-6 is recorded after the mark and is used only for fitting.
    y = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Mark the unengineered design matrix early. The original manual three-fold
    # loop becomes the same StratifiedKFold splitter on mark_as_X.
    X = cv_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — average the three hillshade measurements
    #    into Hillshade_composite, retain Aspect, and remove the source columns.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    features = X.assign(
        Hillshade_composite=hillshade_composite
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    # Same model family and hyperparameters as the original script.
    model = LGBMClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predictions from the fitted 0-6 label domain back to the raw 1-7
    # domain used by mark_as_y. The conversion is gated because in fit mode the
    # prediction node contains the fitted estimator rather than predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — the splitter on mark_as_X drives the three-fold CV, so no cv=
    #    is passed here. Accuracy is unchanged by the reversible label shift and
    #    therefore matches the original mean fold accuracy.
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
