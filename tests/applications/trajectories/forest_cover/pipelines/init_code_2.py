import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's eager FileNotFoundError
    #    handling is omitted because importing this module must not access the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original data-dependent removal of classes
    #    having fewer than three samples. Row filtering must occur before marking X
    #    and y because it changes the number of rows.
    target_zero_based_before_filter = data["Cover_Type"] - 1
    class_counts = target_zero_based_before_filter.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~target_zero_based_before_filter.isin(problematic_classes)
    cv_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target, then perform the original 1-7 to 0-6 transformation
    # downstream of mark_as_y. The manual StratifiedKFold loop becomes the splitter
    # attached to mark_as_X.
    y = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1
    X = cv_data.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Model — preserve the original model family and all hyperparameters.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_zero_based = X.skb.apply(model, y=y_transformed)

    # Convert predictions back to the raw 1-7 target domain for scoring. This must
    # be gated because in fit mode the prediction node contains the fitted estimator.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_zero_based.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — the splitter on mark_as_X drives the three-fold CV, so no cv=
    #    is passed here. Per-fold progress printing from the manual loop is dropped.
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
