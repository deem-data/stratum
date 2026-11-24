import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map model predictions from 0–6 back to the raw target domain 1–7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the CSV read. Importing this module only builds the
    #    lazy plan, so the file is not read until the scoring block is executed.
    #    The original try/except is omitted; a missing file still raises naturally.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes every row whose target class has
    #    fewer than three samples before cross-validation. Row filtering must be
    #    recorded before marking X and y because it changes the number of rows.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    data_cv = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target as required. The original's 1–7 to 0–6 transform is
    # applied only after marking; predictions are mapped back to 1–7 below so
    # accuracy is evaluated in the raw target domain with identical semantics.
    y = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The original manual three-fold loop becomes the same StratifiedKFold
    # splitter on mark_as_X. Identifier and target columns remain excluded.
    X = data_cv.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Model -- preserve the original model family and all hyperparameters.
    pred_transformed = X.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )

    # Restore predictions to the raw 1–7 target domain. This is gated because a
    # prediction node evaluates to the fitted estimator while in fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. The splitter on mark_as_X drives the three-fold validation, so
    #    no cv= is passed here. Progress-only fold printing is intentionally
    #    omitted; mean_test_score is the original mean fold accuracy.
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
