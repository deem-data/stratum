import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode, the prediction node contains the fitted estimator.
        return None
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except is omitted:
    #    the recorded read naturally raises FileNotFoundError if the file is absent.
    train_df = (
        skrub.as_data_op("./input/train.csv")
        .skb.apply_func(pd.read_csv)
    )

    # 2. Prepare Data — reproduce the original removal of classes containing
    #    fewer than three samples. Row filtering must happen before marking X and
    #    y because it changes the number of rows. Counts are computed lazily from
    #    the loaded table when the plan is evaluated.
    n_splits = 3
    class_counts = train_df["Cover_Type"].value_counts()
    eligible_rows = train_df["Cover_Type"].map(class_counts) >= n_splits
    cv_df = train_df[eligible_rows].reset_index(drop=True)

    # Mark the RAW 1-7 target. The original's 0-6 target transformation is
    # recorded only after mark_as_y, and predictions are mapped back below so
    # accuracy is evaluated in the raw target domain.
    y_raw = cv_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The original manual StratifiedKFold loop becomes the splitter attached to
    # mark_as_X. Id and the target are excluded exactly as in the original.
    X = cv_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — create the same composite feature and
    # then remove the three source Hillshade columns while retaining Aspect.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    # Same model family and every hyperparameter value from the original.
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Accuracy is unchanged by consistently shifting labels, but predictions
    # must be returned to 1-7 because the marked scoring target is raw.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter on mark_as_X drives validation.
    #    Per-fold progress printing from the manual loop is intentionally dropped.
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
