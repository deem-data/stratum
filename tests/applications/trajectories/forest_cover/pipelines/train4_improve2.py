import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode, the prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's eager FileNotFoundError
    #    handling is omitted; a missing file naturally raises when the plan runs.
    train_df = (
        skrub.as_data_op("./input/train.csv")
        .skb.apply_func(pd.read_csv)
    )

    # 2. Prepare Data — classes represented by fewer than three rows were excluded
    #    from the original CV dataset. This row filtering must happen before either
    #    mark because it changes the number of samples. Cover_Type and
    #    Cover_Type - 1 have identical class counts.
    class_sizes = (
        train_df.groupby("Cover_Type")["Cover_Type"].transform("size")
    )
    cv_df = train_df[class_sizes >= 3].reset_index(drop=True)

    # Mark the RAW target. The original's 1-7 to 0-6 target transformation is
    # recorded after mark_as_y and inverted on predictions below.
    y = cv_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Keep the three source Hillshade columns until downstream feature engineering.
    # The manual StratifiedKFold loop becomes the splitter on mark_as_X.
    X = cv_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. The two new columns are appended
    #    before the original Hillshade columns are dropped, preserving the
    #    original feature values and column order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Hillshade_Morning_vs_Afternoon=(
            X["Hillshade_9am"] - X["Hillshade_3pm"]
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

    # Accuracy on restored 1-7 predictions against the raw marked target is
    # identical to the original accuracy on transformed 0-6 labels. The operation
    # is gated because fit mode contains an estimator rather than predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    # 4. Score. No cv= here — the splitter on mark_as_X drives the evaluation.
    #    Fold progress and rare-class warning printing are omitted because they do
    #    not contribute to the cross-validated score.
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
