import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_target(predictions, mode):
    """Map predicted labels from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode, a prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except is omitted:
    #    the recorded read naturally raises FileNotFoundError during scoring if
    #    "./input/train.csv" is unavailable.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — exclude classes having fewer than three samples, exactly
    #    as in the original. Row filtering occurs before marking X and y because
    #    it changes their number of rows. reset_index reproduces the original
    #    X_cv/y_cv index reset.
    n_splits = 3
    class_counts = data["Cover_Type"].value_counts()
    eligible_rows = data["Cover_Type"].map(class_counts) >= n_splits
    cv_data = data[eligible_rows].reset_index(drop=True)

    # Mark the RAW 1-7 target. The 0-6 transformation used for model fitting is
    # performed after mark_as_y and is inverted on predictions below.
    y_raw = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the raw design matrix early. The original manual three-fold loop is
    # represented by the same StratifiedKFold splitter on mark_as_X.
    X = cv_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original feature values,
    # names, and order, then remove the three source Hillshade columns while
    # retaining the original Aspect column.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"])

    # Same model family and every hyperparameter value from the original.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predictions from 0-6 back to the marked raw target's 1-7 domain.
    # This is gated on eval_mode because during fitting the prediction node holds
    # the fitted estimator rather than an array of predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_target,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter on mark_as_X drives validation.
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
