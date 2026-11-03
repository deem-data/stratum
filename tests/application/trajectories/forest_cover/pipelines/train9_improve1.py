import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_target_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw target domain, 1-7."""
    if mode == "fit":
        # In fit mode, the prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    lazy plan; the file is read when the guarded scoring block runs.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes having fewer
    #    than three samples. Row filtering must occur before the X/y marks because
    #    it changes the number of rows. The counts and mask remain recorded
    #    operations and are evaluated when the plan runs.
    target_for_filtering = train_df["Cover_Type"] - 1
    class_counts = target_for_filtering.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~target_for_filtering.isin(problematic_classes)
    cv_df = train_df[keep_rows].reset_index(drop=True)

    # Mark the RAW target, as required for scoring in the original 1-7 domain.
    # The model target is transformed afterward to reproduce the original fit.
    y_raw = cv_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the design matrix before recorded feature engineering. The original
    # manual three-fold loop becomes the same StratifiedKFold splitter here.
    X = cv_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and preprocessing — preserve the original
    #    formulas, dropped columns, feature names, and resulting column order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predictions from 0-6 back to the raw 1-7 target domain. This is
    # gated on eval mode because during fitting pred_transformed evaluates to
    # the fitted estimator rather than an array of predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives the
    #    validation. Mean test accuracy matches the original mean fold accuracy.
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
