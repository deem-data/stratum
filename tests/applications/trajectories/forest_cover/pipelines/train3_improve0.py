import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_target_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw target domain of 1-7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module does not read the
    #    file; a missing file is reported only when the plan is scored.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes having fewer
    #    than three samples. Row filtering must happen before the marks because it
    #    changes the number of rows. This also retains every row when there are no
    #    problematic classes.
    target_for_filtering = train_df["Cover_Type"] - 1
    class_counts = target_for_filtering.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~target_for_filtering.isin(problematic_classes)
    cv_data = train_df[keep_rows].reset_index(drop=True)

    # Mark the RAW 1-7 target. The original model was fitted on labels shifted to
    # 0-6, so that transformation occurs only after mark_as_y and is inverted on
    # the prediction node below before accuracy is computed.
    y_raw = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the design matrix before feature engineering. The original manual
    # three-fold loop becomes the same StratifiedKFold splitter on mark_as_X.
    X = cv_data.drop(columns=["Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model — create the same composite at
    #    the end of the table, then remove Id and the three source columns while
    #    retaining the original Aspect feature.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    ).drop(
        columns=[
            "Id",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_features="log2",
        min_samples_leaf=5,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Restore predictions from 0-6 to the raw 1-7 scoring domain. The operation
    # is gated because in fit mode the prediction node contains the fitted
    # estimator rather than an array of predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter declared on mark_as_X drives the
    #    evaluation. The original has no test prediction or submission work.
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
