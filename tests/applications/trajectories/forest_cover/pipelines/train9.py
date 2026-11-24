import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original eager FileNotFoundError
    #    handling is omitted because importing this module must not read the file;
    #    a missing file will naturally raise when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes containing
    #    fewer than three samples. Row filtering must happen before mark_as_X and
    #    mark_as_y because it changes the number of rows.
    raw_cover_type = data["Cover_Type"]
    class_counts = raw_cover_type.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~raw_cover_type.isin(problematic_classes)
    cv_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW 1-7 target, then apply the original 0-6 transformation for
    # model fitting. Predictions are mapped back to 1-7 below before scoring.
    y_original = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_original - 1

    # Mark the raw design matrix before recorded feature engineering. The
    # StratifiedKFold here replaces the original manual three-fold loop.
    X = cv_data.drop(columns=["Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and feature selection, preserving the
    # original formulas and excluded columns.
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
            "Id",
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

    # Restore predictions to the raw target domain used by mark_as_y. The
    # eval-mode guard leaves the fitted estimator unchanged during fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score by accuracy. No cv= is passed here—the splitter attached to
    # mark_as_X drives validation. Submission work is absent, as in the original.
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
