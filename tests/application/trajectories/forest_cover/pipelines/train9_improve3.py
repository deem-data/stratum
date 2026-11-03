import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw target domain, 1-7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except and progress
    #    messages are omitted because they do not contribute to the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded classes having fewer than three
    #    samples before cross-validation. Because this changes the row count, the
    #    filtering must remain before mark_as_X/mark_as_y.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    data_cv = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW 1-7 target first, then perform the original 0-6 transformation
    # downstream of the mark for model fitting.
    y = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The temporary y_stratify construction for classes with fewer than five
    # samples is omitted because the original never used it in the actual split.
    # The manual three-fold loop becomes this identical StratifiedKFold splitter.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. New columns are appended in the
    # same order as in the original, after which the same source columns are
    # removed.
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

    # Scoring compares predictions with the RAW target marked above. Restore the
    # original 1-7 label domain at prediction time. In fit mode the prediction
    # node contains the fitted estimator, so it must be returned unchanged.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is passed here; mark_as_X's StratifiedKFold drives the
    #    validation and mean_test_score is the original mean fold accuracy.
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
