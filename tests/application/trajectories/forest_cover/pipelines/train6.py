import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map predictions from 0-6 back to the raw target domain of 1-7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError handler is
    #    omitted because importing this module must not read the file; a missing file
    #    will naturally raise when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes classes represented by fewer than
    #    three rows before cross-validation. This row filtering remains before the
    #    marks because it changes the number of rows. The counts and filtering are
    #    lazily recorded and therefore still depend on the loaded dataset.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    data_cv = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target, then reproduce the original 1-7 to 0-6 training-label
    # transform downstream of the mark.
    y_raw = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the unengineered design matrix. The manual three-fold loop becomes the
    # same StratifiedKFold splitter attached to mark_as_X.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Assigning both new columns before dropping
    #    the three source columns preserves the original feature values, names, and
    #    column order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
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

    # Scoring compares against the raw mark_as_y node, so map predictions from
    # 0-6 back to 1-7. The eval-mode guard leaves the fitted estimator unchanged
    # during fit, when a prediction node does not yet contain predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is passed here; the splitter on mark_as_X drives the
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
