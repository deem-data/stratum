import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_target_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw target domain of 1-7."""
    if mode == "fit":
        # In fit mode the prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except and progress
    #    messages are omitted because they do not contribute to the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded every class having fewer than three
    #    samples before cross-validation. This filtering must happen before the
    #    marks because it changes the number of rows. When there are no
    #    problematic classes, the selection leaves every row in the table.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    data_cv = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW 1-7 target, then perform the original 0-6 transformation
    # downstream of the mark. Predictions are mapped back to 1-7 below so that
    # scoring against this raw mark remains in the correct target domain.
    y_raw = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The original manual StratifiedKFold loop becomes the splitter attached to
    # mark_as_X.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original feature names,
    #    formulas, append order, and dropped source columns.
    aspect_radians = X["Aspect"].skb.apply_func(np.radians)
    horizontal_hydrology = X["Horizontal_Distance_To_Hydrology"]
    vertical_hydrology = X["Vertical_Distance_To_Hydrology"]

    features = X.assign(
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Distance_to_Hydrology=(
            horizontal_hydrology**2 + vertical_hydrology**2
        ).skb.apply_func(np.sqrt),
    ).drop(
        columns=[
            "Aspect",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # The model predicts transformed labels (0-6), while scoring uses the raw
    # mark_as_y target (1-7). Restore predictions to the raw domain. The
    # operation is gated because in fit mode the node is a fitted estimator.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here: the StratifiedKFold on mark_as_X drives validation.
    #    Adding one to both the predictions and scoring target preserves exactly
    #    the original accuracy computed in the transformed 0-6 domain.
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
