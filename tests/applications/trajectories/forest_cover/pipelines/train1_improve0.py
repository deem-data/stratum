import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Shift predicted classes from 0-6 back to the raw 1-7 label domain."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the CSV read. Importing this module only builds the
    #    lazy plan and therefore does not read the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Reproduce the original data-dependent removal of target
    #    classes having fewer than three rows. Because this changes which rows
    #    are scored, it must occur before mark_as_X and mark_as_y.
    transformed_target_for_filter = data["Cover_Type"] - 1
    class_counts = transformed_target_for_filter.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~transformed_target_for_filter.isin(problematic_classes)
    filtered_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target as required. The original fitted on labels shifted from
    # 1-7 to 0-6, so that transformation is recorded after mark_as_y.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Create the same composite feature before dropping the three source
    # Hillshade columns. assign preserves the original behavior of appending the
    # new named column after the existing columns.
    prepared_data = filtered_data.assign(
        Hillshade_composite=(
            filtered_data["Hillshade_9am"]
            + filtered_data["Hillshade_Noon"]
            + filtered_data["Hillshade_3pm"]
        )
        / 3
    )

    # The manual three-fold loop becomes the splitter on mark_as_X.
    X = prepared_data.drop(
        columns=[
            "Id",
            "Cover_Type",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    ).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Model -- preserve the original model family and all hyperparameters.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = X.skb.apply(model, y=y_transformed)

    # The scorer compares against the marked raw target, so restore predictions
    # to 1-7. This preserves the original accuracy exactly while still marking
    # the raw target. Prediction arithmetic is gated because in fit mode this
    # node contains the fitted estimator rather than prediction values.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is passed here; mark_as_X's StratifiedKFold drives.
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
