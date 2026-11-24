import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except is omitted;
    #    because loading is lazy, a missing file is reported when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — faithfully remove target classes having fewer than three
    #    rows before marking X and y. This content-dependent filtering cannot be
    #    omitted; expressing it as recorded operations preserves the original
    #    cross-validation population.
    raw_target = data["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_data = data[
        ~raw_target.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target, then apply the original 1-7 to 0-6 transformation
    # downstream. Predictions are mapped back to the raw domain below so scoring
    # remains aligned with this mark_as_y node.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Keep the Hillshade inputs until their composite has been computed downstream.
    # The original manual StratifiedKFold loop is represented by this splitter.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original feature values,
    #    names, and appended-column order.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3

    euclidean_hydrology = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    ).skb.apply_func(np.sqrt)

    features = X.assign(
        Hillshade_composite=hillshade_composite,
        Euclidean_Distance_To_Hydrology=euclidean_hydrology,
        Elevation_Hydrology_Interaction=X["Elevation"] * euclidean_hydrology,
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predicted labels from 0-6 back to the raw 1-7 target domain.
    # In fit mode a prediction node contains the fitted estimator rather than
    # predictions, so it must pass through unchanged.
    def inverse_target_transform(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_transformed.skb.apply_func(
        inverse_target_transform,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives the
    #    three-fold validation. Progress and rare-class warning prints from the
    #    manual loop are omitted because they do not produce the CV score.
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
