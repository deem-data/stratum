import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from skrub import selectors as s


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read so importing this module does not read it.
    # The original FileNotFoundError handling is omitted; a missing file naturally
    # raises when the recorded plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data. The original excludes classes having fewer than three
    # samples before every StratifiedKFold run. Row filtering must happen before
    # the marks because it changes the number of rows. Filtering to classes whose
    # counts are at least three is equivalent whether performed on Cover_Type
    # labels 1-7 or on their shifted 0-6 representation.
    class_counts = data["Cover_Type"].value_counts()
    eligible_classes = class_counts[class_counts >= 3].index
    filtered_data = data[
        data["Cover_Type"].isin(eligible_classes)
    ].reset_index(drop=True)

    # Mark the RAW target first. The 1-7 to 0-6 transform used for model fitting
    # is recorded only after mark_as_y; predictions are shifted back below so
    # accuracy is evaluated against the raw labels.
    y_raw = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The manual three-fold loop becomes the splitter on mark_as_X.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model variants. The four variants are
    # exactly those scored by the original ablation study; choose_from fuses them
    # into one grid search without inventing additional feature combinations.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    euclidean_distance = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    ).skb.apply_func(np.sqrt)
    elevation_at_hydrology = (
        X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    )

    with_hillshade_composite = X.assign(
        Hillshade_composite=hillshade_composite
    )

    baseline_features = with_hillshade_composite.assign(
        Euclidean_Distance_To_Hydrology=euclidean_distance
    ).assign(
        Elevation_at_Hydrology=elevation_at_hydrology
    ).skb.drop(
        s.cols("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm")
    )

    ablation1_features = with_hillshade_composite.assign(
        Elevation_at_Hydrology=elevation_at_hydrology
    ).skb.drop(
        s.cols("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm")
    )

    ablation2_features = with_hillshade_composite.assign(
        Euclidean_Distance_To_Hydrology=euclidean_distance
    ).skb.drop(
        s.cols("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm")
    )

    ablation3_features = with_hillshade_composite.assign(
        Euclidean_Distance_To_Hydrology=euclidean_distance
    ).assign(
        Elevation_at_Hydrology=elevation_at_hydrology
    )

    model_params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }

    variants = {
        "Baseline": baseline_features.skb.apply(
            RandomForestClassifier(**model_params),
            y=y_transformed,
        ),
        "Ablation 1": ablation1_features.skb.apply(
            RandomForestClassifier(**model_params),
            y=y_transformed,
        ),
        "Ablation 2": ablation2_features.skb.apply(
            RandomForestClassifier(**model_params),
            y=y_transformed,
        ),
        "Ablation 3": ablation3_features.skb.apply(
            RandomForestClassifier(**model_params),
            y=y_transformed,
        ),
    }
    pred_shifted = skrub.choose_from(variants, name="variant").as_data_op()

    # Shift predictions from 0-6 back to the raw 1-7 target domain. This must be
    # gated because prediction nodes contain fitted estimators in fit mode.
    def restore_raw_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_shifted.skb.apply_func(
        restore_raw_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here: the StratifiedKFold on mark_as_X drives validation.
    # The original's post-hoc contribution prose is omitted; results_ directly
    # contains the accuracy of every named ablation variant.
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
