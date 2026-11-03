import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 3
RANDOM_STATE = 42
N_ESTIMATORS = 100

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError handling
    #    is omitted; because loading is lazy, a missing file is reported when the
    #    plan is scored rather than when this module is imported.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes classes represented fewer than three
    #    times before running StratifiedKFold. Row filtering must happen before the
    #    marks because it changes the number of rows. It was repeated identically
    #    for every ablation, so it is recorded once and shared by all variants.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < N_SPLITS].index
    data = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target. This dataset's Cover_Type labels are documented as 1-7,
    # so the original `if y.min() == 1` branch subtracts one. The transformation is
    # therefore applied after mark_as_y and inverted on predictions below.
    y = data["Cover_Type"].skb.mark_as_y()
    y_zero_indexed = y - 1

    # The original manual three-fold loop becomes the splitter on mark_as_X.
    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model variants.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    elevation_at_hydrology = (
        X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    )
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)
    aspect_sin = aspect_radians.skb.apply_func(np.sin)
    aspect_cos = aspect_radians.skb.apply_func(np.cos)
    euclidean_distance = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    ).skb.apply_func(np.sqrt)
    slope_x_euclidean_distance = X["Slope"] * euclidean_distance

    standard_drops = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]

    baseline_features = X.assign(
        Hillshade_composite=hillshade_composite,
        Elevation_at_Hydrology=elevation_at_hydrology,
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
        Euclidean_Distance_To_Hydrology=euclidean_distance,
        Slope_x_Euclidean_Distance_To_Hydrology=slope_x_euclidean_distance,
    ).drop(columns=standard_drops)

    no_slope_interaction_features = X.assign(
        Hillshade_composite=hillshade_composite,
        Elevation_at_Hydrology=elevation_at_hydrology,
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
        Euclidean_Distance_To_Hydrology=euclidean_distance,
    ).drop(columns=standard_drops)

    no_aspect_features = X.assign(
        Hillshade_composite=hillshade_composite,
        Elevation_at_Hydrology=elevation_at_hydrology,
        Euclidean_Distance_To_Hydrology=euclidean_distance,
        Slope_x_Euclidean_Distance_To_Hydrology=slope_x_euclidean_distance,
    ).drop(columns=standard_drops + ["Aspect"])

    no_euclidean_features = X.assign(
        Hillshade_composite=hillshade_composite,
        Elevation_at_Hydrology=elevation_at_hydrology,
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
    ).drop(columns=standard_drops)

    # The original scores exactly these four ablation variants. Choosing between
    # complete prediction subgraphs avoids inventing a cross-product of features.
    variants = {
        "Baseline": baseline_features.skb.apply(
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            y=y_zero_indexed,
        ),
        "No Slope_x_Euclidean_Distance_To_Hydrology": (
            no_slope_interaction_features.skb.apply(
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                y=y_zero_indexed,
            )
        ),
        "No Aspect or Aspect_sin/cos": no_aspect_features.skb.apply(
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            y=y_zero_indexed,
        ),
        (
            "No Euclidean_Distance_To_Hydrology "
            "(and derived Slope_x_Euclidean_Distance_To_Hydrology)"
        ): no_euclidean_features.skb.apply(
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            y=y_zero_indexed,
        ),
    }
    pred_zero_indexed = skrub.choose_from(
        variants, name="variant"
    ).as_data_op()

    # Accuracy is unchanged by consistently shifting labels, but scoring is
    # against the raw target marked above. Restore predictions from 0-6 to 1-7.
    # The operation is gated because in fit mode the prediction node contains the
    # fitted estimator rather than an array of predictions.
    def restore_cover_type_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_zero_indexed.skb.apply_func(
        restore_cover_type_labels, skrub.eval_mode()
    )

    # 4. Score all four variants. No cv= is passed here; the StratifiedKFold
    #    attached to mark_as_X drives the end-to-end validation.
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
