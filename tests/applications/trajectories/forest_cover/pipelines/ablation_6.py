import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


N_SPLITS = 3
RANDOM_STATE = 42
N_ESTIMATORS = 100


def inverse_label_shift(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode a prediction node contains the fitted estimator.
        return predictions
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    lazy plan; the file is not read until the scoring block is executed.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded target classes having fewer than
    #    N_SPLITS samples before every experiment. Because this changes the row
    #    count, the same data-dependent filtering must occur before the marks.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < N_SPLITS].index
    data = data[~data["Cover_Type"].isin(problematic_classes)]

    # Mark the RAW target as required. The original trained on labels shifted
    # from 1-7 to 0-6, so that transformation is recorded after mark_as_y and
    # inverted on each variant's predictions before accuracy is calculated.
    y_raw = data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The manual three-fold loop becomes the splitter attached to mark_as_X.
    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. New columns are assigned in the same
    #    order as in the original script so RandomForest sees the same feature
    #    order for every ablation.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    hydrology_squared_distance = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    )
    euclidean_hydrology = hydrology_squared_distance.skb.apply_func(np.sqrt)
    elevation_at_hydrology = (
        X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    )
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)
    aspect_sin = aspect_radians.skb.apply_func(np.sin)
    aspect_cos = aspect_radians.skb.apply_func(np.cos)
    human_proximity = (
        X["Horizontal_Distance_To_Roadways"]
        + X["Horizontal_Distance_To_Fire_Points"]
    )

    baseline_features = X.assign(
        Hillshade_composite=hillshade_composite,
    ).assign(
        Euclidean_Distance_To_Hydrology=euclidean_hydrology,
    ).assign(
        Elevation_at_Hydrology=elevation_at_hydrology,
    ).assign(
        Aspect_sin=aspect_sin,
    ).assign(
        Aspect_cos=aspect_cos,
    ).assign(
        Proximity_To_Human_Features=human_proximity,
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    ablation1_features = baseline_features.drop(
        columns=["Proximity_To_Human_Features"]
    )
    ablation2_features = baseline_features.drop(
        columns=["Euclidean_Distance_To_Hydrology"]
    )

    # Ablation 3 keeps the original Aspect and never creates Aspect_sin/cos.
    # It is built separately to preserve the original column ordering.
    ablation3_features = X.assign(
        Hillshade_composite=hillshade_composite,
    ).assign(
        Euclidean_Distance_To_Hydrology=euclidean_hydrology,
    ).assign(
        Elevation_at_Hydrology=elevation_at_hydrology,
    ).assign(
        Proximity_To_Human_Features=human_proximity,
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    baseline_pred_0_indexed = baseline_features.skb.apply(
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    ablation1_pred_0_indexed = ablation1_features.skb.apply(
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    ablation2_pred_0_indexed = ablation2_features.skb.apply(
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    ablation3_pred_0_indexed = ablation3_features.skb.apply(
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        y=y_transformed,
    )

    baseline_pred = baseline_pred_0_indexed.skb.apply_func(
        inverse_label_shift, skrub.eval_mode()
    )
    ablation1_pred = ablation1_pred_0_indexed.skb.apply_func(
        inverse_label_shift, skrub.eval_mode()
    )
    ablation2_pred = ablation2_pred_0_indexed.skb.apply_func(
        inverse_label_shift, skrub.eval_mode()
    )
    ablation3_pred = ablation3_pred_0_indexed.skb.apply_func(
        inverse_label_shift, skrub.eval_mode()
    )

    # The original printed four scores, so fuse exactly those four named
    # variants into one discrete grid-search plan.
    variants = {
        "Baseline": baseline_pred,
        "Ablation 1: Removed 'Proximity_To_Human_Features'": ablation1_pred,
        "Ablation 2: Removed 'Euclidean_Distance_To_Hydrology'": ablation2_pred,
        "Ablation 3: Kept original 'Aspect', removed sin/cos transformation": (
            ablation3_pred
        ),
    }
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score all four variants. No cv= is passed here: the StratifiedKFold
    #    attached to mark_as_X drives the complete end-to-end validation.
    #    The original's hand-written summary/conclusion is replaced by
    #    results_, which contains one cross-validated row per scored variant.
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
