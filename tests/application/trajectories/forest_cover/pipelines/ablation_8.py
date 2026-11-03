import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_target(predictions, mode):
    """Map model predictions from labels 0-6 back to raw labels 1-7."""
    if mode == "fit":
        # During fitting, a prediction node evaluates to the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded, so importing this module neither reads
    #    the CSV nor fits a model. File-error handling and progress output from the
    #    original are omitted because they do not contribute to the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes classes represented by fewer than
    #    three rows before StratifiedKFold. This filtering must precede the marks
    #    because it changes the number of rows.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~data["Cover_Type"].isin(problematic_classes)
    cv_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target, then reproduce the original 1-7 to 0-6 transformation
    # downstream. Predictions are mapped back to 1-7 before scoring.
    y_raw = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The original manual three-fold loop becomes the same splitter on mark_as_X.
    X = cv_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and the four explicitly scored variants.
    # These two engineered features occur in every variant.
    common = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    )

    roadway_log = (
        common["Horizontal_Distance_To_Roadways"]
        .clip(lower=0)
        .skb.apply_func(np.log1p)
    )

    # Baseline: log-transform Roadways and create both interaction features.
    baseline_features = common.assign(
        Horizontal_Distance_To_Roadways_log=roadway_log
    )
    baseline_features = baseline_features.assign(
        Hydro_Road_Interaction=(
            baseline_features["Horizontal_Distance_To_Hydrology"]
            * baseline_features["Horizontal_Distance_To_Roadways_log"]
        ),
        Elevation_x_Fire_Points=(
            baseline_features["Elevation"]
            * baseline_features["Horizontal_Distance_To_Fire_Points"]
        ),
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Horizontal_Distance_To_Roadways",
        ]
    )

    # Ablation 1: retain the original Roadways feature and use it directly in
    # Hydro_Road_Interaction; do not create the log-transformed replacement.
    ablation1_features = common.assign(
        Hydro_Road_Interaction=(
            common["Horizontal_Distance_To_Hydrology"]
            * common["Horizontal_Distance_To_Roadways"]
        ),
        Elevation_x_Fire_Points=(
            common["Elevation"] * common["Horizontal_Distance_To_Fire_Points"]
        ),
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    # Ablation 2: use transformed Roadways and Hydro_Road_Interaction, but omit
    # Elevation_x_Fire_Points.
    ablation2_features = common.assign(
        Horizontal_Distance_To_Roadways_log=roadway_log
    )
    ablation2_features = ablation2_features.assign(
        Hydro_Road_Interaction=(
            ablation2_features["Horizontal_Distance_To_Hydrology"]
            * ablation2_features["Horizontal_Distance_To_Roadways_log"]
        )
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Horizontal_Distance_To_Roadways",
        ]
    )

    # Ablation 3: use transformed Roadways and Elevation_x_Fire_Points, but omit
    # Hydro_Road_Interaction.
    ablation3_features = common.assign(
        Horizontal_Distance_To_Roadways_log=roadway_log
    )
    ablation3_features = ablation3_features.assign(
        Elevation_x_Fire_Points=(
            ablation3_features["Elevation"]
            * ablation3_features["Horizontal_Distance_To_Fire_Points"]
        )
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Horizontal_Distance_To_Roadways",
        ]
    )

    # Keep the original model family and every explicitly supplied parameter.
    # Each model fits labels 0-6, as in the original. Its predictions are then
    # restored to raw labels 1-7 so they match the raw mark_as_y scoring domain.
    baseline_pred_transformed = baseline_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    baseline_pred = baseline_pred_transformed.skb.apply_func(
        restore_original_target,
        skrub.eval_mode(),
    )

    ablation1_pred_transformed = ablation1_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    ablation1_pred = ablation1_pred_transformed.skb.apply_func(
        restore_original_target,
        skrub.eval_mode(),
    )

    ablation2_pred_transformed = ablation2_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    ablation2_pred = ablation2_pred_transformed.skb.apply_func(
        restore_original_target,
        skrub.eval_mode(),
    )

    ablation3_pred_transformed = ablation3_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    ablation3_pred = ablation3_pred_transformed.skb.apply_func(
        restore_original_target,
        skrub.eval_mode(),
    )

    # Fuse exactly the four variants scored by the original ablation study.
    variants = {
        "Baseline": baseline_pred,
        "Ablation 1 (No log1p on Horizontal_Distance_To_Roadways)": (
            ablation1_pred
        ),
        "Ablation 2 (No Elevation_x_Fire_Points)": ablation2_pred,
        "Ablation 3 (No Hydro_Road_Interaction)": ablation3_pred,
    }
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score all variants. No cv= is passed here; the StratifiedKFold declared
    #    on mark_as_X drives the end-to-end validation.
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
