import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded, so importing this module does not access
    #    the file. The original's custom FileNotFoundError printing is omitted
    #    because it is not part of producing the cross-validated scores.
    train_df_original = (
        skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)
    )

    # 2. Prepare Data
    #    The original excluded every class having fewer than three samples before
    #    running StratifiedKFold. This row filtering must happen before the marks
    #    because it changes the number of rows. Applying the condition
    #    unconditionally is equivalent when no problematic classes exist.
    class_count_per_row = (
        train_df_original.groupby("Cover_Type")["Cover_Type"].transform("size")
    )
    train_df = train_df_original[class_count_per_row >= 3].reset_index(drop=True)

    # Mark the RAW target as required. The original trained on labels shifted from
    # 1-7 to 0-6, so that transformation is recorded after mark_as_y and predictions
    # are shifted back before scoring.
    y_raw = train_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the raw design matrix before feature engineering. The original manual
    # three-fold loop becomes the identical splitter attached to mark_as_X.
    X = train_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)
    aspect_sin = aspect_rad.skb.apply_func(np.sin)
    aspect_cos = aspect_rad.skb.apply_func(np.cos)

    # Baseline: hillshade composite, aspect sine/cosine, and both slope-aspect
    # interactions. Assignments are kept in the original order because feature
    # order can affect a randomized forest.
    baseline_engineered = X.assign(
        Hillshade_composite=hillshade_composite,
        Aspect_rad=aspect_rad,
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
        Slope_Aspect_sin=X["Slope"] * aspect_sin,
        Slope_Aspect_cos=X["Slope"] * aspect_cos,
    )
    baseline_features = baseline_engineered.drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Aspect_rad",
        ]
    )

    # Ablation 1: retain aspect sine/cosine but do not create the slope-aspect
    # interaction features.
    no_slope_aspect_engineered = X.assign(
        Hillshade_composite=hillshade_composite,
        Aspect_rad=aspect_rad,
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
    )
    no_slope_aspect_features = no_slope_aspect_engineered.drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Aspect_rad",
        ]
    )

    # Ablation 2: retain only the hillshade composite and remove the original
    # Aspect column without creating any Aspect-derived columns.
    no_aspect_features = X.assign(
        Hillshade_composite=hillshade_composite,
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    # Ablation 3: remove Elevation from the fully engineered baseline.
    no_elevation_features = baseline_features.drop(columns=["Elevation"])

    # The original scores exactly these four variants. They are fused into one
    # discrete choice so one grid search produces one result row per ablation.
    variants_transformed = {
        "Baseline": baseline_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
        "No Slope-Aspect Interaction Features": no_slope_aspect_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
        "No Aspect-related Features (original + engineered)": (
            no_aspect_features.skb.apply(
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
                y=y_transformed,
            )
        ),
        "No Elevation Feature": no_elevation_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
    }
    pred_transformed = skrub.choose_from(
        variants_transformed,
        name="variant",
    ).as_data_op()

    # Restore predictions from 0-6 to the raw target's 1-7 domain. This must be
    # gated because prediction nodes contain fitted estimators in fit mode.
    def restore_raw_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_raw_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the StratifiedKFold attached to mark_as_X drives.
    #    The manual result-summary logic is replaced by results_, which contains
    #    the accuracy of every explicitly named ablation variant.
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
