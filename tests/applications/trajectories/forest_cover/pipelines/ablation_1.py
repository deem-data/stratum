import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from skrub import selectors as s


DATA_PATH = "./input/train.csv"


def restore_original_labels(values, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode, a prediction node contains the fitted estimator.
        return values
    return values + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — recorded read. The original reloaded the same CSV for every
    #    ablation; all variants can share this recorded source without changing
    #    their semantics.
    data = skrub.as_data_op(DATA_PATH).skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded classes having fewer than three
    #    samples before running StratifiedKFold. This equivalent row filtering
    #    must occur before the marks because it changes the number of rows.
    class_sizes = data.groupby("Cover_Type")["Cover_Type"].transform("size")
    filtered_data = data[class_sizes >= 3].reset_index(drop=True)

    # Mark the RAW 1-7 target. The 0-6 transform used for model fitting is applied
    # only after mark_as_y; predictions are shifted back before scoring.
    y_raw = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and the four exact ablation variants.
    aspect_radians = X["Aspect"].skb.apply_func(np.radians)
    aspect_sin = aspect_radians.skb.apply_func(np.sin)
    aspect_cos = aspect_radians.skb.apply_func(np.cos)
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3

    baseline_features = X.assign(
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
        Hillshade_composite=hillshade_composite,
    ).drop(
        columns=[
            "Aspect",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    no_aspect_features = X.assign(
        Hillshade_composite=hillshade_composite,
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    no_hillshade_features = X.assign(
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
    ).drop(columns=["Aspect"])

    no_wilderness_features = (
        X.assign(
            Aspect_sin=aspect_sin,
            Aspect_cos=aspect_cos,
            Hillshade_composite=hillshade_composite,
        )
        .drop(
            columns=[
                "Aspect",
                "Hillshade_9am",
                "Hillshade_Noon",
                "Hillshade_3pm",
            ]
        )
        .skb.drop(s.glob("*Wilderness_Area*"))
    )

    variants = {
        "Baseline (Original Solution)": baseline_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
        "No Aspect Sine/Cosine": no_aspect_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
        "No Hillshade Composite": no_hillshade_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
        "No Wilderness_Area Features": no_wilderness_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
    }

    pred_transformed = skrub.choose_from(variants, name="variant").as_data_op()
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score all four variants in one grid search. No cv= is passed here:
    #    the StratifiedKFold attached to mark_as_X drives validation. The
    #    original's progress messages and post-hoc performance-drop summary are
    #    omitted; search.results_ directly contains every ablation score.
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
