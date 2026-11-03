import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_cover_type_labels(prediction, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # During fitting, a prediction node contains the fitted estimator.
        return prediction
    return prediction + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original eager FileNotFoundError
    #    message is omitted because importing this module must not read the file;
    #    a missing file is reported when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Reproduce the original removal of classes having fewer
    #    than three samples. Row filtering must occur before marking X and y
    #    because it changes their number of rows.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    data_cv = data[~data["Cover_Type"].isin(problematic_classes)].reset_index(
        drop=True
    )

    # Mark the RAW target first, then perform the original 1-7 to 0-6 transform
    # downstream of the mark. Predictions are shifted back before scoring.
    y_raw = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The manual three-fold loop becomes the splitter attached to mark_as_X.
    X = data_cv.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original feature values
    #    and final column order. Aspect_rad remains only an intermediate value,
    #    matching the original code that dropped it before model fitting.
    horizontal_hydrology = X["Horizontal_Distance_To_Hydrology"]
    vertical_hydrology = X["Vertical_Distance_To_Hydrology"]
    total_hydrology_distance = (
        horizontal_hydrology**2 + vertical_hydrology**2
    ).skb.apply_func(np.sqrt)

    aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)
    slope_aspect_sin = X["Slope"] * aspect_rad.skb.apply_func(np.sin)
    slope_aspect_cos = X["Slope"] * aspect_rad.skb.apply_func(np.cos)

    features = X.assign(
        Total_Hydrology_Distance=total_hydrology_distance,
        Slope_Aspect_Sin=slope_aspect_sin,
        Slope_Aspect_Cos=slope_aspect_cos,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Shift predictions back to the raw target domain for accuracy scoring.
    # This is gated because the prediction node holds an estimator in fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_cover_type_labels, skrub.eval_mode()
    )

    # 4. Score. No cv= here: the splitter on mark_as_X drives validation.
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
