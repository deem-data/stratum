import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except is omitted:
    #    because loading is lazy, pandas will raise FileNotFoundError when the
    #    plan is scored if ./input/train.csv is absent.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes having fewer
    #    than three samples. Row filtering must occur before marking X and y
    #    because it changes their number of rows.
    target_zero_based = train_df["Cover_Type"] - 1
    class_counts = target_zero_based.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    keep_rows = ~target_zero_based.isin(problematic_classes)
    cv_df = train_df[keep_rows].reset_index(drop=True)

    # Mark the raw 1-7 target. Its 0-6 transformation is recorded only after
    # mark_as_y, and predictions are converted back to the raw domain below.
    y = cv_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The manual three-fold loop becomes the splitter on mark_as_X. Id and the
    # target are excluded exactly as in the original design matrix.
    X = cv_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original sine/cosine
    #    representation of Aspect and the mean of the three Hillshade columns.
    aspect_radians = X["Aspect"].skb.apply_func(np.radians)
    features = X.assign(
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
    ).drop(
        columns=[
            "Aspect",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_zero_based = features.skb.apply(model, y=y_transformed)

    # Convert predictions from 0-6 back to the marked target's raw 1-7 domain.
    # Prediction arithmetic is gated because in fit mode the node contains the
    # fitted estimator rather than prediction values.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_zero_based.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — the splitter on mark_as_X drives the same three-fold
    #    stratified validation as the original manual loop.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="accuracy",
        )
        print(search.results_)
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
