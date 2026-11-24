import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's eager FileNotFoundError
    #    message is omitted; a missing file will naturally raise during scoring.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes classes having fewer than three
    #    samples from cross-validation. Because this changes the number of rows,
    #    reproduce that filtering before marking X and y. The per-class counts
    #    are identical for raw Cover_Type values 1-7 and shifted labels 0-6.
    #    Progress and exclusion-summary printing are not part of scoring.
    class_sizes = data.groupby("Cover_Type")["Cover_Type"].transform("size")
    cv_data = data[class_sizes >= 3].reset_index(drop=True)

    # Mark the RAW target, then apply the original 1-7 to 0-6 transformation
    # downstream. Predictions are mapped back to the raw target domain below.
    y_raw = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the raw design matrix. Feature engineering and feature removal remain
    # downstream so they are recorded and re-run within every CV fold.
    X = cv_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original formulas and
    #    final feature set.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert predicted labels from 0-6 back to the raw 1-7 scoring domain.
    # In fit mode the prediction node contains the fitted estimator, so it must
    # be returned unchanged rather than used in arithmetic.
    def restore_target_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_transformed.skb.apply_func(
        restore_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. The splitter on mark_as_X reproduces the original manual
    #    three-fold loop; mean_test_score is its mean fold accuracy.
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
