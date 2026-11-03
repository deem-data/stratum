import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's eager FileNotFoundError
    #    message is omitted; with lazy DataOps, any missing-file error is raised
    #    when the guarded scoring block evaluates the plan.
    train_df = (
        skrub.as_data_op("./input/train.csv")
        .skb.apply_func(pd.read_csv)
    )

    # 2. Prepare Data — the original excludes every class having fewer than three
    #    samples before cross-validation. This row filtering must happen before
    #    mark_as_X/mark_as_y because it changes the number of rows.
    class_sizes = (
        train_df.groupby("Cover_Type")["Cover_Type"].transform("size")
    )
    cv_df = train_df[class_sizes >= 3].reset_index(drop=True)

    # The original also computed a 1%-threshold majority/rare split, but those
    # tables were never used by its actual CV loop, so that dead computation is
    # omitted. No rare samples were appended to any fold in the scored code.
    #
    # Mark the RAW 1-7 target. The model's 0-6 target transformation is recorded
    # after this mark, and predictions are converted back to 1-7 below.
    y = cv_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The manual three-fold loop becomes the same StratifiedKFold splitter on X.
    # Stratifying raw labels 1-7 is equivalent to stratifying shifted labels 0-6.
    X = cv_df.drop(columns=["Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model — preserve the two engineered
    #    features and the exact feature exclusions from the original script.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"]
            + X["Hillshade_Noon"]
            + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(
        columns=[
            "Id",
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

    # Convert 0-6 model predictions back to the raw 1-7 target domain. Prediction
    # arithmetic is gated because in fit mode the node contains the fitted model.
    def restore_original_labels(prediction, mode):
        if mode == "fit":
            return prediction
        return prediction + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; mark_as_X's StratifiedKFold drives validation.
    #    Mean test accuracy matches the original mean of its three fold scores.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="accuracy",
        )
        print(search.results_)
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
