import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_cover_type_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode, predictions is the fitted estimator and must be unchanged.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. Importing this module only builds the
    #    lazy plan; the file is read when the scoring block runs.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — the original excludes every class represented by fewer
    #    than three samples. This row filtering must occur before marking X and y
    #    because it changes their number of rows. Mapping each target value to its
    #    full-table count is equivalent to the original value_counts/isin logic.
    raw_cover_type = train_df["Cover_Type"]
    class_counts_per_row = raw_cover_type.map(raw_cover_type.value_counts())
    cv_df = train_df[class_counts_per_row >= 3].reset_index(drop=True)

    # Mark the RAW target, then apply the original 1-7 to 0-6 transformation for
    # model fitting. Predictions are mapped back to the raw domain below so that
    # accuracy is computed against the marked 1-7 target.
    y_raw = cv_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The manual three-fold loop becomes the splitter attached to mark_as_X.
    # Id is excluded from the model; the other columns dropped by the original
    # remain temporarily available for recorded feature engineering below.
    X = cv_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — reproduce the two original derived
    #    features, then remove the three source Hillshade columns and Aspect.
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

    # Same model family and hyperparameters as the original.
    pred_transformed = features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )

    # Restore predictions from 0-6 to the raw 1-7 target domain. The operation is
    # gated because a prediction node evaluates to the fitted estimator in fit
    # mode rather than to an array of predictions.
    pred = pred_transformed.skb.apply_func(
        restore_cover_type_labels,
        skrub.eval_mode(),
    )

    # 4. Score — the splitter on mark_as_X drives the three-fold CV.
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
