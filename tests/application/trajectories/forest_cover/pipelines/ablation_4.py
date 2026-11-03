import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


N_SPLITS = 3
RANDOM_STATE = 42


def restore_original_labels(predictions, mode):
    """Convert fitted labels 0-6 back to the raw target domain 1-7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original eager FileNotFoundError
    #    handling is omitted because importing this module must not read the file;
    #    a missing file will naturally raise when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — remove classes with fewer than three observations before
    #    marking X and y, since row filtering changes the number of samples. The
    #    original repeated this check for every experiment; all variants use the
    #    same target, so recording it once is equivalent.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < N_SPLITS].index
    data = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target. Its 1-7 to 0-6 transformation is recorded afterward,
    # and predictions are converted back to 1-7 before accuracy is scored.
    y = data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Mark the raw design matrix before feature engineering. The manual
    # StratifiedKFold loops become the splitter attached to mark_as_X.
    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — create the same composite feature and
    #    remove the three original Hillshade columns.
    base_features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    # Fuse exactly the four experiments scored by the original ablation study.
    variants = {
        "Baseline": base_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
        "Ablation 1": base_features.drop(
            columns=["Horizontal_Distance_To_Fire_Points"]
        ).skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
        "Ablation 2": base_features.drop(
            columns=["Slope"]
        ).skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
        "Ablation 3": base_features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            y=y_transformed,
        ),
    }

    pred_transformed = skrub.choose_from(
        variants,
        name="variant",
    ).as_data_op()

    # Restore predictions to the raw target domain. This must be gated because
    # prediction nodes evaluate to fitted estimators while the plan is fitting.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; mark_as_X's StratifiedKFold drives validation.
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
