import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's FileNotFoundError
    #    handling and progress printing are omitted because they do not contribute
    #    to the cross-validated scores.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes having fewer
    #    than three samples. This row filtering must happen before X and y are
    #    marked because it changes the number of rows.
    class_counts = data["Cover_Type"].value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    data = data[
        ~data["Cover_Type"].isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target. The original's 1-7 to 0-6 transformation is applied
    # only after mark_as_y, and predictions are mapped back below for scoring.
    y = data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature ablations and models. These are exactly the three
    #    configurations scored by the original script; a single choice over
    #    complete prediction graphs avoids inventing a feature/model cross-product.
    baseline_pred = X.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    baseline_pred = baseline_pred.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    reduced_estimators_pred = X.skb.apply(
        RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    reduced_estimators_pred = reduced_estimators_pred.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    no_soil_features = X.skb.drop(skrub.selectors.glob("*Soil_Type*"))
    no_soil_pred = no_soil_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    no_soil_pred = no_soil_pred.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    variants = {
        "Baseline Solution (Original)": baseline_pred,
        "Ablation 1: RandomForestClassifier with n_estimators=50": (
            reduced_estimators_pred
        ),
        "Ablation 2: Exclude all Soil_Type features": no_soil_pred,
    }
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score all three variants. The splitter on mark_as_X drives the
    #    validation; no cv= is passed here. Submission and contribution-summary
    #    logic are omitted because they are not part of producing the CV scores.
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
