import pandas as pd
import skrub
from skrub import selectors as s
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_cover_type_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw target domain 1-7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's custom FileNotFoundError
    # message is omitted; the recorded read naturally raises if the file is absent.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of classes containing fewer
    # than three samples. Row filtering must happen before marking because it changes
    # the number of rows. It is done once because every ablation uses the same target.
    target_zero_indexed = data["Cover_Type"] - 1
    class_counts = target_zero_indexed.value_counts()
    problematic_classes = class_counts[class_counts < 3].index.tolist()
    keep_rows = ~target_zero_indexed.isin(problematic_classes)
    filtered_data = data[keep_rows].reset_index(drop=True)

    # Mark the RAW target, as required for scoring in its original 1-7 domain.
    # The model is trained on the original's transformed 0-6 labels; predictions
    # are mapped back to 1-7 below before accuracy is computed.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The original manual three-fold loop becomes the identical splitter on X.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and the four explicitly scored variants.
    # Appending the composite and then removing its source columns preserves the
    # original feature names and ordering.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3

    baseline_features = X.assign(
        Hillshade_composite=hillshade_composite
    ).skb.drop(
        s.cols("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm")
    )

    no_hydrology_features = baseline_features.skb.drop(
        s.cols(
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
        )
    )

    no_roadways_features = baseline_features.skb.drop(
        s.cols("Horizontal_Distance_To_Roadways")
    )

    # Ablation 3 uses the untouched original hillshade columns and creates no
    # composite, exactly as in the original experiment.
    original_hillshade_features = X

    baseline_pred_raw = baseline_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    baseline_pred = baseline_pred_raw.skb.apply_func(
        restore_cover_type_labels,
        skrub.eval_mode(),
    )

    ablation1_pred_raw = no_hydrology_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    ablation1_pred = ablation1_pred_raw.skb.apply_func(
        restore_cover_type_labels,
        skrub.eval_mode(),
    )

    ablation2_pred_raw = no_roadways_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    ablation2_pred = ablation2_pred_raw.skb.apply_func(
        restore_cover_type_labels,
        skrub.eval_mode(),
    )

    ablation3_pred_raw = original_hillshade_features.skb.apply(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        y=y_transformed,
    )
    ablation3_pred = ablation3_pred_raw.skb.apply_func(
        restore_cover_type_labels,
        skrub.eval_mode(),
    )

    variants = {
        "Baseline": baseline_pred,
        "Ablation 1: Removed Hydrology features": ablation1_pred,
        "Ablation 2: Removed Horizontal_Distance_To_Roadways": ablation2_pred,
        "Ablation 3: Used Original Hillshade instead of Composite": ablation3_pred,
    }
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score all four variants in one grid search. No cv= is passed here; the
    # StratifiedKFold attached to mark_as_X drives the validation.
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
