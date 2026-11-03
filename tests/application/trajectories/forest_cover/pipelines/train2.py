import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's custom FileNotFoundError
    #    message and progress printing are omitted because they do not contribute
    #    to the cross-validated score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Faithfully identify and remove target classes containing
    #    fewer than three rows before marking X and y. If there are no such
    #    classes, isin(problematic_classes) is false for every row, reproducing
    #    the original script's else branch without deciding from unseen data.
    n_splits = 3
    target_for_filtering = data["Cover_Type"] - 1
    class_counts = target_for_filtering.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index
    filtered_data = data[
        ~target_for_filtering.isin(problematic_classes)
    ].reset_index(drop=True)

    # Mark the RAW target as required. Its 1-7 to 0-6 transformation is recorded
    # afterward and used to fit the classifier.
    y = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The original manual three-fold loop becomes the splitter on mark_as_X.
    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. Assigning the composite before
    #    dropping its source columns preserves the original feature values,
    #    names, and column order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # The model predicts transformed labels (0-6), while scoring uses the marked
    # raw target (1-7). Restore predictions to the raw label domain. This must be
    # gated because in fit mode the prediction node contains the fitted estimator.
    def restore_original_labels(predictions, mode):
        if mode == "fit":
            return predictions
        return predictions + 1

    pred = pred_transformed.skb.apply_func(
        restore_original_labels, skrub.eval_mode()
    )

    # 4. Score. No cv= is passed here; mark_as_X's StratifiedKFold drives the
    #    evaluation. mean_test_score is the original mean fold accuracy.
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
