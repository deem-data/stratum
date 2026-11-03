import pandas as pd
import stratum as skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def restore_original_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # During fitting, a prediction node evaluates to the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False), skrub.config(scheduler=True, debug_graph=True):
    # 1. Load Data — record the CSV read. The original FileNotFoundError message
    #    and progress printing are dropped because they do not produce the CV score.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — preserve the original removal of classes having fewer
    #    than three samples. Row filtering must happen before marking X and y
    #    because it changes the number of rows. Cover_Type and Cover_Type - 1
    #    have identical class counts, so counting the raw labels is equivalent.
    raw_target = train_df["Cover_Type"]
    class_counts = raw_target.value_counts()
    eligible_classes = class_counts[class_counts >= 3].index
    filtered_df = train_df[
        raw_target.isin(eligible_classes)
    ].reset_index(drop=True)

    # Mark the RAW target, then apply the original 1-7 to 0-6 transformation.
    y = filtered_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The original manual three-fold loop becomes the same StratifiedKFold
    # splitter on mark_as_X.
    X = filtered_df.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Model — preserve the original model family and all hyperparameters.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = X.skb.apply(model, y=y_transformed)

    # Predictions must be restored to 1-7 because scoring uses the marked raw
    # target. The conversion is gated because fit mode exposes the estimator.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — the splitter on mark_as_X drives the three-fold validation.
    #    No test prediction or submission code exists or is needed.
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
