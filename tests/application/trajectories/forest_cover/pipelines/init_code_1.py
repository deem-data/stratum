import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class SingletonClassTrainingSplitter(BaseCrossValidator):
    """Reproduce the original CV scheme for the singleton Cover_Type 5 row."""

    def __init__(self, n_splits=3, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("This splitter requires the target labels.")

        # The splitter receives the raw target marked by mark_as_y, so class 5
        # refers to the original 1-7 label space.
        y_array = np.asarray(y).reshape(-1)
        class_5_indices = np.flatnonzero(y_array == 5)

        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        if len(class_5_indices) == 1:
            # Match the original exactly: stratify only the reduced data, put the
            # singleton row into every training fold, and never validate on it.
            eligible_indices = np.flatnonzero(y_array != 5)
            reduced_y = y_array[eligible_indices]

            for reduced_train, reduced_validation in splitter.split(
                np.zeros(len(eligible_indices)), reduced_y
            ):
                train_indices = np.concatenate(
                    [eligible_indices[reduced_train], class_5_indices]
                )
                validation_indices = eligible_indices[reduced_validation]
                yield train_indices, validation_indices
        else:
            # This is the original fallback when Cover_Type 5 is absent or has
            # more than one sample.
            yield from splitter.split(np.zeros(len(y_array)), y_array)


def restore_original_labels(predictions, mode):
    """Convert predictions from the model's 0-6 space back to raw 1-7 labels."""
    if mode == "fit":
        # In fit mode this value is the fitted estimator, not predictions.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the existence check is omitted because it is not part of
    #    producing the CV score and would perform filesystem I/O during import.
    train_df = (
        skrub.as_data_op("./input/train.csv")
        .skb.apply_func(pd.read_csv)
    )

    # 2. Prepare Data — mark the RAW 1-7 target. The custom splitter faithfully
    #    reproduces the original StratifiedKFold scheme, including adding the
    #    singleton Cover_Type 5 row to every training fold while excluding it
    #    from all validation folds.
    y = train_df["Cover_Type"].skb.mark_as_y()
    X = train_df.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=SingletonClassTrainingSplitter(
            n_splits=3,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded target transformation and model — the target is shifted only
    #    after mark_as_y, while the RandomForest hyperparameters are unchanged.
    y_transformed = y - 1
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = X.skb.apply(model, y=y_transformed)

    # Restore predictions to the raw target domain for scoring. This operation
    # is gated because prediction nodes contain estimators during fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives.
    #    OOF-prediction storage and progress messages are omitted because they
    #    do not contribute to the final cross-validated accuracy.
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
