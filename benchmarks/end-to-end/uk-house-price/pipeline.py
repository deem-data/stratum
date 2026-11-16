"""Shared preprocessing for the two UK housing searches."""
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

SEED = 42
FOLDS = 3


def load_data(path):
    data = skrub.as_data_op(str(path)).skb.apply_func(pd.read_csv)
    data = data.rename(columns={"Town/City": "Town"})
    return data.drop(columns="Price").skb.mark_as_X(), data["Price"].skb.mark_as_y()


def manual_features(X, y):
    date = X["Date of Transfer"].skb.apply_func(pd.to_datetime)
    parts = {"year": date.dt.year, "month": date.dt.month, "day": date.dt.day,
             "dayofweek": date.dt.dayofweek, "hour": date.dt.hour}
    X = X.assign(**parts)
    for name, period in (("month", 12), ("day", 30), ("dayofweek", 7), ("hour", 24)):
        angle = parts[name] * (2 * np.pi / period)
        X = X.assign(**{f"{name}_sin": angle.apply(np.sin),
                        f"{name}_cos": angle.apply(np.cos)})
    X = X.drop(columns=["Date of Transfer", "Duration", "Transaction unique identifier",
                        "PPDCategory Type", "Record Status - monthly file only"])

    text = X.skb.select(~skrub.selectors.numeric()).skb.apply(
        skrub.StringEncoder(random_state=SEED))
    regional = X[["District", "County", "Town"]].skb.apply(
        TargetEncoder(target_type="continuous", random_state=SEED), y=y)
    regional = regional.rename(columns={"District": "district_te", "County": "county_te",
                                        "Town": "town_te"})
    numeric = X.skb.select(skrub.selectors.numeric()).skb.concat([regional], axis=1)
    return numeric.skb.apply(StandardScaler()).skb.concat([text], axis=1)


def table_features(X):
    vectorizer = skrub.TableVectorizer(
        high_cardinality=skrub.StringEncoder(random_state=SEED),
        low_cardinality=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        n_jobs=1,
    )
    return X.skb.apply(vectorizer).skb.apply(StandardScaler())


def lightgbm(**params):
    return LGBMRegressor(random_state=SEED, verbosity=-1, **params)
