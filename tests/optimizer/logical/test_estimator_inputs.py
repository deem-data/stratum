"""Estimator inputs are converted to pandas only when the estimator needs it."""
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from stratum.optimizer.logical._ops import check_estm_inputs, estm_supports_polars


def test_a_meta_estimator_supports_polars_only_if_what_it_wraps_does():
    sklearn_only = VotingClassifier([("lr", LogisticRegression()),
                                     ("hgb", HistGradientBoostingClassifier())])
    with_lightgbm = VotingClassifier([("lr", LogisticRegression()),
                                      ("lgbm", LGBMClassifier())])
    assert estm_supports_polars(sklearn_only)
    assert not estm_supports_polars(with_lightgbm)


def test_a_pandas_target_is_left_as_it_is():
    x = pl.DataFrame({"a": [1.0, 2.0], "d": ["2020-01-01", "2020-01-02"]})
    y = pd.Series([0, 1], name="y")
    converted, x2, y2 = check_estm_inputs(LGBMClassifier(), "fit_transform", x, y)
    assert converted and isinstance(x2, pd.DataFrame) and y2 is y


def test_a_polars_target_is_converted_with_x():
    x = pl.DataFrame({"a": [1.0, 2.0]})
    y = pl.Series("y", [0, 1])
    converted, x2, y2 = check_estm_inputs(LGBMClassifier(), "fit_transform", x, y)
    assert converted and isinstance(y2, pd.Series) and y2.tolist() == [0, 1]
