from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ShuffleSplit
import polars as pl
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet,  Ridge

from time import perf_counter
import numpy as np
from sklearn.preprocessing import StandardScaler
import stratum as skrub
test=True

import logging

logging.basicConfig(level=logging.INFO)

file_path = "input/price_paid_records_small.csv" if test else "input/price_paid_records.csv"
df = skrub.as_data_op(file_path).skb.apply_func(pl.read_csv).skb.subsample(n=1000)
df = df.rename({"Town/City": "Town"})
y = df["Price"].skb.mark_as_y()
X = df.drop("Price").skb.mark_as_X()

from sklearn.base import BaseEstimator, TransformerMixin
class TargetEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        print("fit target encoder")
        self.global_mean_ = y.mean()
        y_name = y.name if isinstance(y, pl.Series) and y.name else 'target'
        # Handle both Polars Series and numpy arrays
        if isinstance(y, pl.Series):
            tmp = X.with_columns(y.alias(y_name))
        else:
            tmp = X.with_columns(pl.Series(y_name, y))
        self.cols = X.columns
        self.means = {}
        for col in self.cols:
            # Store as DataFrame with column name and mean for efficient join
            self.means[col] = tmp.group_by(col).agg(pl.col(y_name).mean().alias(f"{col}_mean"))
        return self

    def transform(self, X):
        print("transform target encoder")
        X_out = X.clone()
        for col in self.cols:
            # Use join instead of map for better performance
            mean_col_name = f"{col}_mean"
            X_out = X_out.join(
                self.means[col],
                on=col,
                how="left"
            ).with_columns(
                pl.col(mean_col_name).fill_null(self.global_mean_).alias(col)
            ).drop(mean_col_name)
        return X_out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self):
        return self.cols


def pre_process_1(X, y):
    date = X["Date of Transfer"].str.to_datetime()
    X = X.with_columns(
        year=date.dt.year(), 
        month=date.dt.month(), 
        day=date.dt.day(), 
        dayofweek=date.dt.weekday(), 
        hour=date.dt.hour())
    X = X.with_columns(
        month_sin=(date.dt.month() * (2 * np.pi / 12)).sin(),
        month_cos=(date.dt.month() * (2 * np.pi / 12)).cos(),
        day_sin=(date.dt.day() * (2 * np.pi / 30)).sin(),
        day_cos=(date.dt.day()   * (2 * np.pi / 30)).cos(),
        dayofweek_sin=(date.dt.weekday() * (2 * np.pi / 7)).sin(),
        dayofweek_cos=(date.dt.weekday() * (2 * np.pi / 7)).cos(),
        hour_sin=(date.dt.hour() * (2 * np.pi / 24)).sin(),
        hour_cos=(date.dt.hour() * (2 * np.pi / 24)).cos(),
    )
    X = X.drop([
        "Date of Transfer", 
        'Duration', 
        'Transaction unique identifier', 
        'PPDCategory Type', 
        'Record Status - monthly file only'])

    cat_selector = skrub.selectors.filter(lambda col: col.dtype == pl.String)
    X_cat = X.skb.select(cat_selector)
    X_cat_enc = X_cat.skb.apply(skrub.StringEncoder())
    num_selector = skrub.selectors.filter(lambda col: col.dtype != pl.String)

    X_te = X[["District", "County", "Town"]].skb.apply(TargetEncoder(), y=y)
    X_te = X_te.rename({"District": "district_te", "County": "county_te", "Town": "town_te"})
    X_num = X.skb.select(num_selector)
    X_num = X_num.skb.concat([X_te], axis=1)

    X_num_scaled = X_num.skb.apply(StandardScaler())
    X_vec = X_num_scaled.skb.concat([X_cat_enc], axis=1)
    return X_vec

def pre_process_2(X):
    X_enc = X.skb.apply(skrub.TableVectorizer())
    return X_enc
X_1 = pre_process_1(X,y)
print(X_1.skb.preview())
X_2 = pre_process_2(X)
X_enc = skrub.choose_from({
    "1": X_1, 
    "2": X_2
    }, name="feat_eng").as_data_op()

models = {
    "Ridge": Ridge(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42),
    "ElasticNet": ElasticNet(random_state=42),
}
preds = {k: X_enc.skb.apply(model, y=y) for k,model in models.items()}
preds = skrub.choose_from(preds, name="models").as_data_op()
preds = preds.skb.apply_func(lambda a, m: (a, print(m))[0], skrub.eval_mode())

# play with cvs
cv = 1
cv = ShuffleSplit(n_splits=1,test_size=0.2,random_state=42) if cv == 1 else KFold(n_splits=cv, shuffle=True, random_state=42)
scorer = make_scorer(mean_squared_error)
t0 = perf_counter()
with skrub.config(scheduler=True, stats=True):
    search_stratum = preds.skb.make_grid_search(cv=cv, n_jobs=1, fitted=True, scoring=scorer)
t1 = perf_counter()
print("="*80)
print(f"Stratum gridsearch scheduler time: {t1 - t0} seconds")
print("="*80)
search = preds.skb.make_grid_search(cv=cv, n_jobs=-1, fitted=True, scoring=scorer, refit=False)
t2 = perf_counter()
print("="*80)
print(f"Skrub default gridsearch time: {t2 - t1} seconds")
print("="*80)
print("Results:")
print(search.results_)
print(search_stratum.results_)
print("="*80)