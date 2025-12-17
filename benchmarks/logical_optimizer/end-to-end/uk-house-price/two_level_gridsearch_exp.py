from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, ShuffleSplit
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet,  Ridge

from time import perf_counter
import numpy as np
from sklearn.preprocessing import StandardScaler
import stratum as skrub
import io
from contextlib import redirect_stdout, redirect_stderr
test=True

file_path = "input/price_paid_records_small.csv" if test else "input/price_paid_records.csv"
df = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)
print(df.columns.skb.preview())
df = df.rename(columns={"Town/City": "Town"}, inplace=False)
y = df["Price"].skb.mark_as_y()
X = df.drop("Price", axis=1).skb.mark_as_X()

from sklearn.base import BaseEstimator, TransformerMixin
class TargetEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        print("fit target encoder")
        self.global_mean_ = y.mean()
        tmp = pd.concat([X, y], axis=1)
        self.cols = X.columns
        self.means = {}
        for col in self.cols:
            self.means[col] = tmp.groupby(col)[tmp.columns[-1]].mean()
        return self

    def transform(self, X):
        print("transform target encoder")
        X_out = X.copy()
        for col in self.cols:
            X_out[col] = X_out[col].map(self.means[col]).fillna(self.global_mean_)
        return X_out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self):
        return self.cols


def pre_process_1(X, y):
    date = X["Date of Transfer"].skb.apply_func(pd.to_datetime)
    X = X.assign(
        year=date.dt.year, 
        month=date.dt.month, 
        day=date.dt.day, 
        dayofweek=date.dt.dayofweek, 
        hour=date.dt.hour)
    X = X.assign(
        month_sin=(date.dt.month * (2 * np.pi / 12)).apply(np.sin),
        month_cos=(date.dt.month * (2 * np.pi / 12)).apply(np.cos),
        day_sin=(date.dt.day * (2 * np.pi / 30)).apply(np.sin),
        day_cos=(date.dt.day * (2 * np.pi / 30)).apply(np.cos),
        dayofweek_sin=(date.dt.dayofweek * (2 * np.pi / 7)).apply(np.sin),
        dayofweek_cos=(date.dt.dayofweek * (2 * np.pi / 7)).apply(np.cos),
        hour_sin=(date.dt.hour * (2 * np.pi / 24)).apply(np.sin),
        hour_cos=(date.dt.hour * (2 * np.pi / 24)).apply(np.cos),
    )
    X = X.drop([
        "Date of Transfer", 
        'Duration', 
        'Transaction unique identifier', 
        'PPDCategory Type', 
        'Record Status - monthly file only'], axis=1)

    cat_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
    X_cat = X.skb.select(cat_selector)
    X_cat_enc = X_cat.skb.apply(skrub.StringEncoder())
    num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

    X_te = X[["District", "County", "Town"]].skb.apply(TargetEncoder(), y=y)
    X_te = X_te.rename(columns={"District": "district_te", "County": "county_te", "Town": "town_te"})
    X_num = X.skb.select(num_selector)
    X_num = X_num.skb.concat([X_te], axis=1)

    X_num_scaled = X_num.skb.apply(StandardScaler())
    X_vec = X_num_scaled.skb.concat([X_cat_enc], axis=1)
    return X_vec

def pre_process_2(X):
    X_enc = X.skb.apply(skrub.TableVectorizer())
    return X_enc

X_1 = pre_process_1(X,y)
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
preds = preds.skb.apply_func(lambda a: a)
preds.skb.draw_graph().open()

# play with cvs
# cv = KFold(n_splits=3, shuffle=True, random_state=42)
cv = ShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
scorer = make_scorer(mean_squared_error)
t0 = perf_counter()
with skrub.config(scheduler=True, stats=True):
    results = preds.skb.make_grid_search(cv=cv, n_jobs=1, fitted=True)
t1 = perf_counter()
print("="*80)
print(f"Stratum gridsearch scheduler time: {t1 - t0} seconds")
print("="*80)
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    search = preds.skb.make_grid_search(cv=cv, n_jobs=-1, fitted=True, scoring=scorer, refit=False)
t2 = perf_counter()
print("="*80)
print(f"Skrub default gridsearch time: {t2 - t1} seconds")
print("="*80)
print("Results:")
print(search.results_)
print(results)
print("="*80)