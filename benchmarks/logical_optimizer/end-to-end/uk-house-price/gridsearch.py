from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
import skrub
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lars, Lasso, Ridge

from stratum.runtime import grid_search
from stratum.logical_optimizer import optimize
from time import perf_counter

test=True

file_path = "tmp.csv" if test else "price_paid_records.csv"
df = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)
print(df.columns.skb.preview())
y = df["Price"].skb.mark_as_y()
X = df.drop("Price", axis=1).skb.mark_as_X()

def pre_process(X):
    date = X["Date of Transfer"].skb.apply_func(pd.to_datetime)
    X = X.assign(year=date.dt.year, month=date.dt.month, day=date.dt.day, dayofweek=date.dt.dayofweek, hour=date.dt.hour, minute=date.dt.minute, second=date.dt.second)
    X = X.drop(["Date of Transfer", 'Duration', 'Transaction unique identifier', 'PPDCategory Type', 'Record Status - monthly file only'], axis=1)
    return X


# pipeline 0
X2 = pre_process(X)
vec = skrub.TableVectorizer()
X_vec = X2.skb.apply(vec)
# model = RandomForestRegressor(random_state=42, n_estimators=20, max_depth=10)
model = ElasticNet(random_state=42)
pred0 = X_vec.skb.apply(model, y=y)

# pipeline 1
X2 = pre_process(X)
vec = skrub.TableVectorizer()
X_vec = X2.skb.apply(vec)
model = Lars(random_state=42)
pred1 = X_vec.skb.apply(model, y=y)

# pipeline 2
X2 = pre_process(X)
vec = skrub.TableVectorizer()
X_vec = X2.skb.apply(vec)
model = Ridge(random_state=42)
pred2 = X_vec.skb.apply(model, y=y)

# pipeline 3
X2 = pre_process(X)
vec = skrub.TableVectorizer()
X_vec = X2.skb.apply(vec)
model = Lasso(random_state=42)
pred3 = X_vec.skb.apply(model, y=y)

# pipeline 4
X2 = pre_process(X)
vec = skrub.TableVectorizer()
X_vec = X2.skb.apply(vec)
model = BayesianRidge()
pred4 = X_vec.skb.apply(model, y=y)

preds = {
    "pipeline0": pred0,
    "pipeline1": pred1,
    "pipeline2": pred2,
    "pipeline3": pred3,
    "pipeline4": pred4,
}

merged_pipelines = skrub.choose_from(preds, name="merged pipelines").as_data_op().skb.set_name("GridSearchCV")

pipes = optimize(merged_pipelines)

cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

t0 = perf_counter()
results = grid_search(pipes, cv=cv, print_heavy_hitters=True)
t1 = perf_counter()
print(f"Time taken Stratum: {t1 - t0} seconds")

t0 = perf_counter()
search = pipes.skb.make_grid_search(cv=cv, n_jobs=-1, fitted=True)
t1 = perf_counter()
print(f"Time taken Skrub [n_jobs=-1]: {t1 - t0} seconds")


t0 = perf_counter()
search = pipes.skb.make_grid_search(cv=cv, n_jobs=1, fitted=True)
t1 = perf_counter()
print(f"Time taken Skrub [n_jobs=1]: {t1 - t0} seconds")
