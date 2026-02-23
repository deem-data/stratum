from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from time import perf_counter
from xgboost import XGBRegressor
from dask_ml.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

def is_numeric(col):
    return pd.api.types.is_numeric_dtype(
        col
    ) and not pd.api.types.is_bool_dtype(col)


t0 = perf_counter()
df = pd.read_csv("input/price_paid_records_1M.csv")

df = df.rename(columns={"Town/City": "Town"}, inplace=False)
y = df["Price"].to_numpy()
# Drop ID column so CV splitter gets numeric/categorical features only (no raw UUID strings)
date_col = pd.to_datetime(df["Date of Transfer"])
X = df.assign(
    year=date_col.dt.year, 
    month=date_col.dt.month, 
    day=date_col.dt.day, 
    dayofweek=date_col.dt.dayofweek, 
    hour=date_col.dt.hour)
X = X.assign(
    month_sin=(date_col.dt.month * (2 * np.pi / 12)).apply(np.sin),
    month_cos=(date_col.dt.month * (2 * np.pi / 12)).apply(np.cos),
    day_sin=(date_col.dt.day * (2 * np.pi / 30)).apply(np.sin),
    day_cos=(date_col.dt.day * (2 * np.pi / 30)).apply(np.cos),
    dayofweek_sin=(date_col.dt.dayofweek * (2 * np.pi / 7)).apply(np.sin),
    dayofweek_cos=(date_col.dt.dayofweek * (2 * np.pi / 7)).apply(np.cos),
    hour_sin=(date_col.dt.hour * (2 * np.pi / 24)).apply(np.sin),
    hour_cos=(date_col.dt.hour * (2 * np.pi / 24)).apply(np.cos),
)
X = df.drop(columns=[
    "Price",
    "Transaction unique identifier",
    "Date of Transfer",
    "Duration", 
    "PPDCategory Type", 
    "Record Status - monthly file only"
], axis=1)

string_enc = Pipeline([
    ("tfidf", TfidfVectorizer(max_df=300)),
    ("svd", TruncatedSVD()),
])
num_cols = [col for col in X.columns if is_numeric(X[col])]
cat_cols = [col for col in X.columns if not is_numeric(X[col])]
enc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop='if_binary', dtype='float32', handle_unknown='ignore', sparse_output=False), cat_cols)
])
pipe = Pipeline([
    ("pre", enc),
    ("model", LGBMRegressor())
])

param_grid = {"model": [
    LGBMRegressor(random_state=42),
    Ridge(random_state=42),
    XGBRegressor(random_state=42),
    ElasticNet(random_state=42),
]}

cv = KFold(n_splits=3, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, cv=3, scoring="r2", n_jobs=-1)
search.fit(X, y)

# Pretty print: pipeline (model) and score
results = search.cv_results_
print("\n--- GridSearchCV results (R²) ---")
for rank, i in enumerate(np.argsort(results["rank_test_score"]), start=1):
    model = results["param_model"][i]
    model_name = type(model).__name__
    mean_score = results["mean_test_score"][i]
    print(f"  #{rank}  {model_name:20s}  {mean_score:.4f}")
print(f"\nBest: {type(search.best_estimator_.named_steps['model']).__name__} "
      f"(R² = {search.best_score_:.4f})")

t1 = perf_counter()
print(f"Time taken: {t1 - t0} seconds")