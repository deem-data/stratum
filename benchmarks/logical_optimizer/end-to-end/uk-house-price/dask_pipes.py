from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
import skrub
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from time import perf_counter
from xgboost import XGBRegressor
from dask_ml.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder


t0 = perf_counter()
df = pd.read_csv("input/price_paid_records_1M.csv")

df = df.rename(columns={"Town/City": "Town"}, inplace=False)
y = df["Price"].to_numpy()
# Drop ID column so CV splitter gets numeric/categorical features only (no raw UUID strings)
X = df.drop(columns=["Price", "Transaction unique identifier"])

tv = skrub.TableVectorizer(
    n_jobs=1,
    high_cardinality=skrub.StringEncoder(), 
    low_cardinality=OneHotEncoder(drop='if_binary', dtype='float32', handle_unknown='ignore', sparse_output=False)
)


pipe = Pipeline([
    ("pre", tv),
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