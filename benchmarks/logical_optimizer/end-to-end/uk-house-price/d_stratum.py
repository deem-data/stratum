import stratum as skrub
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from time import perf_counter
from xgboost import XGBRegressor


t0 = perf_counter()
file_path = "input/price_paid_records_1M.csv"
df = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)

df = df.rename(columns={"Town/City": "Town"}, inplace=False)
y = df["Price"].skb.mark_as_y()
X = df.drop(["Price", "Transaction unique identifier"], axis=1).skb.mark_as_X()

X_enc = X.skb.apply(skrub.TableVectorizer(
    n_jobs=1,
    high_cardinality=skrub.StringEncoder(), 
    low_cardinality=OneHotEncoder(drop='if_binary', dtype='float32', handle_unknown='ignore', sparse_output=False))
)

models = [
    LGBMRegressor(random_state=42),
    Ridge(random_state=42),
    XGBRegressor(random_state=42),
    ElasticNet(random_state=42),
]

preds = {f"{i}": X_enc.skb.apply(model, y=y) for i, model in enumerate(models)}
preds = skrub.choose_from(preds, name="model").as_data_op()
scorer = make_scorer(r2_score)
t1 = perf_counter()
cv = KFold(n_splits=3, shuffle=True, random_state=42)
with skrub.config(scheduler=True, stats=20, rust_backend=False, scheduler_parallelism=None,force_polars=False,):
    search_stratum = preds.skb.make_grid_search(cv=cv, n_jobs=1, fitted=True, scoring=scorer)
t2 = perf_counter()
print(f"Preview time taken: {t1 - t0} seconds")
print(f"Search time taken: {t2 - t1} seconds")
print(search_stratum.results_)