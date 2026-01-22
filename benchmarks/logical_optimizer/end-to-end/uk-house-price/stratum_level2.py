from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold
import pandas as pd
from lightgbm import LGBMRegressor

from time import perf_counter
import stratum as skrub
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
import os

# check if we are in the correct directory
if not os.getcwd().endswith("uk-house-price"):
    print(f"Changing directory to benchmarks/logical_optimizer/end-to-end/uk-house-price/")
    os.chdir("benchmarks/logical_optimizer/end-to-end/uk-house-price/")

file_path = sys.argv[1]
df = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)
df = df.rename(columns={"Town/City": "Town"}, inplace=False)
y = df["Price"].skb.mark_as_y()
X = df.drop("Price", axis=1).skb.mark_as_X()

def pre_process_2(X):
    X_enc = X.skb.apply(skrub.TableVectorizer(
        n_jobs=-1,
        high_cardinality=skrub.StringEncoder(), 
        low_cardinality=OneHotEncoder(drop='if_binary', dtype='float32', handle_unknown='ignore', sparse_output=False),
        )
    )
    X_vec = X_enc.skb.apply(StandardScaler())
    return X_vec

X_enc = pre_process_2(X)

models = {
    "LGBM_lr0.01_maxd5": LGBMRegressor(learning_rate=0.01, max_depth=5, random_state=42,verbose=-1),
    "LGBM_lr0.01_maxd7": LGBMRegressor(learning_rate=0.01, max_depth=7, random_state=42,verbose=-1),
    "LGBM_lr0.05_maxd5": LGBMRegressor(learning_rate=0.05, max_depth=5, random_state=42,verbose=-1),
    "LGBM_lr0.05_maxd7": LGBMRegressor(learning_rate=0.05, max_depth=7, random_state=42,verbose=-1),
    "LGBM_lr0.1_maxd5": LGBMRegressor(learning_rate=0.1, max_depth=5, random_state=42,verbose=-1),
    "LGBM_lr0.1_maxd7": LGBMRegressor(learning_rate=0.1, max_depth=7, random_state=42,verbose=-1),
}

preds = {k: X_enc.skb.apply(model, y=y) for k,model in models.items()}
preds = skrub.choose_from(preds, name="m").as_data_op()
# model = skrub.choose_from(models, name="m").as_data_op()
# preds = X_enc.skb.apply(model, y=y)
scorer = make_scorer(r2_score)
cv = KFold(n_splits=3, shuffle=True, random_state=42)

with skrub.config(rust_backend=True, debug_timing=False, scheduler=True, stats=20, scheduler_parallelism="threading", caching=True):
    t0 = perf_counter()
    search = preds.skb.make_grid_search(cv=cv, scoring=scorer, n_jobs=1, fitted=True,)
    t1 = perf_counter()
print(f"Time taken: {t1 - t0} seconds")
print(search.results_)