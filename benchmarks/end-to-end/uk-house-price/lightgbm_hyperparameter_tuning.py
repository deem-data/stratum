import cProfile
from sklearn.metrics import make_scorer, r2_score

import stratum as skrub
from skrub import StringEncoder
from sklearn.model_selection import KFold
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from time import perf_counter

pr = cProfile.Profile()
test=True

dtypes = {
    "Transaction unique identifier": "category",
    "Price": "int32",
    "Property Type": "category",
    "Old/New": "category",
    "Duration": "category",
    "Town/City": "category",
    "District": "category",
    "Country": "category",
    "County": "category",
    "PPDCategory Type": "category",
    "Record Status - monthly file only": "category"
}
file_path = "input/price_paid_records_small.csv" if test else "input/price_paid_records.csv"
df = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)
print(df.columns.skb.preview())
df = df.rename(columns={"Town/City": "Town"}, inplace=False)
y = df["Price"].skb.mark_as_y()
X = df.drop("Price", axis=1).skb.mark_as_X()

def pre_process_2(X):
    X_enc = X.skb.apply(skrub.TableVectorizer(high_cardinality=StringEncoder(), low_cardinality=OneHotEncoder(), n_jobs=-1))
    # Scaling is necessary for ElasticNet and Ridge (converge quick and fast)
    X_vec = X_enc.skb.apply(StandardScaler())
    return X_vec

X_enc = pre_process_2(X)

models = {
    "LightGBM_lr0.01_maxd5": LGBMRegressor(learning_rate=0.01, max_depth=5, random_state=42),
    "LightGBM_lr0.01_maxd7": LGBMRegressor(learning_rate=0.01, max_depth=7, random_state=42),
    "LightGBM_lr0.05_maxd5": LGBMRegressor(learning_rate=0.05, max_depth=5, random_state=42),
    "LightGBM_lr0.05_maxd7": LGBMRegressor(learning_rate=0.05, max_depth=7, random_state=42),
    "LightGBM_lr0.1_maxd5": LGBMRegressor(learning_rate=0.1, max_depth=5, random_state=42),
    "LightGBM_lr0.1_maxd7": LGBMRegressor(learning_rate=0.1, max_depth=7, random_state=42),
}
preds = {k: X_enc.skb.apply(model, y=y) for k,model in models.items()}
preds = skrub.choose_from(preds, name="preds").as_data_op()
scorer = make_scorer(r2_score)
cv = KFold(n_splits=3, shuffle=True, random_state=42)
skrub.set_config(rust_backend=True, debug_timing=False, scheduler=True, stats=True)

t0 = perf_counter()
search = preds.skb.make_grid_search(cv=cv, scoring=scorer, n_jobs=1, fitted=True)
t1 = perf_counter()
print(f"Time taken: {t1 - t0} seconds")
print(search.results_)
