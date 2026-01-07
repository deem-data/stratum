from time import perf_counter
import pandas as pd
from joblib import parallel_backend
#import skrub
import stratum as skrub
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from skrub import StringEncoder, TableVectorizer
import cProfile
import pstats
pr = cProfile.Profile()

# 1. Load Data
file_path = "input/price_paid_records_small.csv"
df_raw = pd.read_csv(file_path)
df = skrub.as_data_op(df_raw)

y = df["Price"].skb.mark_as_y()
X = df.drop("Price", axis=1).skb.mark_as_X()

# 3. Pre-processing (pre_process_2 logic)
vec = TableVectorizer(
    high_cardinality=StringEncoder(),
    low_cardinality=OneHotEncoder(drop='if_binary', dtype='float32', handle_unknown='ignore', sparse_output=False)
)
X_enc = X.skb.apply(vec)
X_enc = X_enc.skb.apply_func(lambda x, m: (x, print(m))[0], skrub.eval_mode())

# 4. Modeling
model = LGBMRegressor(random_state=42)
preds = X_enc.skb.apply(model, y=y)

# 5. Grid search
skrub.set_config(rust_backend=True, debug_timing=False)
cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
t0 = perf_counter()
#pr.enable()
search = preds.skb.make_grid_search(cv=cv, n_jobs=1, fitted=True)
#pr.disable()
t1 = perf_counter()
print(f"Time taken: {t1 - t0} seconds")
print(search.results_)

#stats = pstats.Stats(pr).sort_stats("tottime")
#stats.print_stats(60)
