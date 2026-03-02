from time import perf_counter

import skrub
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer

file_path = "input/price_paid_records_small.csv"
df_raw = pd.read_csv(file_path) #setting datatypes reduces size and speeds up
df = skrub.as_data_op(df_raw)

y = df["Price"].skb.mark_as_y()
X = df.drop("Price", axis=1).skb.mark_as_X()

vec = TableVectorizer()
X_enc = X.skb.apply(vec)
X_vec = X_enc.skb.apply(StandardScaler())
#X_vec2 = X_vec.astype('float64')

model = ElasticNet(random_state=42)
preds = X_vec.skb.apply(model, y=y)

cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
scorer = make_scorer(r2_score)
t0 = perf_counter()
search = preds.skb.make_grid_search(cv=cv, n_jobs=1, scoring=scorer, fitted=True)
t1 = perf_counter()
print(f"Time taken: {t1 - t0} seconds")
print(search.results_)
