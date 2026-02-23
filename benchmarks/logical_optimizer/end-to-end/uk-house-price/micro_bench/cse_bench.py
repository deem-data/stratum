import argparse
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import stratum as skrub
from sklearn.dummy import DummyRegressor
from time import perf_counter

class MyDummyRegressor(DummyRegressor):
    def __init__(self,  id: str):
        super().__init__()
        self.id = id


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="100K")
parser.add_argument("--cse", action="store_true")
parser.add_argument("--pipes", type=int, default=1)
args = parser.parse_args()

size = args.size
if len(size) > 0:
    size = "_" + size

t0 = perf_counter()
# Load data
data_path = f"../input/price_paid_records{size}.csv"
data = skrub.var("data_path", data_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)

y = data["Price"].skb.mark_as_y()
X = data.drop(columns=["Price"]).skb.mark_as_X()

preds = {f"{i}": X.skb.apply(skrub.TableVectorizer(n_jobs=1)).skb.apply(MyDummyRegressor(id=f"pipeline{i}"), y=y) for i in range(args.pipes)}
pred = skrub.choose_from(preds, name="pipe").as_data_op()

cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
with skrub.config(scheduler=True, stats=20, rust_backend=False, scheduler_parallelism=None,force_polars=False,cse=args.cse,DEBUG=False):
    search = pred.skb.make_grid_search(cv=cv, n_jobs=1, fitted=True)
t1 = perf_counter()
print(f"Time taken: {t1 - t0} seconds")
print(search.results_)

