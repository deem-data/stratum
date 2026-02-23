import argparse
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import skrub
from sklearn.dummy import DummyRegressor
from time import perf_counter
import joblib

class MyDummyRegressor(DummyRegressor):
    def __init__(self,  id: str):
        super().__init__()
        self.id = id


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="100K")
parser.add_argument("--cse", action="store_true")
parser.add_argument("--pipes", type=int, default=1)
parser.add_argument("--max_par", type=int, default=1)
parser.add_argument("--jlib", type=str, default=None)
args = parser.parse_args()

size = args.size
if len(size) > 0:
    size = "_" + size

# Load data
def make_pipe(id: str, size: str):
    t0 = perf_counter()
    data_path = f"../input/price_paid_records{size}.csv"
    data = skrub.var("data_path", data_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)

    y = data["Price"].skb.mark_as_y()
    X = data.drop(columns=["Price"]).skb.mark_as_X()

    pred = X.skb.apply(skrub.TableVectorizer(n_jobs=1)).skb.apply(MyDummyRegressor(id=id), y=y)
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    pred.skb.make_grid_search(cv=cv, n_jobs=1, fitted=True)
    t1 = perf_counter()
    print(f"Worker {id} took {t1 - t0} seconds")


t0 = perf_counter()
with joblib.Parallel(n_jobs=args.max_par, backend=args.jlib) as parallel:
    parallel(joblib.delayed(make_pipe)(id, size) for id in range(args.pipes))

t1 = perf_counter()
print(f"Time taken: {t1 - t0} seconds")
