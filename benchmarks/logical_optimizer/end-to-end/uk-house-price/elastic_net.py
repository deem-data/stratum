import polars as pl
from sklearn.preprocessing import StandardScaler
from time import perf_counter
import cProfile
import pstats
import io
from contextlib import contextmanager
import argparse
import os
from sklearn.linear_model import ElasticNet
from numpy import float64


@contextmanager
def profile_context(no_stats: bool = False):
    """Context manager for profiling code blocks"""
    if no_stats:
        yield None
        return
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()

def print_profile_stats(profiler, sort_by='cumulative', top_n=20):
    """Print profiling statistics"""
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats(sort_by)
    ps.print_stats(top_n)
    print(s.getvalue())

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="1M")
parser.add_argument("--polars", action="store_true")
parser.add_argument("--no", action="store_true")
parser.add_argument("--f64", action="store_true")
parser.add_argument("--max", default=1000, type=int)
args = parser.parse_args()
size = args.size

t0_total = perf_counter()
cwd = os.getcwd()
if not cwd.endswith("benchmarks/logical_optimizer/end-to-end/uk-house-price"):
    os.chdir("benchmarks/logical_optimizer/end-to-end/uk-house-price")

df_vec = pl.read_parquet(f"input/tv_vec_{size}.parquet")
price = df_vec["Price"]
df_vec = df_vec.drop("Price")

if not args.polars:
    df_vec = df_vec.to_pandas()

sc = StandardScaler()

x = sc.fit_transform(df_vec)
if args.f64:
    x = x.astype(float64)


model = ElasticNet(max_iter=args.max)

t0_tv_fit = perf_counter()
with profile_context(args.no) as prof_tv:
    pred = model.fit(x, price).predict(x)

t1_tv_fit = perf_counter()
if not args.no:
    print_profile_stats(prof_tv, sort_by='cumulative', top_n=100)
print(f"Time taken for ElasticNet fit_predict: {t1_tv_fit - t0_tv_fit:.4f} seconds")