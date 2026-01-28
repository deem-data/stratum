from joblib import parallel_backend
import stratum as skrub
import polars as pl
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from time import perf_counter
import cProfile
import pstats
import io
from contextlib import contextmanager
import argparse
import os


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
parser.add_argument("--size", type=str, default="100K")
parser.add_argument("--rust", action="store_true")
parser.add_argument("--polars", action="store_true")
parser.add_argument("--no", action="store_true")
args = parser.parse_args()
size = args.size

t0_total = perf_counter()
cwd = os.getcwd()
if not cwd.endswith("benchmarks/logical_optimizer/end-to-end/uk-house-price"):
    os.chdir("benchmarks/logical_optimizer/end-to-end/uk-house-price")

df = pl.read_csv(f"input/price_paid_records_{size}.csv") if args.polars else pd.read_csv(f"input/price_paid_records_{size}.csv")
price = df["Price"] if args.polars else df["Price"].to_numpy()
df = df.drop("Price") if args.polars else df.drop("Price", axis=1)

tv = skrub.TableVectorizer(
    n_jobs=1,
    high_cardinality=skrub.StringEncoder(), 
    low_cardinality=OneHotEncoder(drop='if_binary', dtype='float32', handle_unknown='ignore', sparse_output=False)
)

t0_tv_fit = perf_counter()
with profile_context(args.no) as prof_tv:
    if args.rust:
        with skrub.config(rust_backend=True), parallel_backend('threading'):
            df_vec = tv.fit_transform(df)
    else:
        with skrub.config(rust_backend=False):
            df_vec = tv.fit_transform(df)

t1_tv_fit = perf_counter()
if not args.no:
    print_profile_stats(prof_tv, sort_by='cumulative', top_n=100)
print(f"Time taken for TableVectorizer fit_transform: {t1_tv_fit - t0_tv_fit:.4f} seconds")

if args.polars:
    df_vec = df_vec.with_columns(price)
    df_vec.write_parquet(f"input/tv_vec_{size}.parquet")