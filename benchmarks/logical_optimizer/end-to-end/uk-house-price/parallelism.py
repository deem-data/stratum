from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import polars as pl
from time import perf_counter
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import Parallel, delayed, parallel_backend
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="300K")
parser.add_argument("--seq", action="store_true")
parser.add_argument("--tvan", action="store_true")
parser.add_argument("--pvan", action="store_true")
parser.add_argument("--tjob", action="store_true")
parser.add_argument("--pjob", action="store_true")
parser.add_argument("--ljob", action="store_true")
parser.add_argument("--m", action="store_true")
args = parser.parse_args()

size = args.size

models = [Ridge(random_state=42), XGBRegressor(random_state=42, verbosity=0), LGBMRegressor(random_state=42, verbose=-1), ElasticNet(random_state=42)]
models2 = [Ridge(random_state=69), XGBRegressor(random_state=69, verbosity=0), LGBMRegressor(random_state=69, verbose=-1), ElasticNet(random_state=69)]
models3 = [LGBMRegressor(random_state=42, verbose=-1), LGBMRegressor(random_state=69, verbose=-1), LGBMRegressor(random_state=123, verbose=-1), LGBMRegressor(random_state=420, verbose=-1)]
df = pl.read_parquet(f"input/tmp_{size}.parquet")
df = df.to_pandas()
y = df["Price"]
df = df.drop("Price", axis=1)

def task(model, data, y, verbose=False):
    try:
        t0_ = perf_counter()
        model.fit(data, y)
        t1_fit = perf_counter()
        if verbose:
            print(f"Time taken to fit: {t1_fit - t0_} seconds")
        out = model.predict(data)
        if verbose:
            print(f"Time taken to predict: {perf_counter() - t1_fit} seconds")
        return out
    except Exception as e:
        print(f"Error: {e}")
        return None
# def task(model, data, y, verbose=False):
#     print("Hi")


def sequential(models):
    for model in models:
        task(model, df, y)

def threading_vanilla(models):
    print("Threading Vanilla")
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        list(executor.map(task, models, [df] * len(models), [y] * len(models)))

def process_vanilla(models):
    print("Multiprocessing Vanilla")
    with ProcessPoolExecutor(max_workers=len(models)) as executor:
        list(executor.map(task, models, [df] * len(models), [y] * len(models), [True] * len(models)))

def threading_joblib(models):
    print("Threading Joblib")
    with parallel_backend('threading'):
        Parallel(n_jobs=-1)(delayed(task)(model, df, y) for model in models)

def process_joblib(models):
    print("Process Joblib")
    with parallel_backend('multiprocessing'):
        Parallel(n_jobs=-1)(delayed(task)(model, df, y) for model in models)

def loky_joblib(models):
    print("Loky Joblib")
    with parallel_backend('loky'):
        Parallel(n_jobs=-1)(delayed(task)(model, df, y) for model in models)


def main(args):
    seq = args.seq
    thread_van = args.tvan
    process_van = args.pvan
    thread_joblib = args.tjob
    proc_joblib = args.pjob
    loky_joblib_ = args.ljob

    global models, models3, models2
    models = models3 if args.m else models + models2 
    if seq:
        t0 = perf_counter()
        sequential(models)
        t1 = perf_counter()
        print(f"Sequential Time taken: {t1 - t0} seconds")

    if thread_van:
        t0 = perf_counter()
        threading_vanilla(models)
        t1 = perf_counter()
        print(f"Threading Vanilla Time taken: {t1 - t0} seconds")

    if process_van:
        t0 = perf_counter()
        process_vanilla(models)
        t1 = perf_counter()
        print(f"Process Vanilla Time taken: {t1 - t0} seconds")

    if thread_joblib:
        t0 = perf_counter()
        threading_joblib(models)
        t1 = perf_counter()
        print(f"Threading Joblib Time taken: {t1 - t0} seconds")
    
    if proc_joblib:
        t0 = perf_counter()
        process_joblib(models)
        t1 = perf_counter()
        print(f"Process Joblib Time taken: {t1 - t0} seconds")

    if loky_joblib_:
        t0 = perf_counter()
        loky_joblib(models)
        t1 = perf_counter()
        print(f"Loky Joblib Time taken: {t1 - t0} seconds")

if __name__ == "__main__":
    main(args)