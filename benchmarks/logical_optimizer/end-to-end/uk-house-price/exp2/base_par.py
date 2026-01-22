from types import NoneType
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, ShuffleSplit

import skrub, pandas as pd
from multiprocessing import Process
import argparse
from time import perf_counter, sleep
import joblib

def make_pipeline(size, pipe_id, cv=3):
    import pipeline_definitions as pipe
    # Load data
    print(f"Making pipeline {pipe_id}")
    data_path = f"../input/price_paid_records_{size}.csv"
    data = skrub.var("data_path", data_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)

    y = data["Price"].skb.mark_as_y()
    X = data.drop(columns=["Price"]).skb.mark_as_X()
    X = X.skb.apply_func(lambda a: (a, print(f"RUNNING {pipe_id}"))[0])
    pred = getattr(pipe, f"pipeline{pipe_id}")(X, y)
    cv = ShuffleSplit(n_splits=cv, test_size=0.2, random_state=42) if cv == 1 else KFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = make_scorer(r2_score)
    search = pred.skb.make_grid_search(fitted=True, refit=False,scoring=scoring, cv=cv, n_jobs=1)
    return


def run_experiment(size, cv=3, sleep_time=1, max_parallelism=1, joblib_back_end=None):
    ids = [1, 2, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 30, 31, 32, 33]
    # launch each pipeline in a separate process
    t0 = perf_counter()
    if joblib_back_end is not None:
        print(f"Using joblib backend: {joblib_back_end}")
        with joblib.Parallel(n_jobs=max_parallelism, backend=joblib_back_end) as parallel:
            parallel(joblib.delayed(make_pipeline)(size, id, cv) for id in ids)
    else:
        print(f"Using multiprocessing")
        for i in range(len(ids) // max_parallelism):
            processes = {}
            for id in ids[i * max_parallelism:(i + 1) * max_parallelism]:
                process = Process(target=make_pipeline, args=(size, id, cv))
                process.start()
                processes[id] = process
            # check every 10s which processes are still running
            while len(processes) > 0:
                running = []
                processes_copy = processes.copy()
                for id, process in processes_copy.items():
                    if process.is_alive():
                        running.append(id)
                    else:
                        print(f"Process {id} has finished")
                        del processes[id]
                print(f"Running processes: {running}")
                sleep(sleep_time)
    t1 = perf_counter()
    print(f"Time taken: {t1 - t0} seconds")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="100K")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--max_parallelism", type=int, default=1)
    parser.add_argument("--sleep_time", type=int, default=1)
    parser.add_argument("--jlib",  type=str, default=None)
    args = parser.parse_args()
    run_experiment(args.size, args.cv, args.sleep_time, args.max_parallelism, args.jlib)