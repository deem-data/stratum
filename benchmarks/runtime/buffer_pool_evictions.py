"""Buffer-pool benchmark; CPU/RSS timelines are saved under runtime/profiles/.

Requires the stratum-ai[hardware] extra. Each run produces a profiler JSON report.
"""
import argparse
from time import perf_counter, sleep

import numpy as np
from sklearn.dummy import DummyRegressor
from stratum._profiler import record_run

import stratum as st
from sklearn.model_selection import ShuffleSplit
import pathlib

def plus_one(x, sleep_time=0):
    sleep(sleep_time)
    return x + 1
        

def row_sum(x, sleep_time=0):
    sleep(sleep_time)
    return x.sum(axis=1)

def multi_add(*args):
    i = 0
    for arg in args:
        i += arg
    return i

def pipe0(arr, y):
    arr2 = arr.skb.apply_func(plus_one, 0.30)
    arr3 = arr.skb.apply_func(plus_one, 0.31)
    arr4 = arr.skb.apply_func(plus_one, 0.32)
    arr5 = arr.skb.apply_func(plus_one, 0.33)
    tmp = arr2.skb.apply_func(multi_add, arr3, arr4, arr5)
    pred = tmp.skb.apply(DummyRegressor(), y=y)
    return pred

def fully_connected_binary_op_layer(nodes):
    new_layer = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            new_layer.append(nodes[i] + nodes[j])
    return new_layer

def pipe1(arr, y):
    # layer 1
    l1 = [arr.skb.apply_func(plus_one, 0.30),
          arr.skb.apply_func(plus_one, 0.31),
          arr.skb.apply_func(plus_one, 0.32),
          arr.skb.apply_func(plus_one, 0.33),
          arr.skb.apply_func(plus_one, 0.34),
          arr.skb.apply_func(plus_one, 0.35),
          ]

    # layer 2
    l2 = fully_connected_binary_op_layer(l1)

    # layer 3
    l3 = []
    offset = len(l2) // 2
    for i in range(offset):
        tmp = l2[i] + l2[i + offset]
        l3.append(tmp.skb.apply_func(row_sum, 0.36))

    final = l3[0].skb.apply_func(multi_add, *l3[1:])

    pred = final.skb.apply(DummyRegressor(), y=y)
    return pred

def pipe2(arr, y):
    arr1_1 = arr.skb.apply_func(plus_one, 2.0)
    arr1_2 = arr1_1.skb.apply_func(plus_one, 2.0)
    arr1_3 = arr1_2.skb.apply_func(plus_one, 2.0)
    arr1_4 = arr1_3.skb.apply_func(plus_one, 2.0)

    arr2_2 = arr1_1.skb.apply_func(plus_one, 2.1)
    arr2_3 = arr2_2.skb.apply_func(plus_one, 2.1)
    arr2_4 = arr2_3.skb.apply_func(plus_one, 2.1)

    arr3_3 = arr1_2.skb.apply_func(plus_one, 2.2)
    arr3_4 = arr3_3.skb.apply_func(plus_one, 2.2)

    ar4_1 = arr.skb.apply_func(plus_one, 2.3)
    ar4_2 = ar4_1.skb.apply_func(plus_one, 2.3)
    ar4_3 = ar4_2.skb.apply_func(plus_one, 2.3)
    ar4_4 = ar4_3.skb.apply_func(plus_one, 2.3)

    final = arr1_4.skb.apply_func(multi_add, arr2_4, arr3_4, ar4_4)
    pred = final.skb.apply(DummyRegressor(), y=y)
    return pred


def pipe3(arr, y):
    arr1_1 = arr.skb.apply_func(plus_one, 0.30)
    arr1_2 = arr1_1.skb.apply_func(plus_one, 0.30)
    arr1_3 = arr1_2.skb.apply_func(plus_one, 0.30)
    arr1_4 = arr1_3.skb.apply_func(plus_one, 0.30)

    pred = arr1_4.skb.apply(DummyRegressor(), y=y)
    return pred

def pipe4(arr, y):
    # layer 1
    n = 8
    l1 = [arr.skb.apply_func(plus_one, 0.3 + n*0.01) for _ in range(n)]

    # layer 2
    l2 = fully_connected_binary_op_layer(l1)

    # layer 3
    l3 = []
    offset = len(l2) // 2
    for i in range(offset):
        tmp = l2[i] + l2[i + offset]
        l3.append(tmp.skb.apply_func(row_sum, 0.36))

    final = l3[0].skb.apply_func(multi_add, *l3[1:])

    pred = final.skb.apply(DummyRegressor(), y=y)
    return pred

def main(use_skrub: bool, memory_fraction: float, n_rows: int, n_cols: int, pipe: int, draw: bool):
    # check if input exists
    x_path = pathlib.Path(f"input_x_{n_rows}_{n_cols}.npy")
    y_path = pathlib.Path(f"input_y_{n_rows}.npy")
    if not x_path.exists():
        np.save(x_path, np.random.random((n_rows, n_cols)).astype(np.float64), allow_pickle=False)
        np.save(y_path, np.random.random(n_rows).astype(np.float64), allow_pickle=False)

    with st.config_context(eager_data_ops=False):
        arr = st.as_data_op(x_path.resolve().as_posix()).skb.apply_func(np.load)
        y = st.as_data_op(y_path.resolve().as_posix()).skb.apply_func(np.load)
        y = y.skb.mark_as_y()
        arr = arr.skb.mark_as_X()
        if pipe == 0:
            pred = pipe0(arr, y)
        elif pipe == 1:
            pred = pipe1(arr, y)
        elif pipe == 2:
            pred = pipe2(arr, y)
        elif pipe == 3:
            pred = pipe3(arr, y)
        elif pipe == 4:
            pred = pipe4(arr, y)
        else:
            raise ValueError(f"Invalid pipe: {pipe}")

        if draw:
            pred.skb.draw_graph().open()
            exit()

        backend = 'skrub' if use_skrub else 'stratum'
        diagnostics_dir = pathlib.Path(__file__).parent / "profiles" / f"evictions_{backend}_pipe{pipe}_memory{memory_fraction}"
        run_search = pred.skb.make_grid_search
        if use_skrub:
            # Skrub bypasses Stratum's API profiling hooks.
            run_search = record_run("benchmark")(run_search)
        t0 = perf_counter()
        cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        with st.config(
            scheduler=not use_skrub, buffer_pool_memory_fraction=memory_fraction,
            stats=True, DEBUG=True,
        ), st.dev_config(
            hardware_utilization=True, hardware_sampling_interval=0.02,
            diagnostics_dir=diagnostics_dir,
        ):
            search = run_search(cv=cv, scoring="r2", fitted=True, refit=False)
        t1 = perf_counter()

    print(f"Time taken: {t1 - t0:.2f}s")
    print(search.results_)
    print(f"Profiler reports: {diagnostics_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skrub", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.7,
                        help="Fraction of system memory for the buffer pool (0 < value <= 1)")
    parser.add_argument("--n-rows", type=int, default=30_000_000)
    parser.add_argument("--n-cols", type=int, default=100)
    parser.add_argument("--pipe", type=int, default=1, help="Pipeline to run")
    parser.add_argument("--draw", action="store_true", help="Draw the graph")
    args = parser.parse_args()
    main(
        use_skrub=args.skrub,
        memory_fraction=args.memory_fraction,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        pipe=args.pipe,
        draw=args.draw,
    )
