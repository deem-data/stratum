import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import skrub
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from stratum.logical_optimizer import optimize
from stratum.search import grid_search
from time import time

def pipeline_definition(show_graph=False):
    df_path = "input/train.csv"
    target = "MedHouseVal"

    # 2. Skrub DataOps plan (feature engineering in pipeline)
    df_path = skrub.as_data_op(df_path)
    df = df_path.skb.apply_func(pd.read_csv).skb.subsample(n=100)

    # Mark y and X
    y = df[target].skb.mark_as_y()
    X = df.drop(columns=[target]).skb.mark_as_X()
    # X = X.skb.apply_func(lambda x: (x, print("hello"))[0])

    # Feature engineering steps (each as a separate operation)
    def feat_eng(X):
        return X.assign(BedroomsPerRoom=X["AveBedrms"] / X["AveRooms"],
        IncomeSquared=X["MedInc"] ** 2,
        IncomeRoomInteraction=X["MedInc"] * X["AveRooms"],
        Density=X["Population"] / X["AveOccup"],
        LatitudeLongitude=X["Latitude"] * X["Longitude"],
        MedInc3=X["MedInc"] ** 3,
        RoomDensity=X["AveRooms"] / X["Population"]
    )

    # pipeline 0
    X2 = feat_eng(X)
    scaler = StandardScaler()
    X_scaled = X2.skb.apply(scaler)
    pred0 = X_scaled.skb.apply(LinearRegression(), y=y)

    # Pipeline 1
    X2 = feat_eng(X)
    scaler = StandardScaler()
    X_scaled = X2.skb.apply(scaler) 
    pred1 = X_scaled.skb.apply(Ridge(), y=y)

    # Pipeline 2
    X2 = feat_eng(X)
    scaler = StandardScaler()
    X_scaled = X2.skb.apply(scaler)
    pred2 = X_scaled.skb.apply(Lasso(), y=y)

    # Pipeline 3
    X2 = feat_eng(X)
    scaler = StandardScaler()
    X_scaled = X2.skb.apply(scaler)
    pred3 = X_scaled.skb.apply(ElasticNet(), y=y)

    preds = {
        "pipeline0": pred0,
        "pipeline1": pred1,
        "pipeline2": pred2,
        "pipeline3": pred3,
    }
    pred = skrub.choose_from(preds, name="predictions").as_data_op()
    if show_graph:
        pred.skb.draw_graph().open()
    
    
    return pred


def run_experiment(pred, show_graph=False):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    runs = 1

    def run_and_average(name, search_func, print_results=True):
        times = []
        search_result = None
        for run_idx in range(runs):
            t0 = time()
            search_result = search_func()
            t1 = time()
            times.append(t1 - t0)
            if runs > 1:
                print(f"  Run {run_idx + 1}/{runs}: {t1 - t0:.4f}s")
        
        avg_time = np.mean(times)
        std_time = np.std(times) if runs > 1 else 0
        print(f"Gridsearch time (avg over {runs} runs): {avg_time:.4f}s" + 
                (f" (std: {std_time:.4f}s)" if runs > 1 else ""))
        
        if print_results and search_result is not None:
            if hasattr(search_result, 'results_'):
                print(search_result.results_)
            else:
                print(search_result)
            print("----------------------------------------")
        
        return {"impl": name, "time": avg_time}

    df_vals = []

    # Skrub - non optimized gridsearch (n_jobs=1)
    print("Skrub - non optimized gridsearch (n_jobs=1)")
    df_vals.append(run_and_average(
        "skrub-njobs=1",
        lambda: pred.skb.make_grid_search(fitted=True, cv=cv, n_jobs=1, scoring="neg_mean_squared_error", refit=False)
    ))

    # Skrub - non optimized gridsearch (n_jobs=-1)
    print("Skrub - non optimized gridsearch (n_jobs=-1)")
    df_vals.append(run_and_average(
        "skrub-njobs=-1",
        lambda: pred.skb.make_grid_search(fitted=True, cv=cv, n_jobs=-1, scoring="neg_mean_squared_error", refit=False)
    ))

    # Stratum - gridsearch (n_jobs=1)
    print("Stratum - optimized gridsearch (n_jobs=1)")
    df_vals.append(run_and_average(
        "stratum-njobs=1",
        lambda: grid_search(pred, cv=cv, scoring="neg_mean_squared_error")
    ))

    # Optimization step (only run once)
    t00 = time()
    pred_optimized = optimize(pred)
    t01 = time()
    print("Optimization time: ", t01 - t00) 

    if show_graph:
        pred_optimized.skb.draw_graph().open()

    # Skrub - optimized gridsearch (n_jobs=1)
    print("Skrub - optimized gridsearch (n_jobs=1)")
    df_vals.append(run_and_average(
        "skrub-optimized-njobs=1",
        lambda: pred_optimized.skb.make_grid_search(fitted=True, cv=cv, n_jobs=1, scoring="neg_mean_squared_error", refit=False)
    ))

    # Skrub - optimized gridsearch (n_jobs=-1)
    print("Skrub - optimized gridsearch (n_jobs=-1)")
    df_vals.append(run_and_average(
        "skrub-optimized-njobs=-1",
        lambda: pred_optimized.skb.make_grid_search(fitted=True, cv=cv, n_jobs=-1, scoring="neg_mean_squared_error", refit=False)
    ))

    # Stratum - optimized gridsearch (n_jobs=1)
    print("Stratum - optimized gridsearch (n_jobs=1)")
    df_vals.append(run_and_average(
        "stratum-optimized-njobs=1",
        lambda: grid_search(pred_optimized, cv=cv, scoring="neg_mean_squared_error")
    ))

    df_vals.append({"impl": "baseline", "time": 2.91})

    df = pd.DataFrame(df_vals)

    print("\nSummary:")
    print(df)


show_graph = True
t0 = time()
pred = pipeline_definition(show_graph=show_graph)
t1 = time()
print("Pipeline definition time: ", t1 - t0)
run_experiment(pred, show_graph=show_graph)