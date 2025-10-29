from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC

from stratum.logical_optimizer import optimize

import stratum as skrub
import logging
import tempfile
import numpy as np
import pandas as pd
from time import time

logging.basicConfig(level=logging.INFO)

def make_mixed_df(
    n_rows=10000,
    n_num=5,
    n_cat=2,
    n_text=1,
    n_classes=3,
    noise=0.1,
    random_state=42
):
    """
    Generate a synthetic DataFrame with mixed numeric, categorical, and text columns.
    Includes a numeric target column 'y' (regression style).
    """

    rng = np.random.default_rng(random_state)

    # --- Numeric features ---
    X_num = rng.normal(size=(n_rows, n_num))
    coefs = rng.uniform(-2, 2, size=n_num)
    y = X_num @ coefs + noise * rng.normal(size=n_rows)

    df = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(n_num)])

    # --- Categorical features ---
    for i in range(n_cat):
        df[f"cat_{i}"] = rng.choice(
            [f"cat{c}_{i}" for c in "ABCDEFGHIJKLMNOPQRSTUVXYZ"][:n_classes],
            size=n_rows
        )

    # --- Text features ---
    vocab = ["apple", "banana", "carrot", "date", "eggplant", "fig", "grape"]
    for i in range(n_text):
        df[f"text_{i}"] = [
            " ".join(rng.choice(vocab, size=rng.integers(3, 7)))
            for _ in range(n_rows)
        ]

    # Target
    df["y"] = y
    return df


def redundant_numeric_pipeline(df_path: str):
    path = skrub.var("path", df_path)
    data = path.skb.apply_func(pd.read_csv)
    y = data["y"].skb.mark_as_y()
    X = data.drop(["y"], axis=1).skb.mark_as_X()

    X_num = X.skb.select(skrub.selectors.filter_names(lambda name: name.startswith("num")))

    X_num4 = X_num.assign(**{f"quantile_75_{i}":X_num.quantile(0.75, axis=1) for i in range(30)})

    X_final = X_num4.fillna(X_num4.mean())


    pred = X_final.skb.apply(LinearRegression(), y=y)
    # pred.skb.draw_graph().open()
    print("----------------------------------------")
    stats = make_and_eval_learner(pred)
    print("\nOriginal Pipeline took: ",sum(stats))
    print("----------------------------------------")


    stats_opt = make_and_eval_learner(pred, optimize_enabled=True)
    print("\nTotal optimized Pipeline took: ",sum(stats_opt))
    print("----------------------------------------")
    # pred.skb.draw_graph().open()
    return np.array([stats, stats_opt])

from sklearn.base import BaseEstimator, TransformerMixin

class PandasTfidfVectorizer(BaseEstimator, TransformerMixin):
    """A CountVectorizer that returns a pandas DataFrame instead of a sparse matrix."""
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, X: pd.DataFrame, y=None):
        X = X.iloc[:,0]
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        X = X.iloc[:,0]
        X_counts = self.vectorizer.transform(X)
        df = pd.DataFrame.sparse.from_spmatrix(
            X_counts,
            columns=self.vectorizer.get_feature_names_out()
        )
        return df

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, *args, **kwargs):
        return self.vectorizer.get_feature_names_out(*args, **kwargs)

def tfidf_pipeline(df_path: str, show_graph: bool = False, gridsearch: bool = False):
    path = skrub.var("path", df_path)
    data = path.skb.apply_func(pd.read_csv).skb.subsample(n=100)
    data = data.fillna("")
    y = data["y"].skb.mark_as_y()
    X = data[["text"]].skb.mark_as_X()


    vectorizer = PandasTfidfVectorizer()

    mode = skrub.eval_mode()
    if gridsearch:
        pipes = {f"pipeline{i}":  X.skb.apply(vectorizer).
         skb.apply(model, y=y) for i, model in
         enumerate(
             [LinearRegression(),
              Ridge(),
              LogisticRegression(max_iter=1000),
              LinearSVC(),
              MultinomialNB(),
              ])}
        pred = skrub.choose_from(pipes).as_data_op()


    else:
        pipes = [X.skb.apply(vectorizer).
                 skb.apply(model, y=y).
                 skb.apply_func(lambda x, m: pd.DataFrame({f"pred{i}" : (x if m == "predict" else [1])}), mode)
                 for i, model in enumerate(
                    [LinearRegression(),
                     Ridge(),
                     LogisticRegression(max_iter=1000),
                     LinearSVC(),
                     MultinomialNB(),
                     ])]
        pred = pipes[0].skb.concat(pipes[1:])

    if show_graph:
        pred.skb.draw_graph().open()
    print("----------------------------------------")
    stats = make_gridsearch(pred) if gridsearch else make_and_eval_learner(pred)
    print("\nOriginal Pipeline took: ",sum(stats))
    print("----------------------------------------")


    stats_opt = make_gridsearch(pred, optimize_enabled=True) if gridsearch else make_and_eval_learner(pred, optimize_enabled=True)
    print("\nTotal optimized Pipeline took: ",sum(stats_opt))
    print("----------------------------------------")
    if show_graph:
        pred.skb.draw_graph().open()
    return np.array([stats, stats_opt])


def make_and_eval_learner(pred, random_state=42, optimize_enabled=False) -> tuple[list, list]:
    if optimize_enabled:
        t00 = time()
        pred = optimize(pred)
        t0 = time()
        stats = [t0 - t00]
    else:
        t0 = time()
        stats = [0]
    learner = pred.skb.make_learner(fitted=False)
    t1 = time()
    stats.append(t1-t0)
    split = pred.skb.train_test_split(test_size=0.2, random_state=random_state)
    t2 = time()
    stats.append(t2-t1)
    learner.fit(split["train"])
    t3 = time()
    stats.append(t3-t2)
    learner.predict(split["test"])
    t4 = time()
    stats.append(t4-t3)
    return stats

def make_gridsearch(pred, random_state=42, optimize_enabled=False) -> tuple[list, list]:
    if optimize_enabled:
        t00 = time()
        pred = optimize(pred)
        t0 = time()
        stats = [t0 - t00]
    else:
        t0 = time()
        stats = [0]

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    search = pred.skb.make_grid_search(fitted=True, cv=cv, n_jobs=1)
    print("Search results: \n", search.results_)
    t1 = time()
    stats.append(t1-t0)
    return stats


def run_tfidf_pipeline_benchmark(gridsearch: bool = False):
    data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    df = pd.DataFrame({"text": data.data, "y": data.target})
    df["text"].fillna("", inplace=True)

    list_of_stats = []
    parameters = [100, 500, 1000,2000, 5000, 10000] if not gridsearch else [100, 500, 1000,]
    for n_rows in parameters:
        df_n_rows = df.head(n_rows)
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".csv", delete=False) as f:
            df_n_rows.to_csv(f)
            f.flush()
            temp_path = f.name

        stats = tfidf_pipeline(temp_path, gridsearch = gridsearch, show_graph = False)
        stats = np.hstack((np.array([n_rows, n_rows]).reshape((2, 1)), stats))
        list_of_stats.append(stats)

    stats = np.vstack(list_of_stats)

    columns_results = ["n_rows","optimize"]
    columns_results += ["search"] if gridsearch else ["make_learner","split","train","predict"]
    df = pd.DataFrame(stats, columns=columns_results)

    columns_results.remove("n_rows")
    df["total"] = df[columns_results].sum(axis=1).apply(lambda x: "{:.3f}".format(x))
    for col in columns_results:
        df[col] = df[col].apply(lambda x: "{:.3f}".format(x))
    df["n_rows"] = df["n_rows"].astype(int)

    print(df)
    df.to_csv(f"bench_cse_tfidf{"_gridsearch" if gridsearch else ""}.csv", index=False)


def run_numeric_pipeline_benchmark():
    list_of_stats = []
    for n_rows in [1,5,10,20,30]:
        n_rows *= 10000
        df = make_mixed_df(
            n_rows=n_rows,
            n_num=20,
            n_cat=0,
            n_text=0,
            n_classes=0,
        )
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".csv", delete=False) as f:
            df.to_csv(f)
            f.flush()
            temp_path = f.name

        stats = redundant_numeric_pipeline(temp_path)
        stats = np.hstack((np.array([n_rows,n_rows]).reshape((2,1)), stats))
        list_of_stats.append(stats)
    stats = np.vstack(list_of_stats)
    df = pd.DataFrame(stats, columns=["n_rows","optimize", "make_learner","split","train","predict"])
    df["total"] = df[["optimize", "make_learner","split","train","predict"]].sum(axis=1)
    for col in ["optimize", "make_learner","split","train","predict","total"]:
        df[col] = df[col].apply(lambda x: "{:.3f}".format(x))
    print(df)
    df.to_csv("bench_cse.csv", index=False)


def main():
    run_tfidf_pipeline_benchmark(gridsearch=True)

if __name__ == '__main__':
    main()
