from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC

from stratum.logical_optimizer import optimize
from stratum.search import grid_search

import stratum as skrub
import logging
import tempfile
import numpy as np
import pandas as pd
from time import time

logging.basicConfig(level=logging.INFO)

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

def tfidf_pipeline(df_path: str, show_graph: bool = False, stratum: bool = False):
    path = skrub.as_data_op(df_path)
    data = path.skb.apply_func(pd.read_csv).skb.subsample(n=100)
    data = data.fillna("")
    y = data["y"].skb.mark_as_y()
    X = data[["text"]].skb.mark_as_X()


    vectorizer = PandasTfidfVectorizer()

    pipes = {f"pipeline{i}": X.skb.apply(vectorizer).
             skb.apply(model, y=y) for i, model in
             enumerate(
                 [LinearRegression(),
                  Ridge(),
                  LogisticRegression(max_iter=1000),
                  LinearSVC(),
                  MultinomialNB(),
                  ])}
    pred = skrub.choose_from(pipes).as_data_op()

    if show_graph:
        pred.skb.draw_graph().open()
    print("----------------------------------------")
    stats = make_gridsearch(pred, stratum=stratum)
    print("\nOriginal Pipeline took: ",sum(stats))
    print("----------------------------------------")


    stats_opt = make_gridsearch(pred, optimize_enabled=True, stratum=stratum)
    print("\nTotal optimized Pipeline took: ",sum(stats_opt))
    print("----------------------------------------")
    if show_graph:
        pred.skb.draw_graph().open()
    return np.array([stats, stats_opt])


def make_gridsearch(pred, random_state=42, optimize_enabled=False, stratum=False) -> tuple[list, list]:
    if optimize_enabled:
        t00 = time()
        pred = optimize(pred)
        t0 = time()
        stats = [t0 - t00]
    else:
        t0 = time()
        stats = [0.0]

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    if stratum:
        search = grid_search(pred, cv=cv)
        print("Search results: \n", search)
    else:
        search = pred.skb.make_grid_search(fitted=True, cv=cv, n_jobs=1)
        print("Search results: \n", search.results_)

    t1 = time()
    stats.append(t1-t0)
    return stats


def run_tfidf_pipeline_benchmark(stratum: bool = False):
    data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    df = pd.DataFrame({"text": data.data, "y": data.target})
    df["text"].fillna("", inplace=True)

    list_of_stats = []
    parameters = [100, 500, 1000,]
    for n_rows in parameters:
        df_n_rows = df.head(n_rows)
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".csv", delete=False) as f:
            df_n_rows.to_csv(f)
            f.flush()
            temp_path = f.name

        stats = tfidf_pipeline(temp_path, show_graph = False, stratum=stratum)
        stats = np.hstack((np.array([n_rows, n_rows]).reshape((2, 1)), stats))
        list_of_stats.append(stats)

    stats = np.vstack(list_of_stats)

    columns_results = ["n_rows","optimize", "search"]
    df = pd.DataFrame(stats, columns=columns_results)

    columns_results.remove("n_rows")
    df["total"] = df[columns_results].sum(axis=1).apply(lambda x: "{:.3f}".format(x))
    for col in columns_results:
        df[col] = df[col].apply(lambda x: "{:.3f}".format(x))
    df["n_rows"] = df["n_rows"].astype(int)

    print(df)
    df.to_csv(f"bench_cse_tfidf_gridsearch.csv", index=False)

def main():
    run_tfidf_pipeline_benchmark(stratum=True)

if __name__ == '__main__':
    main()
