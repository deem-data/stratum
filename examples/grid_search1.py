import time

import skrub
from sklearn.model_selection import ShuffleSplit
from skrub import TableReport, TableVectorizer
from skrub.datasets import fetch_credit_fraud
from sklearn.ensemble import ExtraTreesClassifier

bunch = fetch_credit_fraud()
products_df, baskets_df = bunch.products, bunch.baskets
print(products_df.info())
print(baskets_df.info())

# Define the inputs of our skrub pipeline
products = skrub.var("products", products_df).skb.subsample(n=1000)
baskets = skrub.var("baskets", baskets_df).skb.subsample(n=1000)

# Specify our "X" and "y" variables for machine learning
basket_IDs = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

vectorizer = TableVectorizer()
products_vec = products.skb.apply(vectorizer)
baskets_vec = baskets.skb.apply(vectorizer)


#products = skrub.var("products", products_df).skb.subsample(n=1000)
#baskets = skrub.var("baskets", baskets_df).skb.subsample(n=1000)


# A pandas-based data-preparation pipeline that merges the tables
aggregated_products = products_vec.groupby("basket_ID").agg(
    skrub.choose_from(("mean", "max", "count"))).reset_index()

features = basket_IDs.merge(aggregated_products, left_on="ID", right_on="basket_ID", how="left")


# Train and tune hyperparameter
predictions = features.skb.apply(ExtraTreesClassifier(), y=fraud_flags)
single_split_cv = ShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
t0 = time.perf_counter()
search = predictions.skb.make_grid_search(fitted=True, scoring="roc_auc", keep_subsampling=True, cv=single_split_cv)
t1 = time.perf_counter()
print(search.results_)
exec_time = t1 - t0
print(f"Execution time = {exec_time:8.3f}s\n")
