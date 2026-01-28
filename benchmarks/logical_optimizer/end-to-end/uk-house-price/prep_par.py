import skrub
import polars as pl
from sklearn.preprocessing import StandardScaler
import pandas as pd
from time import perf_counter
size = input("size: ")

t0 = perf_counter()

df = pd.read_csv(f"input/price_paid_records_{size}.csv")
price = df["Price"]
df = df.drop("Price", axis=1)

tv = skrub.TableVectorizer()
sc = StandardScaler()
sc.set_output(transform="pandas")
df_vec = tv.fit_transform(df)
df_vec_norm = sc.fit_transform(df_vec)

df_vec_norm["Price"] = price

t1 = perf_counter()

print(f"Time taken: {t1 - t0} seconds")

df_pl = pl.from_pandas(df_vec_norm)
df_pl.write_parquet(f"input/tmp_{size}.parquet")