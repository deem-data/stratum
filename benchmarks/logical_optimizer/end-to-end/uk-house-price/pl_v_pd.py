import polars as pl
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pl.read_parquet("input/tv_vec_1M.parquet")


price = df["Price"]
df = df.drop("Price")
df2 = df.to_pandas()
print(df.dtypes)
print(df2.dtypes)
sc = StandardScaler()
sc2 = StandardScaler()

sc.set_output(transform="polars")
x = sc.fit_transform(df)
x2 = sc2.fit_transform(df2)

# compare the two
print(x.shape)
print(x2.shape)
print(x2.dtype)
#exponent = -1
# while True:
#     if not np.allclose(x, x2, atol=10**exponent):
#         break
#     print(exponent)
#     exponent -= 1
error = x - x2
error_sum = np.sum(np.abs(error))
error_max = np.max(np.abs(error))
error_min = np.min(np.abs(error))
error_std = np.std(error)
error_mean = np.mean(error)

print("Error sum:", error_sum)
print("Error max:", error_max)
print("Error min:", error_min)
print("Error std:", error_std)
print("Error mean:", error_mean)