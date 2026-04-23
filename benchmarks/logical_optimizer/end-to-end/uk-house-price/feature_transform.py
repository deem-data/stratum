import pandas as pd
import stratum as skrub
from sklearn.preprocessing import OneHotEncoder
from skrub import TableVectorizer, StringEncoder

file_path = "input/price_paid_records_small.csv"
df = pd.read_csv(file_path)
df = df.rename(columns={"Town/City": "Town"}, inplace=False)
df.drop("Price", axis=1, inplace=True)
print(df.info())

skrub.set_config(rust_backend=True, debug_timing=True)
enc = TableVectorizer(high_cardinality=StringEncoder(), low_cardinality=OneHotEncoder(), n_jobs=-1) #default setup
X_cat_enc = enc.fit_transform(df)
print(X_cat_enc)

