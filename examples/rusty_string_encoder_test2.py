import pandas as pd
import stratum as skrub
from stratum import StringEncoder
skrub.set_config(rust_backend=True)
skrub.set_config(debug_timing=True)
skrub.set_config(num_threads=8)

s = pd.Series(["foo", "bar", None, "lorem ipsum dolor"]) # nulls handled upstream
print(StringEncoder)
enc = StringEncoder(vectorizer='hashing', analyzer='char', ngram_range=(3,5), n_components=2)
Z = enc.fit_transform(s)
print(Z)
print(type(enc).__name__)
print(type(Z), Z.shape)
assert Z.shape[0] == len(s)
