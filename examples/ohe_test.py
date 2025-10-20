import numpy as np, pandas as pd
import sklearn.preprocessing
from stratum import set_config, OneHotEncoder
set_config(rust_backend=True, debug_timing=True)

X = pd.DataFrame({
    "a": ["x","y",None,"x","z",None],
    "b": ["c","c","d","e","d","c"],
})

enc = OneHotEncoder()
Z = enc.fit_transform(X)
print(Z)

sk = sklearn.preprocessing.OneHotEncoder(drop="if_binary", dtype=np.float32, handle_unknown="ignore", sparse_output=False)
Z_ref = sk.fit_transform(X)
print(Z_ref)
