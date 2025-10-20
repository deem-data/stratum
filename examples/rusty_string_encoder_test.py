import os
import pandas as pd
import stratum as skrub
from stratum import StringEncoder
skrub.set_config(rust_backend=True)

#os.environ['SKRUB_RUST'] = '1'
#os.environ['SKRUB_RUST_DEBUG_TIMING'] = '1'
skrub.set_config(debug_timing=True)
skrub.set_config(num_threads=8)

s = pd.Series(["foo", "bar", None, "lorem ipsum dolor"]) # nulls handled upstream
enc = StringEncoder(vectorizer='hashing', analyzer='char', ngram_range=(3,5), n_components=2)
Z = enc.fit_transform(s)
print(Z.columns)
print(type(Z), Z.shape)
print(Z)

skrub.set_config(rust_backend=False)
enc = StringEncoder(vectorizer='hashing', analyzer='char', ngram_range=(3,5), n_components=2)
Z = enc.fit_transform(s)
print(Z.columns)
print(type(Z), Z.shape)
print(Z)

skrub.set_config(rust_backend=True, debug_timing=False)
# skrub.set_rust_config(allow_patch=False) #kill-switch
enc = StringEncoder(vectorizer='hashing', analyzer='char', ngram_range=(3,5), n_components=2)
Z = enc.fit_transform(s)
print(type(Z), Z.shape)
