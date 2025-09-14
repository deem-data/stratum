import os
import pandas as pd

os.environ['SKRUB_RUST'] = '1'

from skrub import StringEncoder
s = pd.Series(["foo", "bar", None, "lorem ipsum dolor"]) # nulls handled upstream
enc = StringEncoder(vectorizer='hashing', analyzer='char', ngram_range=(3,5), n_components=2)
Z = enc.fit_transform(s)
print(type(Z), Z.shape)
assert Z.shape[0] == len(s)