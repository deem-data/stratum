import sys
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
import sklearn.preprocessing
from stratum import set_config, OneHotEncoder

def capture_std_out(capfd):
    # Capture timing output
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + (captured.err or "")
    return combined_output

@pytest.mark.parametrize("sparse_output", [True, False])
def test_ohe_compare(sparse_output, capfd):
    X = pd.DataFrame({
        #"a": ["x","y",None,"x","z",None], #failing case due to Pandas categorical/codes vs sklearn _encode
        "a": ["x","y","x","x","z","y"],
        "b": ["c","c","d","e","d","c"],
    })
    set_config(rust_backend=True, debug_timing=True)
    enc = OneHotEncoder(sparse_output=sparse_output) #stratum
    Z = enc.fit_transform(X)
    #print(Z)

    sk = sklearn.preprocessing.OneHotEncoder(drop="if_binary", dtype=np.float32, handle_unknown="ignore", sparse_output=sparse_output)
    Z_ref = sk.fit_transform(X)
    #print(Z_ref)

    if sparse_output:
        assert_array_equal(Z.data, Z_ref.data)
        assert_array_equal(Z.indices, Z_ref.indices)
        assert_array_equal(Z.indptr, Z_ref.indptr)
    else: #dense
        assert_array_equal(Z, Z_ref)
        # Assert if rust timing appeared (verifies that rust code is executed)
        assert "[rust]" in capture_std_out(capfd)

