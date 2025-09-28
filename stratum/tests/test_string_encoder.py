import sys
import pandas as pd
import pytest

import stratum as skrub
from stratum import StringEncoder
from stratum.adapters.string_encoder import RustyStringEncoder

skrub.set_config(rust_backend=True, debug_timing=True, num_threads=8)

@pytest.mark.parametrize("analyzer", ["char", "char_wb"])
def test_string_encoder_result(analyzer, capfd):
    s = pd.Series(["foo", "bar", None, "lorem ipsum dolor"]) # nulls handled upstream

    # StringEncoder should point to our subclass
    assert StringEncoder is RustyStringEncoder

    enc = StringEncoder(vectorizer='hashing', analyzer=analyzer, ngram_range=(3,5), n_components=2)
    Z = enc.fit_transform(s)
    assert Z.shape[0] == len(s)

    # Capture timing output
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + (captured.err or "")
    #print(combined_output)

    # Assert if rust timing appeared (verifies that rust code is executed)
    assert "[rust]" in combined_output

