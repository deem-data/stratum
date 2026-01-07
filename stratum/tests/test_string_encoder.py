import sys
import pandas as pd
import pytest
import numpy as np

import stratum as skrub
from stratum import StringEncoder
from stratum.adapters.string_encoder import RustyStringEncoder, _rust_supported_subset, _clean_strings
skrub.set_config(rust_backend=True, debug_timing=True, num_threads=8)

def capture_std_out(capfd):
    # Capture timing output
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + (captured.err or "")
    return combined_output

@pytest.mark.parametrize("analyzer", ["char", "char_wb"])
def test_string_encoder_result(analyzer, capfd):
    s = pd.Series(["foo", "bar", None, "lorem ipsum dolor"]) # nulls handled upstream

    # StringEncoder should point to our subclass
    assert StringEncoder is RustyStringEncoder

    enc = StringEncoder(vectorizer='hashing', analyzer=analyzer, ngram_range=(3,5), n_components=2)
    Z = enc.fit_transform(s)
    assert Z.shape[0] == len(s)

    # Assert if rust timing appeared (verifies that rust code is executed)
    assert "[rust]" in capture_std_out(capfd)

def test_clean_strings_with_nan():
    """Test _clean_strings with NaN values."""
    x_list = ["foo", None, np.nan, float('nan'), "bar", 123]
    result = _clean_strings(x_list)
    assert result == ["foo", "", "", "", "bar", "123"]

def test_clean_strings_fail():
    """Test _clean_strings with NaN values."""
    x_list = ["foo", None, np.nan, float('nan'), "bar", 123]
    result = _clean_strings(x_list)
    assert result == ["foo", "", "", "", "bar", "123"]