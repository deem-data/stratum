import sys
import pandas as pd
import pytest
import numpy as np

from stratum import StringEncoder
from stratum import set_config
from stratum.adapters.string_encoder import (
    RustyStringEncoder,
    _clean_strings,
    _rust_supported_subset,
    _prep_strings,
    _prep_strings_transform,
)

set_config(rust_backend=True, debug_timing=True, num_threads=8)

def capture_std_out(capfd):
    # Capture timing output
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + (captured.err or "")
    return combined_output

@pytest.mark.parametrize("analyzer", ["char", "char_wb"])
@pytest.mark.parametrize("vectorizer", ["hashing", "tfidf"])
def test_string_encoder_result(analyzer, vectorizer, capfd):
    s = pd.Series(["foo", "bar", None, "lorem ipsum dolor"]) # nulls handled upstream

    # StringEncoder should point to our subclass
    assert StringEncoder is RustyStringEncoder

    enc = StringEncoder(vectorizer=vectorizer, analyzer=analyzer, ngram_range=(3,5), n_components=2)
    Z = enc.fit_transform(s)
    assert Z.shape[0] == len(s)

    # Round-trip through transform to cover the transform code path
    s2 = pd.Series(["foo", "baz", "lorem"])
    Z2 = enc.transform(s2)
    assert Z2.shape[0] == len(s2)
    assert Z2.shape[1] == enc.n_components_

    # Assert if rust timing appeared (verifies that rust code is executed)
    assert "[rust]" in capture_std_out(capfd)


def test_string_encoder_padding():
    """n_components larger than achievable rank triggers the padding branch."""
    s = pd.Series(["a", "b"])
    enc = StringEncoder(vectorizer="hashing", analyzer="char",
                        ngram_range=(3, 5), n_components=50)
    Z = enc.fit_transform(s)
    assert Z.shape == (2, 50)
    Z2 = enc.transform(pd.Series(["a", "c"]))
    assert Z2.shape == (2, 50)


def test_string_encoder_fallback_when_rust_disabled():
    """Disabling rust_backend routes through sklearn fallback path."""
    set_config(rust_backend=False)
    try:
        s = pd.Series(["foo", "bar", "baz", "lorem ipsum"])
        enc = StringEncoder(vectorizer="hashing", analyzer="char",
                            ngram_range=(3, 5), n_components=2)
        Z = enc.fit_transform(s)
        assert Z.shape[0] == len(s)
        Z2 = enc.transform(pd.Series(["foo", "bar"]))
        assert Z2.shape[0] == 2
    finally:
        set_config(rust_backend=True)


def test_transform_without_fit_falls_back():
    """Calling transform before fit_transform should take the sklearn fallback."""
    enc = StringEncoder(vectorizer="hashing", analyzer="char",
                        ngram_range=(3, 5), n_components=2)
    # Force sklearn fit so internal sklearn state exists, but _rust_state_ stays None
    set_config(rust_backend=False)
    try:
        enc.fit(pd.Series(["foo", "bar", "baz"]))
    finally:
        set_config(rust_backend=True)
    # _rust_state_ is None → transform should fall back
    Z = enc.transform(pd.Series(["foo", "bar"]))
    assert Z.shape[0] == 2


class _Dummy:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_rust_supported_subset_rejections():
    ok, _ = _rust_supported_subset(_Dummy(vectorizer="hashing", stop_words=None,
                                         analyzer="char", ngram_range=(3, 5)))
    assert ok

    ok, msg = _rust_supported_subset(_Dummy(vectorizer="count"))
    assert not ok and "vectorizer" in msg

    ok, msg = _rust_supported_subset(_Dummy(vectorizer="tfidf", stop_words="english"))
    assert not ok and "stop_words" in msg

    ok, msg = _rust_supported_subset(_Dummy(vectorizer="tfidf", stop_words=None,
                                            analyzer="word"))
    assert not ok and "analyzer" in msg

    ok, msg = _rust_supported_subset(_Dummy(vectorizer="tfidf", stop_words=None,
                                            analyzer="char", ngram_range=(5, 3)))
    assert not ok and "ngram_range" in msg


def test_prep_strings_roundtrip():
    s = pd.Series(["foo", None, "bar"])
    out = _prep_strings(s)
    assert len(out) == 3
    assert all(isinstance(v, str) for v in out)

    out2 = _prep_strings_transform(s)
    assert len(out2) == 3
    assert all(isinstance(v, str) for v in out2)

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