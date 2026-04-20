import unittest
import os
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from stratum.adapters.standard_scaler import NumpyStandardScaler, RustyStandardScaler
from stratum._config import config
import stratum._rust_backend as rb
import logging
logging.basicConfig(level=logging.DEBUG)

class TestAdaperStandardScaler(unittest.TestCase):
    def test_rusty_scaler(self):
        rng = np.random.default_rng(42)
        shapes = [(1000, 10), (10000, 100), (100000, 100), (1000000, 10), (10000000, 1)]
        for shape in shapes:
            x = rng.standard_normal(size=shape, dtype=np.float64) * 10 + 100
            sk = SKStandardScaler()
            sk_out = sk.fit_transform(x)
            with config(rust_backend=True, debug_timing=True):
                rusty = RustyStandardScaler()
                rusty_out = rusty.fit_transform(x)
            print(rusty_out)
            np.testing.assert_allclose(rusty_out, sk_out, rtol=1e-6, atol=1e-6)

    def test_rusty_scaler_sample_weight(self):
        rng = np.random.default_rng(42)
        shapes = [(1000, 10), (10000, 100), (100000, 100), (1000000, 10), (10000000, 1)]
        for shape in shapes:
            x = rng.standard_normal(size=shape, dtype=np.float64) * 10 + 100
            weight = rng.random(shape[0])
            sk = SKStandardScaler()
            sk_out = sk.fit(x, sample_weight=weight).transform(x)
            with config(rust_backend=True, debug_timing=True):
                rusty = RustyStandardScaler()
                rusty_out = rusty.fit(x, sample_weight=weight).transform(x)
            np.testing.assert_allclose(rusty_out, sk_out, rtol=1e-6, atol=1e-6)

    def test_numpy_scaler_no_reuse_mean(self):
        rng = np.random.default_rng(42)
        shapes = [(1000, 10), (10000, 100), (100000, 100), (1000000, 10), (10000000, 1)]
        for shape in shapes:
            x = rng.standard_normal(size=shape, dtype=np.float64) * 10 + 100
            sk = SKStandardScaler()
            sk_out = sk.fit_transform(x)
            numpy = NumpyStandardScaler()
            numpy_out = numpy.fit_transform(x)
            np.testing.assert_allclose(numpy_out, sk_out, rtol=1e-6, atol=1e-6)

    def test_numpy_scaler_reuse_mean(self):
        rng = np.random.default_rng(42)
        shapes = [(1000, 10), (10000, 100), (100000, 100), (1000000, 10), (10000000, 1)]
        for shape in shapes:
            x = rng.standard_normal(size=shape, dtype=np.float64) * 10 + 100
            sk = SKStandardScaler()
            sk_out = sk.fit_transform(x)
            numpy = NumpyStandardScaler(reuse_mean=True)
            numpy_out = numpy.fit_transform(x)
            np.testing.assert_allclose(numpy_out, sk_out, rtol=1e-6, atol=1e-6)

    def test_numpy_scaler_sample_weight(self):
        rng = np.random.default_rng(42)
        shapes = [(1000, 10), (10000, 100), (100000, 100), (1000000, 10), (10000000, 1)]
        for shape in shapes:
            x = rng.standard_normal(size=shape, dtype=np.float64) * 10 + 100
            weight = rng.random(shape[0])
            sk = SKStandardScaler()
            sk_out = sk.fit(x, sample_weight=weight).transform(x)
            numpy = NumpyStandardScaler(reuse_mean=True)
            numpy_out = numpy.fit(x, sample_weight=weight).transform(x)
            np.testing.assert_allclose(numpy_out, sk_out, rtol=1e-6, atol=1e-6)

    def test_rusty_scaler_fallback1(self):
        x = np.random.random((100, 10))
        with config(rust_backend=False, debug_timing=True):
            rusty = RustyStandardScaler()
            rusty.fit_transform(x)

    def test_rusty_scaler_fallback2(self):
        x = np.random.random((100, 10))
        with config(rust_backend=True, debug_timing=True):
            rusty = RustyStandardScaler()
            def dummy_error(a, b, c):
                raise Exception("Dummy Rust error")
            original = rb.standard_scale_transform
            try:
                rb.standard_scale_transform = dummy_error
                rusty.fit_transform(x)
            finally:
                rb.standard_scale_transform = original

    def test_rusty_scaler_fallback3(self):
        x = np.random.random((100, 10))
        with config(rust_backend=True, debug_timing=True):
            rusty = RustyStandardScaler(copy=False)
            print(rusty._supported_params)
            rusty.fit_transform(x)

    def test_numpy_scaler(self):
        x = np.random.random((100, 10))
        sk = SKStandardScaler(copy=False)
        sk_out = sk.fit(x).transform(x)
        numpy = NumpyStandardScaler(copy=False)
        numpy_out = numpy.fit(x).transform(x)
        np.testing.assert_allclose(numpy_out, sk_out, rtol=1e-6, atol=1e-6)
    
    def test_core_config_rust(self):
        scaler = RustyStandardScaler(n_jobs=1)
        assert scaler.n_jobs == 1

        # set n_jobs to more than the number of cores
        scaler = RustyStandardScaler(n_jobs=100000)
        assert scaler.n_jobs == os.cpu_count()

@pytest.mark.parametrize("scaler_cls, expected_msg",
    [(RustyStandardScaler, "This RustyStandardScaler instance is not fitted yet. Call 'fit' "
                              "with appropriate arguments before using this estimator."),
    (NumpyStandardScaler, "This NumpyStandardScaler instance is not fitted yet. Call 'fit' "
                          "with appropriate arguments before using this estimator.")],)

def test_scalers_not_fit_pytest(scaler_cls, expected_msg):
    x = np.random.random((100, 10)).astype(np.float32)
    with config(rust_backend=True, debug_timing=True):
        scaler = scaler_cls()
        with pytest.raises(Exception) as excinfo:
            scaler.transform(x)
    assert str(excinfo.value) == expected_msg


if __name__ == "__main__":
    unittest.main()

