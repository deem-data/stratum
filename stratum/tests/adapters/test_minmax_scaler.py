import sys

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from stratum import set_config, MinMaxScaler
from stratum import _rust_backend as rb
from stratum.adapters.minmax_scaler import RustyMinMaxScaler, MIN_BLOCK_LEN

set_config(rust_backend=True, debug_timing=True, num_threads=8)


def capture_std_out(capfd):
    # Capture timing output^
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + (captured.err or "")
    return combined_output


def _make_data(shape, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape, dtype=np.float32) * 10 + 100


@pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")
@pytest.mark.parametrize("shape", [(50, 4), (500, 1), (37, 8)])
def test_minmax_parity_default_params(shape, capfd):
    # MinMaxScaler should point to our subclass
    assert MinMaxScaler is RustyMinMaxScaler

    x = _make_data(shape)
    sk_out = SkMinMaxScaler().fit_transform(x)
    out = MinMaxScaler().fit_transform(x)

    np.testing.assert_allclose(out, sk_out, rtol=1e-6, atol=1e-6)
    # Assert if rust timing appeared (verifies that rust code is executed)
    assert "[rust]" in capture_std_out(capfd)


@pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")
def test_minmax_parity_clip_true(capfd):
    """clip=True must still use the Rust fastpath."""
    x_fit = _make_data((100, 3))

    sk = SkMinMaxScaler(clip=True).fit(x_fit)
    scaler = MinMaxScaler(clip=True).fit(x_fit)

    # values outside the fitted range on every column, to exercise the clamp
    x_new = np.vstack([sk.data_min_ - 10, sk.data_max_ + 10]).astype(np.float32)

    sk_out = sk.transform(x_new)
    out = scaler.transform(x_new)

    np.testing.assert_allclose(out, sk_out, rtol=1e-6, atol=1e-6)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert "[rust]" in capture_std_out(capfd)


@pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")
def test_minmax_non_default_feature_range_falls_back_to_sklearn(capfd):
    """The Rust kernel hardcodes output to [0, 1] and has no feature_range
    parameter; non-default feature_range must fall back to sklearn rather
    than silently producing values scaled to the wrong range."""
    x = _make_data((60, 4))

    sk_out = SkMinMaxScaler(feature_range=(-1, 1)).fit_transform(x)
    out = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)

    np.testing.assert_allclose(out, sk_out, rtol=1e-6, atol=1e-6)
    assert "[rust]" not in capture_std_out(capfd)


def test_minmax_constant_column():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((40, 3)).astype(np.float32)
    x[:, 1] = 5.0  # zero-variance column

    sk_out = SkMinMaxScaler().fit_transform(x)
    out = MinMaxScaler().fit_transform(x)

    np.testing.assert_allclose(out, sk_out, rtol=1e-6, atol=1e-6)
    assert np.all(out[:, 1] == 0.0)


@pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")
def test_minmax_fallback_when_rust_backend_disabled(capfd):
    set_config(rust_backend=False)
    try:
        x = _make_data((50, 4))
        sk_out = SkMinMaxScaler().fit_transform(x)
        out = MinMaxScaler().fit_transform(x)
        np.testing.assert_allclose(out, sk_out, rtol=1e-6, atol=1e-6)
        assert "[rust]" not in capture_std_out(capfd)
    finally:
        set_config(rust_backend=True)


@pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")
def test_minmax_fallback_when_allow_patch_false(capfd):
    set_config(allow_patch=False)
    try:
        x = _make_data((50, 4))
        sk_out = SkMinMaxScaler().fit_transform(x)
        out = MinMaxScaler().fit_transform(x)
        np.testing.assert_allclose(out, sk_out, rtol=1e-6, atol=1e-6)
        assert "[rust]" not in capture_std_out(capfd)
    finally:
        set_config(allow_patch=True)


@pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")
def test_minmax_transform_without_fit_falls_back(capfd):
    """Fit under the sklearn path, then transform under the Rust path."""
    x_fit = _make_data((50, 4))
    x_new = _make_data((10, 4), seed=1)

    scaler = MinMaxScaler()
    set_config(rust_backend=False)
    try:
        scaler.fit(x_fit)
        # should use sklearn here
        assert "[rust]" not in capture_std_out(capfd)
    finally:
        set_config(rust_backend=True)

    out = scaler.transform(x_new)
    assert "[rust]" in capture_std_out(capfd)
    # shouldve use rust for transform
    sk_out = SkMinMaxScaler().fit(x_fit).transform(x_new)
    np.testing.assert_allclose(out, sk_out, rtol=1e-6, atol=1e-6)


def test_n_chunks_unit():
    scaler = RustyMinMaxScaler(n_jobs=4)
    assert scaler._n_chunks(np.zeros(10)) == 1
    assert scaler._n_chunks(np.zeros(MIN_BLOCK_LEN * 2)) == 2
    # more blocks than n_jobs -> capped at n_jobs
    assert scaler._n_chunks(np.zeros(MIN_BLOCK_LEN * 10)) == 4

    scaler_single = RustyMinMaxScaler(n_jobs=1)
    assert scaler_single._n_chunks(np.zeros(MIN_BLOCK_LEN * 10)) == 1
