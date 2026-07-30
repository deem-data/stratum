import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from stratum import _rust_backend as rb
from stratum.adapters.string_encoder import RustyStringEncoder


pytestmark = pytest.mark.skipif(
    not rb.HAVE_RUST, reason="Rust backend not built"
)


def _fit_tfidf():
    strings = [
        "alpha beta",
        "beta gamma",
        "delta epsilon",
        "alpha delta",
    ]
    return strings, rb.tfidf_fit(strings, "char", 2, 3)


def test_tfidf_fit_returns_owned_handle_and_transforms():
    strings, fit_result = _fit_tfidf()
    model, data, indices, indptr, n_rows, n_cols = fit_result

    assert not isinstance(model, int)
    assert type(model).__name__ == "_TfidfModelHandle"
    assert n_rows == len(strings)
    assert len(indptr) == n_rows + 1

    # A second Python reference keeps the native model alive after the
    # original fit-result reference is dropped.
    model_alias = model
    del model, fit_result
    gc.collect()
    transformed = rb.tfidf_transform(model_alias, ["alpha beta", "unknown"])
    out_data, out_indices, out_indptr, out_rows, out_cols = transformed
    assert out_rows == 2
    assert out_cols == n_cols
    assert len(out_indptr) == out_rows + 1
    assert len(out_data) == len(out_indices)

    with pytest.raises(TypeError):
        rb.tfidf_transform(0, strings)


@pytest.mark.parametrize(
    ("fit", "transform", "expected_handle"),
    [
        (rb.fd_fit, rb.fd_transform, "_FdEmbedModelHandle"),
        (
            rb.truncated_svd_fit,
            rb.truncated_svd_transform,
            "_TruncatedSvdModelHandle",
        ),
    ],
)
def test_projection_fit_returns_owned_handle_and_transforms(
    fit, transform, expected_handle
):
    _, (_, data, indices, indptr, n_rows, n_cols) = _fit_tfidf()
    data = np.ascontiguousarray(data, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)

    if fit is rb.fd_fit:
        model, fitted = fit(
            data, indices, indptr, n_rows, n_cols, 2, 2, 0
        )
    else:
        model, fitted = fit(
            data, indices, indptr, n_rows, n_cols, 2, 0
        )

    assert not isinstance(model, int)
    assert type(model).__name__ == expected_handle
    model_alias = model
    del model
    gc.collect()
    fitted = np.asarray(fitted)
    transformed = np.asarray(
        transform(model_alias, data, indices, indptr, n_rows, n_cols)
    )
    assert transformed.shape == fitted.shape
    assert np.isfinite(transformed).all()


def test_fitted_string_encoder_handle_supports_concurrent_transform():
    train = pd.Series(
        [
            "alpha beta",
            "beta gamma",
            "delta epsilon",
            "alpha delta",
            "gamma epsilon",
        ],
        name="text",
    )
    test = pd.Series(["alpha", "gamma", "epsilon"], name="text")
    encoder = RustyStringEncoder(
        vectorizer="tfidf",
        analyzer="char",
        ngram_range=(2, 3),
        n_components=2,
        random_state=0,
    )
    encoder._stratum_force_rust = True
    encoder.fit_transform(train)

    state = encoder._rust_state_
    assert type(state["tfidf_model_id"]).__name__ == "_TfidfModelHandle"
    assert type(state["svd_model_id"]).__name__ == "_FdEmbedModelHandle"

    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(lambda _: encoder.transform(test), range(8)))

    expected = outputs[0]
    for output in outputs[1:]:
        np.testing.assert_allclose(
            output.to_numpy(), expected.to_numpy(), rtol=1e-6, atol=1e-6
        )
