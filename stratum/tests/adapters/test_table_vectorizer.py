from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np
import pandas as pd
import pytest
import skrub
from sklearn.preprocessing import OneHotEncoder
from skrub import StringEncoder, TableVectorizer
from skrub._utils import random_string

from stratum.adapters.table_vectorizer import (
    ExactFusedTableVectorizer,
    _FusedTableVectorizer,
)


def _string_encoder(n_components=2):
    return StringEncoder(
        n_components=n_components,
        analyzer="char",
        ngram_range=(1, 2),
        random_state=0,
    )


def _one_hot_encoder():
    return OneHotEncoder(
        drop="if_binary",
        dtype="float32",
        handle_unknown="ignore",
        sparse_output=False,
    )


def _fit_pair(X, **params):
    reference = TableVectorizer(**params)
    fused = ExactFusedTableVectorizer(**params)

    with warnings.catch_warnings(record=True) as reference_warnings:
        warnings.simplefilter("always")
        reference_output = reference.fit_transform(X)
    with warnings.catch_warnings(record=True) as fused_warnings:
        warnings.simplefilter("always")
        fused_output = fused.fit_transform(X)

    warning_signature = lambda items: [
        (item.category, str(item.message)) for item in items
    ]
    assert warning_signature(fused_warnings) == warning_signature(reference_warnings)
    return reference, reference_output, fused, fused_output


def _assert_estimator_metadata(reference, fused):
    assert type(reference) is type(fused)
    assert reference.get_params(deep=False) == fused.get_params(deep=False)

    for name in (
        "categories_",
        "drop_idx_",
        "all_outputs_",
        "extracted_features_",
        "format_",
        "output_dtype_",
        "output_time_zone_",
        "input_name_",
        "n_components_",
    ):
        reference_has = hasattr(reference, name)
        assert reference_has == hasattr(fused, name)
        if not reference_has:
            continue
        reference_value = getattr(reference, name)
        fused_value = getattr(fused, name)
        if isinstance(reference_value, list):
            assert len(reference_value) == len(fused_value)
            for reference_item, fused_item in zip(reference_value, fused_value):
                if isinstance(reference_item, np.ndarray):
                    assert reference_item.shape == fused_item.shape
                    for reference_value, fused_value in zip(
                        reference_item.flat, fused_item.flat
                    ):
                        if pd.isna(reference_value) and pd.isna(fused_value):
                            continue
                        assert reference_value == fused_value
                else:
                    assert reference_item == fused_item
        elif isinstance(reference_value, np.ndarray):
            np.testing.assert_array_equal(reference_value, fused_value)
        else:
            assert reference_value == fused_value


def _assert_fitted_metadata(reference, fused):
    for name in (
        "all_outputs_",
        "feature_names_in_",
        "n_features_in_",
        "kind_to_columns_",
        "column_to_kind_",
        "input_to_outputs_",
        "output_to_input_",
    ):
        assert getattr(reference, name) == getattr(fused, name)

    assert list(reference.transformers_) == list(fused.transformers_)
    for input_name, reference_transformer in reference.transformers_.items():
        _assert_estimator_metadata(
            reference_transformer, fused.transformers_[input_name]
        )

    assert list(reference.all_processing_steps_) == list(
        fused.all_processing_steps_
    )
    for input_name in reference.all_processing_steps_:
        reference_steps = reference.all_processing_steps_[input_name]
        fused_steps = fused.all_processing_steps_[input_name]
        assert len(reference_steps) == len(fused_steps)
        for reference_step, fused_step in zip(reference_steps, fused_steps):
            assert type(reference_step) is type(fused_step)
            if isinstance(reference_step, Mapping):
                assert list(reference_step) == list(fused_step)
                for output_name in reference_step:
                    _assert_estimator_metadata(
                        reference_step[output_name], fused_step[output_name]
                    )
            else:
                _assert_estimator_metadata(reference_step, fused_step)


def _assert_pair(reference, reference_output, fused, fused_output):
    assert isinstance(fused, _FusedTableVectorizer)
    assert type(reference_output) is type(fused_output)
    pd.testing.assert_frame_equal(
        reference_output,
        fused_output,
        check_dtype=True,
        check_exact=True,
    )
    _assert_fitted_metadata(reference, fused)


@pytest.mark.parametrize(
    "X",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "float": pd.Series(
                        [1.5, np.nan, 3.25], index=[11, 17, 29], dtype="float64"
                    ),
                    "integer": pd.Series(
                        [1, 2, None], index=[11, 17, 29], dtype="Int64"
                    ),
                    "boolean": pd.Series(
                        [True, None, False], index=[11, 17, 29], dtype="boolean"
                    ),
                    "nullable_float": pd.Series(
                        [1.0, None, 2.0], index=[11, 17, 29], dtype="Float64"
                    ),
                }
            ),
            id="numeric-dtypes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "numeric_strings": ["1.5", "2", "3.25"],
                    "mixed_strings": ["1", "not-a-number", "3"],
                },
                index=[5, 7, 13],
            ),
            id="numeric-strings",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "datetime_dtype": pd.to_datetime(
                        ["2024-01-01", "2024-01-02", "2024-01-03"]
                    ),
                    "datetime_strings": [
                        "2024-02-01",
                        "2024-02-02",
                        "2024-02-03",
                    ],
                    "words": ["alpha", "beta", "gamma"],
                },
                index=[100, 200, 300],
            ),
            id="datetime-dtypes-and-strings",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "strings": ["red", "blue", "red", "green"],
                    "categories": pd.Series(
                        pd.Categorical(
                            ["small", "large", "small", "medium"],
                            categories=["small", "medium", "large", "unused"],
                        ),
                        index=[9, 4, 2, 1],
                    ),
                },
                index=[9, 4, 2, 1],
            ),
            id="strings-and-categories",
        ),
    ],
)
def test_exact_fused_matches_reference_for_default_column_roles(X):
    reference, reference_output, fused, fused_output = _fit_pair(
        X,
        cardinality_threshold=4,
        high_cardinality=_string_encoder(),
    )
    _assert_pair(reference, reference_output, fused, fused_output)

    reference_test = reference.transform(X)
    fused_test = fused.transform(X)
    pd.testing.assert_frame_equal(reference_test, fused_test, check_exact=True)


def test_cardinality_boundary_is_strict_and_routes_once():
    X = pd.DataFrame(
        {
            "below": ["a", "b", "a", "b", "a", "b"],
            "equal": ["a", "b", "c", "d", "a", "b"],
            "above": ["a", "b", "c", "d", "e", "f"],
        },
        index=[20, 10, 40, 30, 60, 50],
    )
    reference, reference_output, fused, fused_output = _fit_pair(
        X,
        cardinality_threshold=4,
        high_cardinality=_string_encoder(),
    )
    _assert_pair(reference, reference_output, fused, fused_output)
    assert fused.column_to_kind_ == {
        "below": "low_cardinality",
        "equal": "high_cardinality",
        "above": "high_cardinality",
    }


def test_uninformative_columns_match_and_keep_reference_metadata():
    X = pd.DataFrame(
        {
            "all_null": [None, None, None, None],
            "constant": ["same", "same", "same", "same"],
            "unique": ["u0", "u1", "u2", "u3"],
            "number": [1, 2, 3, 4],
        },
        index=[7, 3, 9, 1],
    )
    reference, reference_output, fused, fused_output = _fit_pair(
        X,
        drop_if_constant=True,
        drop_if_unique=True,
    )
    _assert_pair(reference, reference_output, fused, fused_output)
    assert fused.input_to_outputs_ == {
        "all_null": ["all_null"],
        "constant": ["constant"],
        "unique": ["unique"],
        "number": ["number"],
    }


def test_empty_and_configured_null_strings_match():
    X = pd.DataFrame(
        {
            "default_nulls": ["", "NA", "  ", "kept"],
            "configured_null": ["missing", "kept", "missing", "kept"],
            "category": ["x", "y", "x", "y"],
        },
        index=[101, 103, 107, 109],
    )
    reference, reference_output, fused, fused_output = _fit_pair(
        X,
        cardinality_threshold=4,
        null_strings=["missing"],
        high_cardinality=_string_encoder(),
    )
    _assert_pair(reference, reference_output, fused, fused_output)


@pytest.mark.parametrize("low_cardinality", ["drop", "passthrough", _one_hot_encoder()])
@pytest.mark.parametrize("high_cardinality", ["drop", "passthrough", _string_encoder()])
def test_configured_low_and_high_roles_match(low_cardinality, high_cardinality):
    X = pd.DataFrame(
        {
            "low": ["a", "b", "a", "b", "a", "b"],
            "high": ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"],
            "number": [1, 2, 3, 4, 5, 6],
        },
        index=[13, 11, 7, 5, 3, 1],
    )
    reference, reference_output, fused, fused_output = _fit_pair(
        X,
        cardinality_threshold=3,
        low_cardinality=low_cardinality,
        high_cardinality=high_cardinality,
    )
    _assert_pair(reference, reference_output, fused, fused_output)


def test_output_name_collisions_follow_skrub_naming(monkeypatch):
    monkeypatch.setattr(skrub._utils, "random_string", lambda: "fixed")
    X = pd.DataFrame(
        {
            "city": ["a", "b", "a", "b"],
            "city_b": [1, 2, 3, 4],
        },
        index=[30, 10, 40, 20],
    )
    reference, reference_output, fused, fused_output = _fit_pair(
        X,
        cardinality_threshold=3,
        high_cardinality="drop",
    )
    _assert_pair(reference, reference_output, fused, fused_output)
    assert any("__skrub_fixed__" in name for name in fused.all_outputs_)


def test_transform_handles_unseen_categories_and_changed_null_patterns():
    train = pd.DataFrame(
        {
            "low": ["a", "b", "a", "b"],
            "high": ["alpha", "bravo", "charlie", "delta"],
            "number": [1, 2, 3, 4],
        },
        index=[100, 200, 300, 400],
    )
    test = pd.DataFrame(
        {
            "low": ["b", "unseen", None],
            "high": ["echo", None, "alpha"],
            "number": [5, 6, 7],
        },
        index=[901, 902, 903],
    )
    reference, _, fused, _ = _fit_pair(
        train,
        cardinality_threshold=3,
        high_cardinality=_string_encoder(),
    )
    reference_output = reference.transform(test)
    fused_output = fused.transform(test)
    pd.testing.assert_frame_equal(reference_output, fused_output, check_exact=True)
    pd.testing.assert_frame_equal(
        fused_output,
        fused.transform(test),
        check_exact=True,
    )


def test_zero_row_and_zero_column_inputs_match():
    cases = [
        pd.DataFrame(
            {
                "empty_numeric": pd.Series(dtype="float64"),
                "empty_string": pd.Series(dtype="object"),
            },
            index=pd.Index([], name="rows"),
        ),
        pd.DataFrame(index=pd.Index([], name="rows")),
        pd.DataFrame(index=pd.Index([10, 20], name="rows")),
    ]
    for X in cases:
        reference, reference_output, fused, fused_output = _fit_pair(
            X,
            high_cardinality=_string_encoder(),
        )
        _assert_pair(reference, reference_output, fused, fused_output)


def test_input_schema_errors_match():
    train = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    reference, _, fused, _ = _fit_pair(train, high_cardinality="drop")
    bad = train[["b", "a"]]

    with pytest.raises(Exception) as reference_error:
        reference.transform(bad)
    with pytest.raises(Exception) as fused_error:
        fused.transform(bad)

    assert type(fused_error.value) is type(reference_error.value)
    assert str(fused_error.value) == str(reference_error.value)


def test_unsupported_sparse_leaf_error_matches():
    X = pd.DataFrame({"low": ["a", "b", "a"]})
    sparse_ohe = OneHotEncoder(handle_unknown="ignore")
    reference = TableVectorizer(low_cardinality=sparse_ohe, high_cardinality="drop")
    fused = ExactFusedTableVectorizer(
        low_cardinality=sparse_ohe,
        high_cardinality="drop",
    )

    with pytest.raises(Exception) as reference_error:
        reference.fit_transform(X)
    with pytest.raises(Exception) as fused_error:
        fused.fit_transform(X)

    assert type(fused_error.value) is type(reference_error.value)
    assert str(fused_error.value) == str(reference_error.value)


def test_unsupported_specific_transformers_are_rejected_before_execution():
    estimator = ExactFusedTableVectorizer(
        specific_transformers=[("passthrough", ["a"])]
    )
    assert not estimator.supports(estimator)
    with pytest.raises(ValueError, match="specific_transformers"):
        estimator.fit_transform(pd.DataFrame({"a": [1, 2]}))
