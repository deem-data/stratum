"""Runtime values: a polars value holds its pandas index beside it."""
import pandas as pd
import polars as pl
import pytest

from stratum.runtime._frames import (
    PolarsIndex, StratumFrame, StratumSeries, as_result, polars_index_of, row_numbers,
    take_index, unwrap, wrap)
from stratum.runtime._object_serialization import deserialize_object, serialize_object
from stratum.runtime._object_size import get_size

FRAME = pl.DataFrame({"v": [1.5, 2.5, 3.5]})
KEYS = pl.DataFrame({"k": [3, 1, 2], "j": ["a", None, "c"]})


def test_a_default_range_index_is_not_held():
    assert PolarsIndex.of(pd.RangeIndex(3)) is None
    for index in (pd.RangeIndex(1, 4), pd.RangeIndex(3, name="i"), pd.Index([2, 0, 1])):
        assert PolarsIndex.of(index).pandas is index


def test_index_columns_convert_as_frame_columns_would():
    expected = pd.MultiIndex.from_frame(KEYS.to_pandas().astype({"j": "str"}))
    pd.testing.assert_index_equal(PolarsIndex(columns=KEYS).to_pandas(), expected)
    single = PolarsIndex(columns=KEYS.select("k")).to_pandas()
    pd.testing.assert_index_equal(single, pd.Index([3, 1, 2], name="k"))


def test_taking_rows_takes_them_from_every_form_held():
    rows = pl.Series([2, 0, 0], dtype=pl.UInt32)
    index = PolarsIndex(pd.Index([30, 10, 20], name="x"), KEYS).take(rows)
    pd.testing.assert_index_equal(index.pandas, pd.Index([20, 30, 30], name="x"))
    assert index.columns.to_dicts() == [KEYS.row(2, named=True), *[KEYS.row(0, named=True)] * 2]
    # No index held reads as 0..n-1.
    pd.testing.assert_index_equal(take_index(None, rows).to_pandas(), pd.Index([2, 0, 0]))
    assert row_numbers(3).to_list() == [0, 1, 2]


def test_only_a_value_with_an_index_held_beside_it_is_wrapped():
    index = PolarsIndex(columns=KEYS)
    pandas_frame = FRAME.to_pandas()
    assert wrap(FRAME) is FRAME and wrap(pandas_frame) is pandas_frame
    wrapped = wrap(FRAME, index)
    assert isinstance(wrapped, StratumFrame) and wrap(wrapped, index) is wrapped
    assert isinstance(wrap(FRAME["v"], index), StratumSeries)
    assert unwrap(wrapped) is FRAME and polars_index_of(wrapped) is index
    assert unwrap(FRAME) is FRAME and polars_index_of(FRAME) is None
    with pytest.raises(TypeError):
        StratumFrame(FRAME.to_pandas(), index)


def test_the_index_reads_as_pandas_on_either_library():
    pd.testing.assert_index_equal(StratumFrame(FRAME).index, pd.RangeIndex(3))
    held = StratumFrame(FRAME, PolarsIndex(pd.Index([7, 8, 9])))
    pd.testing.assert_index_equal(held.index, pd.Index([7, 8, 9]))
    frame = pd.DataFrame({"v": [1]}, index=["r"])
    assert StratumFrame(frame).index is frame.index


def test_a_result_shows_index_columns_first_and_drops_another_index():
    keyed = as_result(wrap(FRAME, PolarsIndex(columns=KEYS)))
    assert keyed.columns == ["k", "j", "v"]
    assert as_result(wrap(FRAME, PolarsIndex(pd.Index([7, 8, 9])))) is FRAME


def test_the_held_index_counts_towards_the_size():
    index = PolarsIndex(pd.Index([7, 8, 9]), KEYS)
    assert get_size(wrap(FRAME, index)) > get_size(FRAME) + KEYS.estimated_size()


@pytest.mark.parametrize("index", [PolarsIndex(pd.Index(["x", "y", "z"], name="n")),
                                   PolarsIndex(columns=KEYS)])
def test_a_spilled_value_comes_back_with_its_index(tmp_path, index):
    spilled = serialize_object(wrap(FRAME, index), tmp_path / "value")
    value = deserialize_object(spilled)
    assert isinstance(value, StratumFrame) and value.data.equals(FRAME)
    pd.testing.assert_index_equal(value.index, index.to_pandas())
