"""A comparison with a missing value is False (``!=``: True) on polars, as in pandas."""
import numpy as np
import pandas as pd
import pytest
import skrub

import stratum as st
from stratum._api import evaluate

FRAME = pd.DataFrame({"x": [1.5, np.nan, -3.0], "s": pd.Series(["a", None, "c"], dtype="str")})


@pytest.mark.parametrize("path", ["assign", "selection"])
def test_comparisons_with_missing_values_match_pandas(path):
    with skrub.config_context(eager_data_ops=False):
        source = skrub.var("frame", FRAME)
        checks = dict(gt=source["x"] > 0, le=source["x"] <= 0, eq=source["x"] == 1.5,
                      ne=source["x"] != 0, s_eq=source["s"] == "a", s_ne=source["s"] != "a")
        plan = (source.assign(**checks) if path == "assign"
                else source[~(source["x"] > 0)])
    expected = (FRAME.assign(gt=FRAME["x"] > 0, le=FRAME["x"] <= 0, eq=FRAME["x"] == 1.5,
                             ne=FRAME["x"] != 0, s_eq=FRAME["s"] == "a",
                             s_ne=FRAME["s"] != "a")
                if path == "assign" else FRAME[~(FRAME["x"] > 0)])
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan).to_pandas()
    pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                  expected.reset_index(drop=True), check_dtype=False)
