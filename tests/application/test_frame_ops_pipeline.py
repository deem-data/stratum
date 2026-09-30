"""End-to-end showcase of Stratum's logical abstractions and the physical
compiling pass, exercised through one realistic supervised pipeline.

A single e-commerce pipeline -- read a CSV, engineer features, encode the
categoricals, fit a model -- chains everything the recent optimizer work
introduced into one default-selector pipeline:

* a source **read** ``pd.read_csv`` lowered to a source op;
* a **mask selection** ``df[predicate]`` that folds a boolean expression tree
  into one :class:`SelectionOp` (``stratum.optimizer.logical._selection_ops``);
* a folded **column map** ``df.assign(...)`` -- arithmetic, a ``.dt`` accessor
  over a parsed date -- collapsing into one :class:`AssignMapOp`
  (``stratum.optimizer.logical._map_ops``);
* a literal **column projection** ``df[[...]]`` -> :class:`ColumnProjectionOp`
  and two selector-driven ``skb.select(...)`` splits -> :class:`ColumnSelectorOp`
  (``stratum.optimizer.logical._projection_ops``);
* a **transformer** (:class:`TransformerOp`) with multiple registered physical
  implementations, where the default chooses the sklearn/skrub implementation,
  and an **estimator** (:class:`EstimatorOp`) that fits the model.

The tests assert three things:

1. each backend-agnostic *logical* abstraction is recognised;
2. the *compiling* pass binds the frame abstractions to concrete
   :class:`PhysicalOp` implementations, and binds the transformer according to
   the configured implementation selector;
3. the compiled plan trains and scores end-to-end via ``make_grid_search``.
"""
import pytest
import pandas as pd
import numpy as np
from skrub import StringEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._selection_ops import SelectionKind, SelectionOp
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._projection_ops import (
    ColumnProjectionOp, ColumnSelectorOp)
from stratum.optimizer.logical._dataframe_ops import ConcatOp
from stratum.optimizer.logical._ops import PredictorOp, TransformerOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from tests._helpers import csv_file
from stratum.optimizer._optimize import SearchConfig
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.optimizer.physical import FlagBasedSelector
from stratum.frontend._skrub_graph import get_data
from stratum.runtime._scheduler import SequentialScheduler


def make_orders(n=60):
    """A seeded, messy order book with a regression target (``satisfaction``).

    Cancelled and zero-quantity rows are noise the filter must drop; the string
    ``category``/``country`` columns feed the encoder."""
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "order_id": range(n),
        "order_date": pd.to_datetime(
            rng.choice(pd.date_range("2021-01-01", "2021-12-31"), size=n)
        ).astype(str),
        "quantity": rng.randint(0, 30, size=n),
        "unit_price": rng.uniform(1, 100, size=n).round(2),
        "category": rng.choice(["books", "toys", "home", "garden"], size=n),
        "country": rng.choice(["DE", "FR", "US"], size=n),
        "status": rng.choice(["completed", "cancelled"], size=n, p=[0.8, 0.2]),
        "satisfaction": rng.uniform(1, 5, size=n).round(1),
    })


def build_pipeline(file_path, model=None):
    """read -> filter -> map -> projection -> select -> encode -> concat -> fit."""
    model = model if model is not None else Ridge(random_state=0)

    # (0) READ: the pipeline starts from a CSV on disk, lowered to a source read.
    src = st.as_data_op(file_path).skb.apply_func(pd.read_csv)

    # (1) FILTER: keep completed orders that actually shipped units. The whole
    # boolean tree folds into a single SelectionOp(MASK) predicate.
    active = src[(src["status"] == "completed") & (src["quantity"] > 0)]
    y = active["satisfaction"].skb.mark_as_y()

    # (2) MAP: derive numeric features. Arithmetic and a `.dt` accessor over a
    # parsed date are natively lazy, so they fold into one AssignMapOp.
    order_date = active["order_date"].skb.apply_func(pd.to_datetime)
    featured = active.assign(
        revenue=active["quantity"] * active["unit_price"],
        net_revenue=active["quantity"] * active["unit_price"] * (1 - 0.19),
        order_month=order_date.dt.month,
        is_bulk=active["quantity"] >= 10,
    )

    # (3) PROJECTION: keep the modelling columns by literal name (drops the id,
    # raw date, status and target) -> ColumnProjectionOp.
    X = featured[["quantity", "unit_price", "revenue", "net_revenue",
                  "order_month", "category", "country"]].skb.mark_as_X()

    # (4) SELECTORS: split numeric features from the string columns, each via a
    # ColumnSelectorOp.
    X_num = X.skb.select(st.selectors.numeric())
    X_cat = X.skb.select(~st.selectors.numeric())

    # (5) TRANSFORMER: encode the string columns. The default selector picks the
    # sklearn/skrub physical implementation.
    X_cat_enc = X_cat.skb.apply(StringEncoder())

    # (6) CONCAT the numeric + encoded blocks, then (7) fit the ESTIMATOR.
    X_vec = X_num.skb.concat([X_cat_enc], axis=1)
    return X_vec.skb.apply(model, y=y)


def _optimize(dag, selector=None, pandas_query=False):
    return optimize_(dag, OptConfig(dataframe_ops=True, selector=selector,
                                    pandas_query=pandas_query))[0]


@pytest.fixture(params=["pandas", "polars"])
def backend(request):
    """The backend to plan against, as a selector pinned to it.

    Yielded as a selector rather than set as a flag: the backend is the
    selector's own state, so a test that does not thread it into ``optimize``
    is not testing that backend at all.
    """
    return FlagBasedSelector(backend=request.param)


def test_frame_ops_pipeline_plan(backend):
    """Every logical abstraction is recognised, and the compiling pass binds the
    frame abstractions to concrete physical implementations -- on either
    backend, since each frame family has a pandas and a polars impl."""
    with csv_file(make_orders()) as path:
        ops = _optimize(build_pipeline(path), backend)

    # The frame abstractions each appear and are compiled to a PhysicalOp -- no
    # abstract frame op survives to execution.
    for cls in (SelectionOp, AssignMapOp, ColumnProjectionOp, ColumnSelectorOp):
        matches = [o for o in ops if isinstance(o, cls)]
        assert matches, f"no {cls.__name__} in plan"
        assert all(isinstance(o, PhysicalOp) for o in matches), \
            f"{cls.__name__} not compiled to a physical op"

    # The filter folded into a mask predicate (not a method selection).
    sel = next(o for o in ops if isinstance(o, SelectionOp))
    assert sel.kind is SelectionKind.MASK

    # The learning ops (transformer, estimator) and the concat are present.
    assert any(isinstance(o, TransformerOp) for o in ops)
    assert any(isinstance(o, PredictorOp) for o in ops)
    assert any(isinstance(o, ConcatOp) for o in ops)


def test_transformer_binds_default_selector():
    """The configured default keeps the StringEncoder on the skrub backend.

    The legacy ``rust_backend`` flag remains available to concrete adapters, but
    it does not override the selected implementation policy.
    """

    def encoder_estimator(rust):
        with csv_file(make_orders()) as path, st.config(
                implementation_selector="default", rust_backend=rust):
            ops = _optimize(build_pipeline(path))
        transformers = [o for o in ops if isinstance(o, TransformerOp)]
        assert len(transformers) == 1
        return transformers[0].estimator

    # Default: the backend-agnostic sklearn/skrub implementation.
    default_est = encoder_estimator(rust=False)
    assert isinstance(default_est, StringEncoder)

    # The Rust flag does not override the configured default selector.
    rust_est = encoder_estimator(rust=True)
    assert type(rust_est) is type(default_est)


def test_selection_binds_query_impl_under_flag():
    """The *same* logical MASK ``SelectionOp`` binds to different pandas impls
    depending on the plan context: boolean-mask indexing by default, the
    ``DataFrame.query()`` fast path when ``OptConfig.pandas_query`` is on. The
    choice is a plan-time bind, so no ``pandas_query`` branch survives into
    execution.

    Pandas-only: the setting has no effect on the polars backend, so this test
    does not use the ``backend`` fixture."""
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    def selection_impl(enabled):
        with csv_file(make_orders()) as path:
            ops = _optimize(build_pipeline(path), pandas_query=enabled)
        sels = [o for o in ops if isinstance(o, SelectionOp)]
        assert len(sels) == 1
        # The predicate (`status == "completed" & quantity > 0`) is fully
        # query-expressible, so the flag alone decides which impl binds.
        assert sels[0].kind is SelectionKind.MASK
        return sels[0]

    # Default: boolean-mask indexing.
    assert isinstance(selection_impl(enabled=False), PandasIndexSelectionOp)
    # pandas_query on: the query() fast path is bound in at plan time.
    assert isinstance(selection_impl(enabled=True), PandasQuerySelectionOp)


def test_query_selection_trains_end_to_end():
    """With ``pandas_query`` on, the query fast path runs through the scheduler
    and the pipeline still trains and scores end-to-end (pandas backend).

    Planned and scheduled directly rather than through ``make_grid_search``:
    ``pandas_query`` is an optimizer-internal setting on ``OptConfig``, and
    ``make_grid_search`` is a skrub drop-in (ADR 0002) that takes no optimizer
    config, so it always plans with the defaults.
    """
    scorer = make_scorer(r2_score)
    with csv_file(make_orders()) as path:
        preds = build_pipeline(path)
        env = get_data(preds)
        search = SearchConfig(metric=resolve_scoring(scorer))
        with st.config(rust_backend=False, explain=("logical", "physical_impl")):
            plan, split_pos, flagged = optimize_(
                preds, OptConfig(dataframe_ops=True, pandas_query=True),
                env=env, search=search)
        # The query fast path is what is scheduled, not just what was planned.
        assert any(type(o).__name__ == "PandasQuerySelectionOp" for o in plan)
        sched = SequentialScheduler(plan, split_pos, flagged)
        sched.grid_search(cv=2)
        assert sched.results_ is not None
        assert len(sched.results_) > 0


def test_frame_ops_pipeline_grid_search():
    """The compiled plan trains and scores end-to-end through Stratum's
    scheduler, driven by ``make_grid_search`` -- the same entry point the other
    application tests use. ``make_grid_search`` builds its own plan, so this
    runs on the configured selector (the pandas-first default); the polars route
    through the scheduler is covered by ``test_selector_pipeline``. The pipeline
    carries a single candidate (no ``choose_from``), which grid search handles
    as a one-pipeline search."""
    scorer = make_scorer(r2_score)
    with csv_file(make_orders()) as path:
        preds = build_pipeline(path)
        with st.config(scheduler=True, rust_backend=False, debug_graph=False):
            search = preds.skb.make_grid_search(fitted=True, cv=2, scoring=scorer)
            assert search.results_ is not None
            assert len(search.results_) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
