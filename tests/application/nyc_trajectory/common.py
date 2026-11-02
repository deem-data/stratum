"""Plan-building helpers for the NYC housing Class-C violation risk task (v2 workspace).

Copied from stratum-bench, trajectories/nyc-housing/mle-claude-run/pipelines, for
tests/application/test_nyc_trajectory.py. The only changes: the lake is a local
directory of generated tables, which the test sets as ``LAKE`` before the pipelines
are imported, and is read without storage options; GPU models run on the CPU; model
sizes (trees, iterations, epochs) are a tenth of the original, scaled alike so that
values that differed still differ; and every TableVectorizer seeds its StringEncoder,
which the original left unseeded. Docstrings and DESCRIPTIONs keep the original sizes.

Pipelines are skrub DataOps plans (skrub_dataops_guide.md). A pipeline file only
DEFINES a plan (`pred`, `DESCRIPTION`, `PARENT`); the ml-score harness scores it.
Exploration scripts reuse the same builders and end in `.skb.eval()`.

No local input/: every read is a recorded node on the GCS lake. NOTHING IS CACHED
TO DISK -- every plan and fold recomputes from the lake (RAM is plentiful).

Framing: rows = (lot, cutoff) for three backtest cutoffs, each built exactly like
the test set:
    cutoff 2020-01-01 <- PLUTO 19v2     cutoff 2021-01-01 <- PLUTO 20v7
    cutoff 2022-01-01 <- PLUTO 21v4     (test: 2023-01-01 <- 22v3)
  lots with unitsres >= 3; y = >= 1 Class C hpd_violation inspected in
  [cutoff, cutoff + 12 months), BBL per the task rule.

CV: expanding window over cutoffs (CutoffSplit): train 2020 -> score 2021, then
train 2020+2021 -> score 2022. Each fold scores ONE future year, like the test.

POINT-IN-TIME RULE (the main leakage risk). The lake is a snapshot of
2023-01-01. For a backtest cutoff T every feature may use only:
  - event rows whose event date is < T (EventSpec.date), and
  - immutable fields of those rows (type, category, amounts),
never status/disposition/closing columns (they show the 2023 state) and never
"is on a list" flags from snapshot tables. Event rows are filtered per row's own
cutoff in event_features(). The label window is [T, T+1y), disjoint from every
feature window, so no row's own label can reach its features.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import skrub

SEED = 42
LAKE = None   # set by the test
STORAGE = None

CUTOFFS = {pd.Timestamp("2020-01-01"): "19v2",
           pd.Timestamp("2021-01-01"): "20v7",
           pd.Timestamp("2022-01-01"): "21v4"}
TEST_CUTOFF, TEST_RELEASE = pd.Timestamp("2023-01-01"), "22v3"
HISTORY_YEARS = 3
FIRST_CUTOFF = min(CUTOFFS)
READ_SINCE = FIRST_CUTOFF - pd.DateOffset(years=HISTORY_YEARS)   # 2017-01-01
READ_UNTIL = TEST_CUTOFF                                          # lake end

PLUTO_COLS = ["bbl", "borocode", "unitsres", "unitstotal", "numbldgs", "numfloors",
              "yearbuilt", "yearalter1", "bldgarea", "resarea", "lotarea",
              "assesstot", "bldgclass", "ownertype", "cd"]
BORO_CODE = {"MANHATTAN": "1", "BRONX": "2", "BROOKLYN": "3", "QUEENS": "4",
             "STATEN ISLAND": "5", "MN": "1", "BX": "2", "BK": "3", "QN": "4", "SI": "5",
             "1": "1", "2": "2", "3": "3", "4": "4", "5": "5"}


# --- CV --------------------------------------------------------------------------
class CutoffSplit:
    """Expanding-window split over cutoff groups: fold k trains on all cutoffs
    before the (k+1)-th and scores on the (k+1)-th. Needs groups = cutoff column."""

    def split(self, X, y=None, groups=None):
        g = np.asarray(groups)
        cuts = np.sort(np.unique(g))
        for c in cuts[1:]:
            yield np.flatnonzero(g < c), np.flatnonzero(g == c)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(np.unique(np.asarray(groups))) - 1

    def __repr__(self):
        return "CutoffSplit(expanding: 2020->2021, 2020+2021->2022)"


def make_cv():
    return CutoffSplit()


# --- key resolution ----------------------------------------------------------------
def _digits(s, width):
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(s.str.fullmatch(r"\d+")).str.zfill(width)


def _valid10(s):
    """Keep values that are exactly 10 digits, NA otherwise."""
    return s.where(s.str.fullmatch(r"\d{10}"))


def bbl_key(df, bbl=None, boro=None, block=None, lot=None):
    """Task BBL rule: a 10-digit `bbl` value if present, else boro(1) + block(5) + lot(4).

    Recorded ops only (which branches exist is decided at build time), so it
    lowers to fine-grained string ops instead of one opaque UDF."""
    if bbl is None and boro is None:
        raise ValueError("pass `bbl` or `boro`/`block`/`lot`")
    out = None
    if bbl is not None:
        out = _valid10(df[bbl].astype("string").str.strip()
                              .str.replace(r"\.0$", "", regex=True))
    if boro is not None:
        built = _valid10(df[boro].astype("string").str.strip().str.upper().map(BORO_CODE)
                         + _digits(df[block], 5) + _digits(df[lot], 4))
        out = built if out is None else out.fillna(built)
    return out.astype("string")


def read_bin_bridge(lake):
    """BIN -> BBL from building_footprints (base_bbl), one BBL per BIN."""
    path = skrub.as_data_op(f"{lake}/building_footprints")
    f = path.skb.apply_func(pd.read_parquet, storage_options=STORAGE, columns=["bin", "base_bbl"])
    f = f.assign(bin=f["bin"].astype("string"),
                 bbl=bbl_key(f, bbl="base_bbl")).dropna(subset=["bbl"])
    return f.drop_duplicates("bin").set_index("bin")["bbl"]


# --- event tables --------------------------------------------------------------------
@dataclass
class EventSpec:
    """How to turn one lake table into (bbl, date, cat, value) events."""
    table: str
    date: str
    date_format: str | None = None      # for text dates
    bbl: str | None = None
    boro: str | None = None
    block: str | None = None
    lot: str | None = None
    bin: str | None = None              # fallback via read_bin_bridge
    cat: str | None = None              # immutable category column
    value: str | None = None            # optional numeric column to sum
    partitioned: bool = True
    where: dict = field(default_factory=dict)   # {col: allowed values}
    exclude: dict = field(default_factory=dict)  # {col: excluded values}

    def columns(self):
        cols = [self.date, self.bbl, self.boro, self.block, self.lot, self.bin,
                self.cat, self.value, *self.where, *self.exclude]
        return list(dict.fromkeys(c for c in cols if c))


SPECS = {
    "hpd_violations": EventSpec("hpd_violations", "inspectiondate", bbl="bbl",
                                boro="boroid", block="block", lot="lot", cat="class"),
    # violation TYPE mix: category = HPD order number, one stream per class
    "hpd_viol_C_orders": EventSpec("hpd_violations", "inspectiondate", bbl="bbl",
                                   boro="boroid", block="block", lot="lot",
                                   cat="ordernumber", where={"class": ["C"]}),
    "hpd_viol_B_orders": EventSpec("hpd_violations", "inspectiondate", bbl="bbl",
                                   boro="boroid", block="block", lot="lot",
                                   cat="ordernumber", where={"class": ["B"]}),
    "hpd_complaints": EventSpec("hpd_complaints", "received_date", bbl="bbl",
                                boro="borough", block="block", lot="lot",
                                cat="major_category"),
    "hpd_litigations": EventSpec("hpd_litigations", "caseopendate", bbl="bbl",
                                 boro="boroid", block="block", lot="lot",
                                 cat="casetype", partitioned=False),
    "hpd_vacate_orders": EventSpec("hpd_vacate_orders", "vacate_effective_date",
                                   bbl="bbl", cat="primary_vacate_reason",
                                   value="number_of_vacated_units", partitioned=False),
    "hpd_omo_charges": EventSpec("hpd_omo_charges", "omocreatedate", bbl="bbl",
                                 boro="boro_id", block="block", lot="lot",
                                 cat="worktypegeneral", value="omoawardamount",
                                 partitioned=False),
    "hpd_hwo_charges": EventSpec("hpd_hwo_charges", "hwocreatedate", bbl="bbl",
                                 boro="boroid", block="block", lot="lot",
                                 cat="worktypegeneral", value="hwoapprovedamount",
                                 partitioned=False),
    "hpd_bedbug_reports": EventSpec("hpd_bedbug_reports", "filing_date", bbl="bbl",
                                    value="infested_dwelling_unit_count",
                                    partitioned=False),
    "hpd_aep_buildings": EventSpec("hpd_aep_buildings", "aep_start_date", bbl="bbl",
                                   partitioned=False),
    "dob_violations": EventSpec("dob_violations", "issue_date", date_format="%Y%m%d",
                                boro="boro", block="block", lot="lot", bin="bin",
                                cat="violation_type_code"),
    "dob_ecb_violations": EventSpec("dob_ecb_violations", "issue_date",
                                    date_format="%Y%m%d", boro="boro", block="block",
                                    lot="lot", bin="bin", cat="severity"),
    "dob_complaints": EventSpec("dob_complaints", "date_entered",
                                date_format="%m/%d/%Y", bin="bin",
                                cat="complaint_category"),
    "rodent_inspections": EventSpec("dohmh_rodent_inspections", "inspection_date",
                                    bbl="bbl", boro="boro_code", block="block",
                                    lot="lot", cat="result"),
    "evictions": EventSpec("evictions", "executed_date", bbl="bbl", partitioned=False,
                           where={"residential_commercial_ind": ["Residential"]}),
    "tax_liens": EventSpec("dof_tax_lien_sales", "month", boro="borough", block="block",
                           lot="lot", cat="water_debt_only", partitioned=False),
    # 311, HPD-routed requests excluded (they are hpd_complaints already)
    "sr311_2010": EventSpec("service_requests_311_2010_2019", "created_date", bbl="bbl",
                            cat="complaint_type", exclude={"agency": ["HPD"]}),
    "sr311_2020": EventSpec("service_requests_311_2020_present", "created_date",
                            bbl="bbl", cat="complaint_type", exclude={"agency": ["HPD"]}),
}


def read_raw(lake, name, since=READ_SINCE, until=READ_UNTIL):
    """Step 1 -- the parquet read: spec columns, year partitions of the window."""
    s = SPECS[name]
    filters = ([("year", ">=", since.year), ("year", "<=", until.year)]
               if s.partitioned else None)
    table_path = skrub.as_data_op(f"{lake}/{s.table}")
    return table_path.skb.apply_func(pd.read_parquet, storage_options=STORAGE,
                                     columns=s.columns(), filters=filters)


def keep_allowed(df, name):
    """Step 2 -- row filter on immutable fields (e.g. residential evictions)."""
    for col, allowed in SPECS[name].where.items():
        df = df[df[col].isin(allowed)]
    for col, banned in SPECS[name].exclude.items():
        df = df[~df[col].isin(banned)]
    return df


def parse_dates(df, name):
    """Step 3 -- the event date (text dates parsed with the spec's format)."""
    s = SPECS[name]
    return df[s.date].skb.apply_func(pd.to_datetime, format=s.date_format, errors="coerce")


def resolve_key(df, name, bridge=None):
    """Step 4 -- task BBL rule, then BIN -> BBL via the footprint bridge. A table
    with only a BIN (dob_complaints) is keyed by the bridge alone."""
    s = SPECS[name]
    via_bin = (df[s.bin].astype("string").str.strip().map(bridge).astype("string")
               if s.bin is not None and bridge is not None else None)
    if s.bbl is None and s.boro is None:
        return via_bin
    key = bbl_key(df, s.bbl, s.boro, s.block, s.lot)
    return key if via_bin is None else key.fillna(via_bin)


def assemble_events(df, date, key, name, since=READ_SINCE, until=READ_UNTIL):
    """Step 5 -- DataFrame[bbl, date, cat, value], history window, keyed rows only."""
    s = SPECS[name]
    ev = key.to_frame("bbl").assign(
        date=date,
        cat=df[s.cat].astype("string").str.strip() if s.cat else pd.NA,
        value=(df[s.value].skb.apply_func(pd.to_numeric, errors="coerce") if s.value
               else np.nan),
    )
    ev = ev[(ev["date"] >= since) & (ev["date"] < until)]
    return ev.dropna(subset=["bbl"]).reset_index(drop=True)


def load_events(name, lake=None, since=READ_SINCE):
    """Plan nodes: read -> filter -> dates -> key -> events (BIN tables get the bridge).
    `since` widens the read window for long-history blocks (default: 3y before
    the first cutoff); features still only use events before each row's cutoff."""
    lake = LAKE if lake is None else lake
    raw = read_raw(lake, name, since)
    raw = keep_allowed(raw, name)
    date = parse_dates(raw, name)
    bridge = (read_bin_bridge(lake) if SPECS[name].bin is not None
              else None)
    key = resolve_key(raw, name, bridge)
    return assemble_events(raw, date, key, name, since)


def load_union(names, lake=None):
    """One event stream from several tables with the same meaning (e.g. the two
    311 tables, split at 2020): each read as its own node chain, then concatenated."""
    lake = LAKE if lake is None else lake
    first, *rest = [load_events(n, lake) for n in names]
    return first.skb.concat(rest, axis=0).reset_index(drop=True)


# --- rows + raw target ------------------------------------------------------------
def read_lots(lake, cutoffs=CUTOFFS):
    """(lot, cutoff) rows: each cutoff's PLUTO release, unitsres >= 3."""
    p = skrub.as_data_op(f"{lake}/pluto").skb.apply_func(
        pd.read_parquet, storage_options=STORAGE, columns=["release", *PLUTO_COLS],
        filters=[("release", "in", list(cutoffs.values()))])
    parts = []
    for cut, rel in cutoffs.items():   # build-time loop over the constant cutoffs
        q = p[(p["release"] == rel) & (p["unitsres"] >= 3)].drop(columns="release")
        q = q.assign(bbl=q["bbl"].astype("int64").astype(str), cutoff=cut)
        parts.append(q)
    first, *rest = parts
    lots = first.skb.concat(rest, axis=0) if rest else first
    return lots.sort_values(["cutoff", "bbl"], ignore_index=True)


def attach_label(lots, viol):
    """y = >= 1 Class C violation inspected in [cutoff, cutoff + 12 months)."""
    keys = lots[["bbl", "cutoff"]]
    c = viol[viol["cat"] == "C"][["bbl", "date"]]
    pairs = keys.drop_duplicates().merge(c, on="bbl")
    in_window = ((pairs["date"] >= pairs["cutoff"])
                 & (pairs["date"] < pairs["cutoff"] + pd.DateOffset(years=1)))
    hits = pairs[in_window][["bbl", "cutoff"]].drop_duplicates().assign(y=1)
    y = (keys.merge(hits, on=["bbl", "cutoff"], how="left")["y"]
         .fillna(0).astype(int).set_axis(lots.index))
    return lots.assign(y=y)


def load_xy(subsample=30_000):
    """Recorded reads -> (lot, cutoff) rows + raw label, marked EARLY.

    Returns (X, y, viol). X keeps `bbl` and `cutoff` (needed by the feature
    builders); drop both before the model (see model_features).
    """
    lake = LAKE
    lots = read_lots(lake)
    viol = load_events("hpd_violations", lake)
    rows = attach_label(lots, viol)
    if subsample:
        rows = rows.skb.subsample(n=subsample, how="random")
    y = rows["y"].skb.mark_as_y()
    X = rows.drop(columns=["y"]).skb.mark_as_X(
        cv=make_cv(), split_kwargs={"groups": rows["cutoff"]})
    return X, y, viol


# --- generic pre-cutoff event features --------------------------------------------
def top_categories(ev, k):
    """Top-k categories by frequency among events BEFORE the first cutoff (so the
    choice itself never looks at label-window data)."""
    early = ev[ev["date"] < FIRST_CUTOFF]
    return early["cat"].value_counts().head(k).index.tolist()


_PAD = "__pad__"   # never a real category; see table_block


def _pad_to(cats, k):
    """Pad a category list to length k with _PAD (one call: stratum lowers a
    recorded list `+` to an elementwise numeric add)."""
    return (list(cats) + [_PAD] * k)[:k]


def event_features(X, ev, prefix, windows=(90, 365, 1095), cats=None, recency=True,
                   value=False):
    """Per (bbl, cutoff): event counts in the last w days before cutoff,
    per-category counts (365d / 1095d), days since last event, optional
    value sum. Strictly date < cutoff, and only the last max(windows) days.

    Written as recorded ops (one SQL-like block, not one opaque UDF). `cats` is a
    list whose items are strings or DataOps (table_block's data-derived top-k);
    a DataOp category gets a placeholder column, renamed at the end, so the
    column set and order do not depend on knowing the category at build time.
    """
    W = max(windows)
    clean = lambda s: s.replace(" ", "_").replace("/", "_")
    name = lambda s: clean(f"{prefix}_{s}")

    # FROM (SELECT DISTINCT bbl, cutoff FROM X) JOIN ev USING (bbl)
    # WHERE cutoff - W days <= date < cutoff
    j = X[["bbl", "cutoff"]].drop_duplicates().merge(ev, on="bbl")
    j = j.assign(age=(j["cutoff"] - j["date"]) / pd.Timedelta(days=1))
    j = j[(j["age"] > 0) & (j["age"] <= W)]

    # SUM(CASE WHEN <cond> THEN 1 ELSE 0 END) AS <name>
    counts = {name(f"n{w}d"): j["age"].le(w) for w in windows}
    rename = {}
    for i, c in enumerate(cats or []):
        for w in (365, 1095):
            if isinstance(c, str):
                col = name(f"{c}_n{w}d")
            else:
                col = f"{prefix}__cat{i}_n{w}d"
                rename[col] = f"{prefix}_" + c.replace(" ", "_").replace("/", "_") + f"_n{w}d"
            counts[col] = j["cat"].eq(c) & j["age"].le(w)

    aggs = {k: (k, "sum") for k in counts}
    if recency: aggs[name("days_since")] = ("age", "min")      # MIN(age)
    if value:   aggs[name(f"value_{W}d")] = ("value", "sum")   # SUM(value)

    feats = j.assign(**counts).groupby(["bbl", "cutoff"]).agg(**aggs)
    if recency:   # whole days, as (cutoff - last date).dt.days
        feats = feats.assign(**{name("days_since"): feats[name("days_since")] // 1})

    # X LEFT JOIN feats, COALESCE(counts, 0); days_since stays NULL
    zero = dict.fromkeys(set(aggs) - {name("days_since")}, 0)
    out = X.join(feats, on=["bbl", "cutoff"]).fillna(zero)
    if not rename:
        return out
    pads = [f"{prefix}_{_PAD}_n{w}d" for w in (365, 1095)]   # table_block padding
    return out.rename(columns=rename).drop(columns=pads, errors="ignore")


def model_features(feats):
    """Drop the row keys before the model."""
    return feats.drop(columns=["bbl", "cutoff"])


# --- shared base block + model (the anchor, pipeline_01) --------------------------
def base_features(X, viol, lake=None):
    """Anchor feature set (port of v1 pipeline_06): PLUTO attributes (already in X)
    + HPD violation history (counts 90d/1y/3y, per class A/B/C, recency), Class C
    recency, HPD complaint history (counts, heat/hot water, recency).
    One recorded node per block."""
    feats = event_features(X, viol, "viol", cats=["A", "B", "C"])
    feats = event_features(feats, viol[viol["cat"] == "C"], "violC",
                                           windows=(90,))
    comp = load_events("hpd_complaints", lake)
    feats = event_features(feats, comp, "comp", cats=["HEAT/HOT WATER"])
    return feats


def make_model():
    """TableVectorizer (low-cardinality -> categorical) + HistGB, v1-tuned params."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from skrub import StringEncoder, TableVectorizer, ToCategorical
    return make_pipeline(
        TableVectorizer(low_cardinality=ToCategorical(), high_cardinality=StringEncoder(random_state=0)),
        HistGradientBoostingClassifier(max_iter=50, learning_rate=0.03,
                                       max_leaf_nodes=15, l2_regularization=1.0,
                                       random_state=0))


def apply_hgb(X, y):
    """make_model as two recorded steps (vectorizer, then HistGB), so each is its
    own node; make_model stays the Pipeline for the voting ensembles."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from skrub import StringEncoder, TableVectorizer, ToCategorical
    return (X.skb.apply(TableVectorizer(low_cardinality=ToCategorical(), high_cardinality=StringEncoder(random_state=0)))
             .skb.apply(HistGradientBoostingClassifier(max_iter=50, learning_rate=0.03,
                                                       max_leaf_nodes=15,
                                                       l2_regularization=1.0,
                                                       random_state=0), y=y))


# --- Custom / weighted scoring (unused: average_precision is a scorer string) -----
SCORER = None
SCORER_NAME = None


def attach_scoring(pred, sample_weight=None):
    if SCORER is None:
        return pred
    kwargs = {"sample_weight": sample_weight} if sample_weight is not None else None
    return pred.skb.with_scoring(SCORER, kwargs=kwargs, name=SCORER_NAME)


# --- one table = one feature block --------------------------------------------------
def table_block(feats, name, lake=None, k_cats=5, value=False, windows=(90, 365, 1095)):
    """Add one lake table's pre-cutoff event block to `feats`.

    `name` is a SPECS key or a list of keys read as one stream (load_union).
    Recorded nodes: event read chain -> top-k categories (chosen on events
    before the first cutoff) -> event_features block.
    """
    ev = load_union(name, lake) if isinstance(name, (list, tuple)) else load_events(name, lake)
    prefix = name if isinstance(name, str) else name[0].rsplit("_", 1)[0]
    cats = None
    if k_cats:   # pad to k (a table may have fewer categories); pads are dropped
        top = top_categories(ev, k_cats).skb.apply_func(_pad_to, k_cats)
        cats = [top[i] for i in range(k_cats)]
    return event_features(feats, ev, prefix, windows=windows, cats=cats, value=value)


def neighbour_features(X, ev, prefix, windows=(365, 1095)):
    """Per (lot, cutoff): events in the same tax block (boro+block = bbl[:6]) in the
    last w days BEFORE the cutoff, EXCLUDING the lot's own events, plus the same
    per other multi-dwelling lot of the block. Pre-cutoff events only, no labels."""
    new = pd.DataFrame(index=X.index)
    for cut in X["cutoff"].unique():
        m = (X["cutoff"] == cut).to_numpy()
        keys = X.loc[m, "bbl"]
        blocks = keys.str[:6]
        n_other = blocks.map(blocks.value_counts()) - 1
        for w in windows:
            past = ev[(ev["date"] < cut) & (ev["date"] >= cut - pd.Timedelta(days=w))]
            own = keys.map(past["bbl"].value_counts()).fillna(0)
            blk = blocks.map(past["bbl"].str[:6].value_counts()).fillna(0)
            other = blk - own
            new.loc[m, f"{prefix}_blk_n{w}d"] = other.to_numpy()
            new.loc[m, f"{prefix}_blk_per_lot{w}d"] = (other / n_other.where(n_other > 0)).to_numpy()
    return pd.concat([X, new], axis=1)


# --- the pipeline_15 feature set, shared by the model-family comparison -----------
def features_v15(X, viol, lake=None):
    """pipeline_15 features: base + non-HPD 311 + rodent inspections + DOB complaints."""
    feats = base_features(X, viol, lake)
    feats = table_block(feats, ["sr311_2010", "sr311_2020"], lake)
    feats = table_block(feats, "rodent_inspections", lake)
    feats = table_block(feats, "dob_complaints", lake)
    return feats


def dense_scaled(feats):
    """Numeric, NaN-free, scaled matrix for linear models / NNs: TableVectorizer
    (one-hot low-cardinality) -> median impute + missing indicators -> quantile to
    normal (the count features are heavy-tailed). Each step its own node."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer
    from skrub import StringEncoder, TableVectorizer
    Xv = model_features(feats).skb.apply(TableVectorizer(high_cardinality=StringEncoder(random_state=0)))
    Xv = Xv.skb.apply(SimpleImputer(strategy="median", add_indicator=True))
    return Xv.skb.apply(QuantileTransformer(output_distribution="normal",
                                            n_quantiles=1000, subsample=200_000,
                                            random_state=0))


# --- ACRIS (property records): dated documents -> ownership / financing events ----
# Point-in-time: a document counts only from its recorded_datetime (filtered per
# row's cutoff in event_features). Party names are NOT used here.
ACRIS_TYPES = ["DEED", "MTGE", "AL&R", "SAT", "AGMT", "ASST", "RPTT&RET"]
ACRIS_SINCE = pd.Timestamp("2008-01-01")   # 12y before the first cutoff (sale recency)


def read_acris_master(lake, since=ACRIS_SINCE):
    m = skrub.as_data_op(f"{lake}/acris_master").skb.apply_func(
        pd.read_parquet, storage_options=STORAGE,
        columns=["document_id", "doc_type", "document_amt", "recorded_datetime"],
        filters=[("year", ">=", since.year)])
    return m[m["doc_type"].isin(ACRIS_TYPES) & (m["recorded_datetime"] >= since)
             & (m["recorded_datetime"] < READ_UNTIL)]


def read_acris_legals(lake):
    lg = skrub.as_data_op(f"{lake}/acris_legals").skb.apply_func(
        pd.read_parquet, storage_options=STORAGE,
        columns=["document_id", "borough", "block", "lot"])
    bbl = bbl_key(lg, boro="borough", block="block", lot="lot")
    return lg[["document_id"]].assign(bbl=bbl).dropna().drop_duplicates()


def acris_events(master, legals):
    """(bbl, date, cat=doc_type, value=document_amt), one row per document x lot."""
    ev = master.merge(legals, on="document_id")
    cols = {"bbl": "bbl", "recorded_datetime": "date", "doc_type": "cat",
            "document_amt": "value", "document_id": "document_id"}
    return ev[list(cols)].rename(columns=cols).reset_index(drop=True)


def load_acris(lake=None):
    lake = LAKE if lake is None else lake
    return acris_events(read_acris_master(lake), read_acris_legals(lake))


def make_lgbm():
    """pipeline_21 winner: LightGBM 400 trees / 15 leaves -- the cheap single model
    used to screen new feature blocks before they go into the ensemble."""
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import make_pipeline
    from skrub import StringEncoder, TableVectorizer, ToCategorical
    return make_pipeline(
        TableVectorizer(low_cardinality=ToCategorical(), high_cardinality=StringEncoder(random_state=0)),
        LGBMClassifier(learning_rate=0.03, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, min_child_samples=50, reg_lambda=1.0,
                       n_jobs=32, random_state=0, verbose=-1, num_leaves=15,
                       n_estimators=40))


def apply_lgbm(X, y):
    """make_lgbm as two recorded steps (vectorizer, then LightGBM), so each is its
    own node; make_lgbm stays the Pipeline for the voting ensemble."""
    from lightgbm import LGBMClassifier
    from skrub import StringEncoder, TableVectorizer, ToCategorical
    return (X.skb.apply(TableVectorizer(low_cardinality=ToCategorical(), high_cardinality=StringEncoder(random_state=0)))
             .skb.apply(LGBMClassifier(learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, min_child_samples=50,
                                       reg_lambda=1.0, n_jobs=32, random_state=0,
                                       verbose=-1, num_leaves=15, n_estimators=40),
                        y=y))


# --- long history + ACRIS on top of features_v15 (pipelines 26 / 24 -> 28) -------
LONG_SINCE = pd.Timestamp("2010-01-01")   # 10y before the first cutoff (see README:
                                          # pre-2013 violations are an open-only base)


def long_history(feats, lake=None):
    """pipeline_26 block: HPD violations (all, Class C) + complaints over 5y/10y."""
    viol_long = load_events("hpd_violations", lake, since=LONG_SINCE)
    comp_long = load_events("hpd_complaints", lake, since=LONG_SINCE)
    feats = event_features(feats, viol_long, "violL",
                                           windows=(1825, 3650), recency=False)
    feats = event_features(feats, viol_long[viol_long["cat"] == "C"],
                                           "violCL", windows=(1825, 3650))
    return event_features(feats, comp_long, "compL",
                                          windows=(1825, 3650), recency=False)


def acris_block(feats, lake=None):
    """pipeline_24 block: ACRIS document counts/types/amounts + deed recency."""
    acris = load_acris(lake)
    feats = event_features(feats, acris, "acris", windows=(365, 1095, 3650),
                                           cats=["DEED", "MTGE", "AL&R", "SAT"], value=True)
    return event_features(feats, acris[acris["cat"] == "DEED"], "deed",
                                          windows=(3650,), value=True)


def features_v28(X, viol, lake=None):
    return acris_block(long_history(features_v15(X, viol, lake), lake), lake)


def make_ensemble(weights=(2, 2, 1)):
    """pipeline_23 ensemble: soft vote HistGB (make_model) : LightGBM (make_lgbm) :
    logistic regression on dense quantile-scaled input."""
    from sklearn.ensemble import VotingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer
    from skrub import StringEncoder, TableVectorizer
    logreg = make_pipeline(
        TableVectorizer(high_cardinality=StringEncoder(random_state=0)), SimpleImputer(strategy="median", add_indicator=True),
        QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                            subsample=200_000, random_state=0),
        LogisticRegression(C=1.0, max_iter=3000))
    return VotingClassifier([("hgb", make_model()), ("lgbm", make_lgbm()),
                             ("logreg", logreg)], voting="soft", weights=list(weights))


# --- owner portfolios (ACRIS grantee of the latest deed before each cutoff) --------
def read_acris_grantees(lake):
    """Grantee names (party_type 2), normalized: upper-case, punctuation and
    corporate suffixes stripped, whitespace collapsed."""
    pa = skrub.as_data_op(f"{lake}/acris_parties").skb.apply_func(
        pd.read_parquet, storage_options=STORAGE,
        columns=["document_id", "party_type", "name"],
        filters=[("party_type", "==", "2")])
    name = (pa["name"].astype("string").str.upper()
            .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
            .str.replace(r"\b(LLC|INC|CORP|CORPORATION|LP|L P|CO|LTD|THE)\b", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())
    return pa[["document_id"]].assign(owner=name).dropna().query("owner != ''")


def owner_portfolio(X, acris, grantees, viol, windows=(365, 1095)):
    """Per (lot, cutoff): owner = first grantee (alphabetical) of the lot's latest DEED
    recorded before the cutoff; portfolio = all lots with that owner as of the
    cutoff. Features: portfolio size and violations (all / Class C) per OTHER lot
    of the portfolio in the last w days before the cutoff (own lot excluded).
    Everything dated before the cutoff; violation events, never labels."""
    deeds = acris[acris["cat"] == "DEED"]
    first_grantee = grantees.sort_values("owner").drop_duplicates("document_id")
    new = pd.DataFrame(index=X.index)
    for cut in X["cutoff"].unique():
        m = (X["cutoff"] == cut).to_numpy()
        keys = X.loc[m, "bbl"]
        d = deeds[deeds["date"] < cut].sort_values("date").drop_duplicates("bbl", keep="last")
        owner_of = d.merge(first_grantee, on="document_id").set_index("bbl")["owner"]
        size = owner_of.value_counts()
        own_owner = keys.map(owner_of)
        n_other = own_owner.map(size).fillna(1) - 1
        new.loc[m, "owner_n_lots"] = own_owner.map(size).to_numpy()
        for label, ev in (("all", viol), ("C", viol[viol["cat"] == "C"])):
            for w in windows:
                past = ev[(ev["date"] < cut) & (ev["date"] >= cut - pd.Timedelta(days=w))]
                per_lot = past["bbl"].value_counts()
                per_owner = per_lot.groupby(owner_of.reindex(per_lot.index)).sum()
                own = keys.map(per_lot).fillna(0)
                other = own_owner.map(per_owner).fillna(0) - own
                new.loc[m, f"owner_{label}_per_other_lot_{w}d"] = (
                    other / n_other.where(n_other > 0)).to_numpy()
    return pd.concat([X, new], axis=1)


def per_unit_rates(feats):
    """History counts divided by residential units (a count means different things
    for a 3-unit and a 100-unit building). Row-local, no cross-row information."""
    units = feats["unitsres"].where(feats["unitsres"] > 0)
    counts = feats.filter(regex=r"^(viol_n|viol_C_n|violL_n|violCL_n|comp_n|compL_n|sr311_n)")
    rates = counts.div(units, axis=0).add_suffix("_per_unit")
    return feats.join(rates)
