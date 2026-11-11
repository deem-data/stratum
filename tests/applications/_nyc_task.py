"""The NYC housing plan, reduced to the three lake tables used by base_features.

The recorded operations follow the task's common.py. The lake path is supplied by
the application test so no external data or credentials are needed.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import stratum as skrub

SEED = 42
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


class CutoffSplit:
    def split(self, X, y=None, groups=None):
        g = np.asarray(groups)
        cuts = np.sort(np.unique(g))
        for c in cuts[1:]:
            yield np.flatnonzero(g < c), np.flatnonzero(g == c)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(np.unique(np.asarray(groups))) - 1


def make_cv():
    return CutoffSplit()


def _digits(s, width):
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(s.str.fullmatch(r"\d+")).str.zfill(width)


def _valid10(s):
    return s.where(s.str.fullmatch(r"\d{10}"))


def bbl_key(df, bbl=None, boro=None, block=None, lot=None):
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


@dataclass
class EventSpec:
    table: str
    date: str
    date_format: str | None = None
    bbl: str | None = None
    boro: str | None = None
    block: str | None = None
    lot: str | None = None
    bin: str | None = None
    cat: str | None = None
    value: str | None = None
    partitioned: bool = True
    where: dict = field(default_factory=dict)
    exclude: dict = field(default_factory=dict)

    def columns(self):
        cols = [self.date, self.bbl, self.boro, self.block, self.lot, self.bin,
                self.cat, self.value, *self.where, *self.exclude]
        return list(dict.fromkeys(c for c in cols if c))


SPECS = {
    "hpd_violations": EventSpec("hpd_violations", "inspectiondate", bbl="bbl",
                                boro="boroid", block="block", lot="lot", cat="class"),
    "hpd_complaints": EventSpec("hpd_complaints", "received_date", bbl="bbl",
                                boro="borough", block="block", lot="lot",
                                cat="major_category"),
}


def read_raw(lake, name, since=READ_SINCE, until=READ_UNTIL):
    s = SPECS[name]
    filters = ([("year", ">=", since.year), ("year", "<=", until.year)]
               if s.partitioned else None)
    table_path = skrub.as_data_op(f"{lake}/{s.table}")
    return table_path.skb.apply_func(pd.read_parquet, storage_options=STORAGE,
                                     columns=s.columns(), filters=filters)


def keep_allowed(df, name):
    for col, allowed in SPECS[name].where.items():
        df = df[df[col].isin(allowed)]
    for col, banned in SPECS[name].exclude.items():
        df = df[~df[col].isin(banned)]
    return df


def parse_dates(df, name):
    s = SPECS[name]
    return df[s.date].skb.apply_func(pd.to_datetime, format=s.date_format, errors="coerce")


def resolve_key(df, name, bridge=None):
    s = SPECS[name]
    via_bin = (df[s.bin].astype("string").str.strip().map(bridge).astype("string")
               if s.bin is not None and bridge is not None else None)
    if s.bbl is None and s.boro is None:
        return via_bin
    key = bbl_key(df, s.bbl, s.boro, s.block, s.lot)
    return key if via_bin is None else key.fillna(via_bin)


def assemble_events(df, date, key, name, since=READ_SINCE, until=READ_UNTIL):
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
    lake = LAKE if lake is None else lake
    raw = read_raw(lake, name, since)
    raw = keep_allowed(raw, name)
    date = parse_dates(raw, name)
    bridge = None
    key = resolve_key(raw, name, bridge)
    return assemble_events(raw, date, key, name, since)


def read_lots(lake, cutoffs=CUTOFFS):
    p = skrub.as_data_op(f"{lake}/pluto").skb.apply_func(
        pd.read_parquet, storage_options=STORAGE, columns=["release", *PLUTO_COLS],
        filters=[("release", "in", list(cutoffs.values()))])
    parts = []
    for cut, rel in cutoffs.items():
        q = p[(p["release"] == rel) & (p["unitsres"] >= 3)].drop(columns="release")
        q = q.assign(bbl=q["bbl"].astype("int64").astype(str), cutoff=cut)
        parts.append(q)
    first, *rest = parts
    lots = first.skb.concat(rest, axis=0) if rest else first
    return lots.sort_values(["cutoff", "bbl"], ignore_index=True)


def attach_label(lots, viol):
    keys = lots[["bbl", "cutoff"]]
    c = viol[viol["cat"] == "C"][["bbl", "date"]]
    pairs = keys.drop_duplicates().merge(c, on="bbl")
    in_window = ((pairs["date"] >= pairs["cutoff"])
                 & (pairs["date"] < pairs["cutoff"] + pd.DateOffset(years=1)))
    hits = pairs[in_window][["bbl", "cutoff"]].drop_duplicates().assign(y=1)
    y = (keys.merge(hits, on=["bbl", "cutoff"], how="left")["y"]
         .fillna(0).astype(int).set_axis(lots.index))
    return lots.assign(y=y)


def load_xy(lake, subsample=30_000):
    lots = read_lots(lake)
    viol = load_events("hpd_violations", lake)
    rows = attach_label(lots, viol)
    if subsample:
        rows = rows.skb.subsample(n=subsample, how="random")
    y = rows["y"].skb.mark_as_y()
    X = rows.drop(columns=["y"]).skb.mark_as_X(
        cv=make_cv(), split_kwargs={"groups": rows["cutoff"]})
    return X, y, viol


_PAD = "__pad__"


def event_features(X, ev, prefix, windows=(90, 365, 1095), cats=None, recency=True,
                   value=False):
    W = max(windows)
    clean = lambda s: s.replace(" ", "_").replace("/", "_")
    name = lambda s: clean(f"{prefix}_{s}")

    j = X[["bbl", "cutoff"]].drop_duplicates().merge(ev, on="bbl")
    j = j.assign(age=(j["cutoff"] - j["date"]) / pd.Timedelta(days=1))
    j = j[(j["age"] > 0) & (j["age"] <= W)]

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
    if recency: aggs[name("days_since")] = ("age", "min")
    if value:   aggs[name(f"value_{W}d")] = ("value", "sum")

    feats = j.assign(**counts).groupby(["bbl", "cutoff"]).agg(**aggs)
    if recency:
        feats = feats.assign(**{name("days_since"): feats[name("days_since")] // 1})

    zero = dict.fromkeys(set(aggs) - {name("days_since")}, 0)
    out = X.join(feats, on=["bbl", "cutoff"]).fillna(zero)
    if not rename:
        return out
    pads = [f"{prefix}_{_PAD}_n{w}d" for w in (365, 1095)]
    return out.rename(columns=rename).drop(columns=pads, errors="ignore")


def model_features(feats):
    return feats.drop(columns=["bbl", "cutoff"])


def base_features(X, viol, lake):
    feats = event_features(X, viol, "viol", cats=["A", "B", "C"])
    feats = event_features(feats, viol[viol["cat"] == "C"], "violC", windows=(90,))
    comp = load_events("hpd_complaints", lake)
    feats = event_features(feats, comp, "comp", cats=["HEAT/HOT WATER"])
    return feats


def apply_hgb(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from skrub import TableVectorizer, ToCategorical
    return (X.skb.apply(TableVectorizer(low_cardinality=ToCategorical()))
             .skb.apply(HistGradientBoostingClassifier(max_iter=50, learning_rate=0.03,
                                                       max_leaf_nodes=15,
                                                       l2_regularization=1.0,
                                                       random_state=0), y=y))
