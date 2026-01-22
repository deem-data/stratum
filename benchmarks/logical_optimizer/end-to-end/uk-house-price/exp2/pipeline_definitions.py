import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging
logging.basicConfig(level=logging.WARN)
import stratum as skrub
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans

print("pipeline_definitions.py loaded")

# Smoothed target mean encoder
class SmoothedTargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, m=20):
        self.m = m

    def fit(self, X, y):
        self.cols = list(X.columns)
        X = X.copy()
        self.global_mean_ = np.mean(y)
        self.maps_ = {}
        self.counts_ = {}
        for col in self.cols:
            stats = pd.DataFrame({"col": X[col], "y": y})
            means = stats.groupby("col")["y"].mean()
            counts = stats.groupby("col")["y"].count()
            smooth = (means * counts + self.global_mean_ * self.m) / (counts + self.m)
            self.maps_[col] = smooth
            self.counts_[col] = counts
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            mean_map = self.maps_[col]
            cnt_map = self.counts_[col]
            X[col + "_mean"] = X[col].map(mean_map).fillna(self.global_mean_)
            X[col + "_count"] = X[col].map(cnt_map).fillna(0)
        X = X.drop(columns=self.cols, errors="ignore")
        return X

class TownClusterEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=25, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None):
        X = X.copy()

        # Aggregate town-level features
        town_profile = (
            X.groupby("Town/City")
            .agg(
                {
                    "date_ordinal": "mean",
                    "Property Type": "count",
                    "is_new": "mean",
                    "is_freehold": "mean",
                }
            )
            .rename(columns={"Property Type": "town_sales"})
        )

        # Log transform sales
        town_profile["town_sales"] = np.log1p(town_profile["town_sales"])

        # Store feature columns
        self.feature_columns_ = town_profile.columns.tolist()

        # Fill NA
        town_profile = town_profile.fillna(0)

        # Scale
        self.scaler_ = StandardScaler()
        profile_scaled = self.scaler_.fit_transform(town_profile)

        # KMeans
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        town_profile["town_cluster"] = self.kmeans_.fit_predict(profile_scaled)

        # Save mapping
        self.town_cluster_map_ = town_profile["town_cluster"].to_dict()

        return self

    def transform(self, X):
        X = X.copy()

        # Map learned clusters
        X["town_cluster"] = X["Town/City"].map(self.town_cluster_map_)

        # Handle unseen towns (assign -1)
        X["town_cluster"] = X["town_cluster"].fillna(-1).astype(int)

        return X

drop_cols = ["Date of Transfer", "PPDCategory Type", "Record Status - monthly file only",'Transaction unique identifier']

def pipeline1(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year, 
        month=date_col.dt.month, 
        day=date_col.dt.day, 
        dayofweek=date_col.dt.dayofweek, 
        dayofyear=date_col.dt.dayofyear,
        quarter=date_col.dt.quarter, 
        is_month_end=date_col.dt.is_month_end.astype(int)
    )

    X2_num_cols = X2.skb.select(skrub.selectors.numeric())

    enc = SmoothedTargetMeanEncoder(m=20)
    X2_cat_cols1 = X2.skb.select(~skrub.selectors.numeric()).skb.apply(enc, y=y_log)
    X3 = X2_cat_cols1.skb.concat([X2_num_cols], axis=1).skb.apply(StandardScaler())

    model = LGBMRegressor(random_state=42, verbose=-1)
    pred_log = X3.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline2(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year, 
        month=date_col.dt.month, 
        day=date_col.dt.day, 
        dayofweek=date_col.dt.dayofweek, 
        dayofyear=date_col.dt.dayofyear, 
        quarter=date_col.dt.quarter, 
        is_month_end=date_col.dt.is_month_end.astype(int)
    )
    PropOldDur_col = X_dropped["Property Type"].astype(str) + "_" + X_dropped["Old/New"].astype(str) + "_" + X_dropped["Duration"].astype(str)
    X3 = X2.assign(PropOldDur=PropOldDur_col)

    X3_num_cols = X3.skb.select(skrub.selectors.numeric())

    enc = SmoothedTargetMeanEncoder(m=20)
    X3_cat_cols1 = X3.skb.select(~skrub.selectors.numeric()).skb.apply(enc, y=y_log)
    X4 = X3_cat_cols1.skb.concat([X3_num_cols], axis=1).skb.apply(StandardScaler())

    model = LGBMRegressor(random_state=42, verbose=-1)
    pred_log = X4.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline4(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year, 
        month=date_col.dt.month, 
        day=date_col.dt.day, 
        dayofweek=date_col.dt.dayofweek, 
        dayofyear=date_col.dt.dayofyear, 
        quarter=date_col.dt.quarter, 
        is_month_end=date_col.dt.is_month_end.astype(int)
    )
    X3 = X2.assign(
        month_sin= (2 * np.pi * date_col.dt.month / 12.0).apply(np.sin),
        month_cos= (2 * np.pi * date_col.dt.month / 12.0).apply(np.cos),
        doy_sin= (2 * np.pi * date_col.dt.dayofyear / 365.0).apply(np.sin),
        doy_cos= (2 * np.pi * date_col.dt.dayofyear / 365.0).apply(np.cos)
    )

    X3_num_cols = X3.skb.select(skrub.selectors.numeric())

    enc = SmoothedTargetMeanEncoder(m=20)
    X3_cat_cols1 = X3.skb.select(~skrub.selectors.numeric()).skb.apply(enc, y=y_log)
    X4 = X3_cat_cols1.skb.concat([X3_num_cols], axis=1).skb.apply(StandardScaler())

    model = LGBMRegressor(random_state=42, verbose=-1)
    pred_log = X4.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline5(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year, 
        month=date_col.dt.month, 
        day=date_col.dt.day, 
        dayofweek=date_col.dt.dayofweek, 
        dayofyear=date_col.dt.dayofyear, 
        quarter=date_col.dt.quarter, 
        is_month_end=date_col.dt.is_month_end.astype(int)
    )
    X3 = X2.assign(
        month_sin= (2 * np.pi * date_col.dt.month / 12.0).apply(np.sin),
        month_cos= (2 * np.pi * date_col.dt.month / 12.0).apply(np.cos),
        doy_sin= (2 * np.pi * date_col.dt.dayofyear / 365.0).apply(np.sin),
        doy_cos= (2 * np.pi * date_col.dt.dayofyear / 365.0).apply(np.cos)
    )

    PropOldDur_col = X_dropped["Property Type"].astype(str) + "_" + X_dropped["Old/New"].astype(str) + "_" + X_dropped["Duration"].astype(str)
    TownDistCounty_col = X_dropped["Town/City"].astype(str) + "_" + X_dropped["District"].astype(str) + "_" + X_dropped["County"].astype(str)
    X4 = X3.assign(
        PropOldDur=PropOldDur_col, 
        TownDistCounty=TownDistCounty_col
    )

    X4_num_cols = X4.skb.select(skrub.selectors.numeric())

    enc = SmoothedTargetMeanEncoder(m=20)
    X4_cat_cols1 = X4.skb.select(~skrub.selectors.numeric()).skb.apply(enc, y=y_log)
    X5 = X4_cat_cols1.skb.concat([X4_num_cols], axis=1).skb.apply(StandardScaler())
    
    model = LGBMRegressor(random_state=42, verbose=-1)
    pred_log = X5.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline6(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year, 
        month=date_col.dt.month, 
        day=date_col.dt.day, 
        dayofweek=date_col.dt.dayofweek, 
        dayofyear=date_col.dt.dayofyear, 
        quarter=date_col.dt.quarter, 
        is_month_end=date_col.dt.is_month_end.astype(int),
        date_ordinal = date_col.map(pd.Timestamp.toordinal)
    )
    X3 = X2.assign(
        month_sin= (2 * np.pi * date_col.dt.month / 12.0).apply(np.sin),
        month_cos= (2 * np.pi * date_col.dt.month / 12.0).apply(np.cos),
        doy_sin= (2 * np.pi * date_col.dt.dayofyear / 365.0).apply(np.sin),
        doy_cos= (2 * np.pi * date_col.dt.dayofyear / 365.0).apply(np.cos)
    )

    PropOldDur_col = X_dropped["Property Type"].astype(str) + "_" + X_dropped["Old/New"].astype(str) + "_" + X_dropped["Duration"].astype(str)
    TownDistCounty_col = X_dropped["Town/City"].astype(str) + "_" + X_dropped["District"].astype(str) + "_" + X_dropped["County"].astype(str)
    is_new_col = (X_dropped["Old/New"] == "Y").astype(int)
    is_freehold_col = (X_dropped["Duration"] == "F").astype(int)

    X4 = X3.assign(
        PropOldDur=PropOldDur_col, 
        TownDistCounty=TownDistCounty_col,
        is_freehold=is_freehold_col,
        is_new=is_new_col
    )
    X5 = X4.skb.apply(TownClusterEncoder(n_clusters=25, random_state=42, n_init=10),how="no_wrap")

    X5_num_cols = X5.skb.select(skrub.selectors.numeric())

    enc = SmoothedTargetMeanEncoder(m=20)
    X5_cat_cols1 = X5.skb.select(~skrub.selectors.numeric()).skb.apply(enc, y=y_log)
    X6 = X5_cat_cols1.skb.concat([X5_num_cols], axis=1).skb.apply(StandardScaler())
    
    model = LGBMRegressor(random_state=42, verbose=-1)
    pred_log = X6.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline7(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year, 
        month=date_col.dt.month, 
        day=date_col.dt.day, 
        dayofweek=date_col.dt.dayofweek, 
        dayofyear=date_col.dt.dayofyear,
        quarter=date_col.dt.quarter, 
        is_month_end=date_col.dt.is_month_end.astype(int)
    )

    X2_num_cols = X2.skb.select(skrub.selectors.numeric())

    enc = SmoothedTargetMeanEncoder(m=20)
    X2_cat_cols1 = X2.skb.select(~skrub.selectors.numeric()).skb.apply(enc, y=y_log)
    X3 = X2_cat_cols1.skb.concat([X2_num_cols], axis=1).skb.apply(StandardScaler())

    model = Ridge(alpha=1.0, random_state=42)
    pred_log = X3.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline8(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year, 
        month=date_col.dt.month, 
        day=date_col.dt.day, 
        dayofweek=date_col.dt.dayofweek, 
        dayofyear=date_col.dt.dayofyear,
        quarter=date_col.dt.quarter, 
        is_month_end=date_col.dt.is_month_end.astype(int)
    )

    X2_num_cols = X2.skb.select(skrub.selectors.numeric())

    enc = SmoothedTargetMeanEncoder(m=20)
    X2_cat_cols1 = X2.skb.select(~skrub.selectors.numeric()).skb.apply(enc, y=y_log)
    X3 = X2_cat_cols1.skb.concat([X2_num_cols], axis=1).skb.apply(StandardScaler())

    model = Ridge(alpha=0.1, random_state=42)
    pred_log = X3.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline9(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year, 
        month=date_col.dt.month, 
        day=date_col.dt.day, 
        dayofweek=date_col.dt.dayofweek, 
        dayofyear=date_col.dt.dayofyear,
        quarter=date_col.dt.quarter, 
        is_month_end=date_col.dt.is_month_end.astype(int)
    )

    X2_num_cols = X2.skb.select(skrub.selectors.numeric())

    enc = SmoothedTargetMeanEncoder(m=20)
    X2_cat_cols1 = X2.skb.select(~skrub.selectors.numeric()).skb.apply(enc, y=y_log)
    X3 = X2_cat_cols1.skb.concat([X2_num_cols], axis=1).skb.apply(StandardScaler())

    model = Ridge(alpha=0.01, random_state=42)
    pred_log = X3.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline_10(X,y):
    X_encoded = X.skb.apply(skrub.TableVectorizer()).skb.apply(StandardScaler())
    model1 = Ridge(alpha=1.0, random_state=42)
    pred = X_encoded.skb.apply(model1, y=y)
    return pred

def pipeline_11(X,y):
    X_encoded = X.skb.apply(skrub.TableVectorizer()).skb.apply(StandardScaler())
    model1 = Ridge(alpha=0.1, random_state=42)
    pred = X_encoded.skb.apply(model1, y=y)
    return pred

def pipeline_12(X,y):
    X_encoded = X.skb.apply(skrub.TableVectorizer()).skb.apply(StandardScaler())
    model1 = Ridge(alpha=0.01, random_state=42)
    pred = X_encoded.skb.apply(model1, y=y)
    return pred

def pipeline_13(X,y):
    X_encoded = X.skb.apply(skrub.TableVectorizer())
    model2 = LGBMRegressor(random_state=42, verbose=-1)
    pred = X_encoded.skb.apply(model2, y=y)
    return pred

def pipeline_14(X,y):
    X_encoded = X.skb.apply(skrub.TableVectorizer()).astype("float64")
    X_scaled = X_encoded.skb.apply(StandardScaler())
    model2 = ElasticNet(alpha=0.01, random_state=42, max_iter=1000)
    pred = X_scaled.skb.apply(model2, y=y)
    return pred

def pipeline20(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year, 
        month=date_col.dt.month, 
        day=date_col.dt.day, 
        dayofweek=date_col.dt.dayofweek, 
        dayofyear=date_col.dt.dayofyear, 
        quarter=date_col.dt.quarter, 
        is_month_end=date_col.dt.is_month_end.astype(int)
    )

    X2_num_cols = X2.skb.select(skrub.selectors.numeric())

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X2_cat_cols2 = X2.skb.select(~skrub.selectors.numeric()).skb.apply(enc)
    X3 = X2_cat_cols2.skb.concat([X2_num_cols.skb.apply(StandardScaler())], axis=1)
    model = LGBMRegressor(random_state=42, verbose=-1)
    pred_log = X3.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline21(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year,
        month=date_col.dt.month,
        day=date_col.dt.day,
        dayofweek=date_col.dt.dayofweek,
        quarter=date_col.dt.quarter
    )
    X3_cat_cols = X2.skb.select(~skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="most_frequent")).skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    X3_num_cols = X2.skb.select(skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="median")).skb.apply(StandardScaler())
    X4 = X3_cat_cols.skb.concat([X3_num_cols], axis=1)
    
    model = Ridge(alpha=1.0, random_state=42)
    pred_log = X4.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x, m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline22(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year,
        month=date_col.dt.month,
        day=date_col.dt.day,
        dayofweek=date_col.dt.dayofweek,
        quarter=date_col.dt.quarter
    )
    X3_cat_cols = X2.skb.select(~skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="most_frequent")).skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    X3_num_cols = X2.skb.select(skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="median")).skb.apply(StandardScaler())
    X4 = X3_cat_cols.skb.concat([X3_num_cols], axis=1)
    
    model = Ridge(alpha=0.1, random_state=42)
    pred_log = X4.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x, m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline23(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year,
        month=date_col.dt.month,
        day=date_col.dt.day,
        dayofweek=date_col.dt.dayofweek,
        quarter=date_col.dt.quarter
    )
    X3_cat_cols = X2.skb.select(~skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="most_frequent")).skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    X3_num_cols = X2.skb.select(skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="median")).skb.apply(StandardScaler())
    X4 = X3_cat_cols.skb.concat([X3_num_cols], axis=1)
    
    model = Ridge(alpha=0.01, random_state=42)
    pred_log = X4.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x, m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred


def pipeline30(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year,
        month=date_col.dt.month,
        day=date_col.dt.day,
        dayofweek=date_col.dt.dayofweek,
        quarter=date_col.dt.quarter
    )
    X3_cat_cols = X2.skb.select(~skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="most_frequent")).skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    X3_num_cols = X2.skb.select(skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="median")).skb.apply(StandardScaler())
    X4 = X3_cat_cols.skb.concat([X3_num_cols], axis=1)
    
    model = Lasso(alpha=1.0, random_state=42)
    pred_log = X4.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x, m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline31(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year,
        month=date_col.dt.month,
        day=date_col.dt.day,
        dayofweek=date_col.dt.dayofweek,
        quarter=date_col.dt.quarter
    )
    X3_cat_cols = X2.skb.select(~skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="most_frequent")).skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    X3_num_cols = X2.skb.select(skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="median")).skb.apply(StandardScaler())
    X4 = X3_cat_cols.skb.concat([X3_num_cols], axis=1)
    
    model = Lasso(alpha=0.1, random_state=42)
    pred_log = X4.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x, m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline32(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year,
        month=date_col.dt.month,
        day=date_col.dt.day,
        dayofweek=date_col.dt.dayofweek,
        quarter=date_col.dt.quarter
    )
    X3_cat_cols = X2.skb.select(~skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="most_frequent")).skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    X3_num_cols = X2.skb.select(skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="median")).skb.apply(StandardScaler())
    X4 = X3_cat_cols.skb.concat([X3_num_cols], axis=1)
    
    model = Lasso(alpha=0.01, random_state=42)
    pred_log = X4.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x, m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred

def pipeline33(X, y):
    y_log = y.skb.apply_func(np.log1p)
    date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
    X_dropped = X.drop(columns=drop_cols)
    X2 = X_dropped.assign(
        year=date_col.dt.year,
        month=date_col.dt.month,
        day=date_col.dt.day,
        dayofweek=date_col.dt.dayofweek,
        quarter=date_col.dt.quarter
    )
    X3_cat_cols = X2.skb.select(~skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="most_frequent")).skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    X3_num_cols = X2.skb.select(skrub.selectors.numeric()).skb.apply(SimpleImputer(strategy="median")).skb.apply(StandardScaler())
    X4 = X3_cat_cols.skb.concat([X3_num_cols], axis=1)
    
    model = Lasso(alpha=0.001, random_state=42)
    pred_log = X4.skb.apply(model, y=y_log)
    pred = pred_log.skb.apply_func(lambda x, m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())
    return pred