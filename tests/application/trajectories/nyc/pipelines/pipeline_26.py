import pandas as pd
import skrub

from common import event_features, features_v15, load_events, load_xy, apply_lgbm, model_features

DESCRIPTION = ("+ long history: HPD violations (all, Class C) and HPD complaints over 5y/10y "
               "before cutoff (read since 2010, pipeline-local), Class C recency up to 10y")
PARENT = "pipeline_21"
LONG_SINCE = pd.Timestamp("2010-01-01")   # 10y before the first cutoff

X, y, viol = load_xy()
feats = features_v15(X, viol)
viol_long = load_events("hpd_violations", since=LONG_SINCE)
comp_long = load_events("hpd_complaints", since=LONG_SINCE)
feats = event_features(feats, viol_long, "violL", windows=(1825, 3650),
                                       recency=False)
feats = event_features(feats, viol_long[viol_long["cat"] == "C"], "violCL",
                                       windows=(1825, 3650))
feats = event_features(feats, comp_long, "compL", windows=(1825, 3650),
                                       recency=False)
pred = apply_lgbm(model_features(feats), y)
