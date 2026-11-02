import pandas as pd
import skrub

from common import event_features, features_v15, load_events, load_xy, apply_lgbm, model_features

DESCRIPTION = ("ablation of pipeline_26: 5y windows only (read since 2015-01-01), so no "
               "window reaches into the pre-2013 open-only violations base")
PARENT = "pipeline_26"
LONG_SINCE = pd.Timestamp("2015-01-01")   # 5y before the first cutoff

X, y, viol = load_xy()
feats = features_v15(X, viol)
viol_long = load_events("hpd_violations", since=LONG_SINCE)
comp_long = load_events("hpd_complaints", since=LONG_SINCE)
feats = event_features(feats, viol_long, "violL", windows=(1825,),
                                       recency=False)
feats = event_features(feats, viol_long[viol_long["cat"] == "C"], "violCL",
                                       windows=(1825,))
feats = event_features(feats, comp_long, "compL", windows=(1825,),
                                       recency=False)
pred = apply_lgbm(model_features(feats), y)
