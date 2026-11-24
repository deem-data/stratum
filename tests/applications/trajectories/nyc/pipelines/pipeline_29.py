from common import features_v28, load_xy, make_ensemble, model_features

DESCRIPTION = ("pipeline_23 ensemble (HistGB:LightGBM:logreg = 2:2:1) on the pipeline_28 "
               "features (v15 + long history + ACRIS)")
PARENT = "pipeline_28"

X, y, viol = load_xy()
feats = features_v28(X, viol)
pred = model_features(feats).skb.apply(make_ensemble(), y=y)
