from common import features_v28, load_xy, apply_lgbm, model_features, table_block

DESCRIPTION = ("+ HPD complaint type mix: counts per top-10 complaint major categories "
               "(1y/3y) beyond heat/hot water")
PARENT = "pipeline_28"

X, y, viol = load_xy()
feats = features_v28(X, viol)
feats = table_block(feats, "hpd_complaints", k_cats=10)
pred = apply_lgbm(model_features(feats), y)
