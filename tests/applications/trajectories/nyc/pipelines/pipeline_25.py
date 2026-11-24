from common import features_v15, load_xy, apply_lgbm, model_features, table_block

DESCRIPTION = ("+ violation TYPE mix: counts per top-10 HPD order numbers of past Class C "
               "and of past Class B violations (1y/3y), e.g. self-closing doors, pests, "
               "window guards, lead paint, heat")
PARENT = "pipeline_21"

X, y, viol = load_xy()
feats = features_v15(X, viol)
feats = table_block(feats, "hpd_viol_C_orders", k_cats=10)
feats = table_block(feats, "hpd_viol_B_orders", k_cats=10)
pred = apply_lgbm(model_features(feats), y)
