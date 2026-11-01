from common import base_features, load_xy, apply_hgb, model_features, table_block

DESCRIPTION = "+ hpd_omo_charges block (emergency-repair orders: counts, top-5 work types, recency, award sum)"
PARENT = "pipeline_01"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, "hpd_omo_charges", value=True)
pred = apply_hgb(model_features(feats), y)
