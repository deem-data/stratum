from common import base_features, load_xy, apply_hgb, model_features, table_block

DESCRIPTION = "+ hpd_litigations block (case counts 90d/1y/3y, top-5 case types, recency)"
PARENT = "pipeline_01"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, "hpd_litigations")
pred = apply_hgb(model_features(feats), y)
