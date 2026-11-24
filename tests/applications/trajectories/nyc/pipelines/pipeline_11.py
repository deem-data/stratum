from common import base_features, load_xy, apply_hgb, model_features, table_block

DESCRIPTION = "+ evictions block (residential executed evictions: counts, recency)"
PARENT = "pipeline_01"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, "evictions", k_cats=0)
pred = apply_hgb(model_features(feats), y)
