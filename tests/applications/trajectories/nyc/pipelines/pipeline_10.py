from common import base_features, load_xy, apply_hgb, model_features, table_block

DESCRIPTION = "+ rodent_inspections block (counts, top-5 results, recency)"
PARENT = "pipeline_01"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, "rodent_inspections")
pred = apply_hgb(model_features(feats), y)
