from common import base_features, load_xy, apply_hgb, model_features, table_block

DESCRIPTION = "+ 311 service requests, non-HPD agencies, both 311 tables as one stream (counts, top-5 complaint types, recency)"
PARENT = "pipeline_01"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, ["sr311_2010", "sr311_2020"])
pred = apply_hgb(model_features(feats), y)
