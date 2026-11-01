from common import base_features, load_xy, apply_hgb, model_features, table_block

DESCRIPTION = ("+ rodent_inspections + dob_complaints on pipeline_07 (the two small but "
               "fold-consistent screen positives, combined); evictions left out (drift)")
PARENT = "pipeline_07"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, ["sr311_2010", "sr311_2020"])
feats = table_block(feats, "rodent_inspections")
feats = table_block(feats, "dob_complaints")
pred = apply_hgb(model_features(feats), y)
