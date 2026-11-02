import skrub

from common import (base_features, load_events, load_xy, apply_hgb, model_features,
                    neighbour_features, table_block)

DESCRIPTION = ("+ tax-block neighbourhood: other lots' HPD violations (all, Class C) "
               "and HPD complaints in the same block, 1y/3y, own lot excluded")
PARENT = "pipeline_07"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, ["sr311_2010", "sr311_2020"])
feats = skrub.deferred(neighbour_features)(feats, viol, "viol")
feats = skrub.deferred(neighbour_features)(feats, viol[viol["cat"] == "C"], "violC")
feats = skrub.deferred(neighbour_features)(feats, load_events("hpd_complaints"), "comp")
pred = apply_hgb(model_features(feats), y)
