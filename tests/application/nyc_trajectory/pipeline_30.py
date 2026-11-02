import skrub

from common import (LAKE, features_v28, load_acris, load_xy, apply_lgbm, model_features,
                    owner_portfolio, read_acris_grantees)

DESCRIPTION = ("+ owner portfolio (ACRIS grantee of the latest deed before cutoff): "
               "portfolio size, violations (all / Class C) per other lot of the same "
               "owner, 1y/3y, own lot excluded")
PARENT = "pipeline_28"

X, y, viol = load_xy()
feats = features_v28(X, viol)
lake = LAKE
grantees = read_acris_grantees(lake)
feats = skrub.deferred(owner_portfolio)(feats, load_acris(lake), grantees, viol)
pred = apply_lgbm(model_features(feats), y)
