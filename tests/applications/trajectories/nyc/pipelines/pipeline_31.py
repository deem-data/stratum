import skrub

from common import features_v28, load_xy, apply_lgbm, model_features, per_unit_rates

DESCRIPTION = ("+ per-unit rates: HPD violation / Class C / complaint / 311 counts "
               "(1y-10y windows) divided by residential units")
PARENT = "pipeline_28"

X, y, viol = load_xy()
feats = features_v28(X, viol)
feats = per_unit_rates(feats)
pred = apply_lgbm(model_features(feats), y)
