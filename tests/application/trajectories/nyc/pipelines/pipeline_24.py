import skrub

from common import event_features, features_v15, load_acris, load_xy, apply_lgbm, model_features

DESCRIPTION = ("+ ACRIS ownership/financing events (recorded before cutoff): document counts "
               "1y/3y/10y, deed/mortgage/AL&R/satisfaction counts, recency, amounts; "
               "deed-only 10y count, days since last sale, sale amounts")
PARENT = "pipeline_21"

X, y, viol = load_xy()
feats = features_v15(X, viol)
acris = load_acris()
feats = event_features(feats, acris, "acris", windows=(365, 1095, 3650),
                                       cats=["DEED", "MTGE", "AL&R", "SAT"], value=True)
feats = event_features(feats, acris[acris["cat"] == "DEED"], "deed",
                                       windows=(3650,), value=True)
pred = apply_lgbm(model_features(feats), y)
