import skrub
from skrub import StringEncoder, TableVectorizer, ToCategorical
from xgboost import XGBClassifier

from common import features_v15, load_xy, model_features

DESCRIPTION = ("model family: XGBoost (GPU hist, native categoricals) on pipeline_15 "
               "features (fused sweep: max_depth x n_estimators, lr 0.03)")
PARENT = "pipeline_15"

X, y, viol = load_xy()
feats = features_v15(X, viol)
Xv = model_features(feats).skb.apply(TableVectorizer(low_cardinality=ToCategorical(), high_cardinality=StringEncoder(random_state=0)))
model = XGBClassifier(
    tree_method="hist", device="cpu", enable_categorical=True, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    eval_metric="aucpr", random_state=0,
    max_depth=skrub.choose_from([4, 8], name="max_depth"),
    n_estimators=skrub.choose_from([40, 120], name="n_estimators"),
)
pred = Xv.skb.apply(model, y=y)
