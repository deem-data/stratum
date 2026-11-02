import skrub
from lightgbm import LGBMClassifier
from skrub import StringEncoder, TableVectorizer, ToCategorical

from common import features_v15, load_xy, model_features

DESCRIPTION = ("LightGBM follow-up: pipeline_16 winner sat on the grid edge (fewest trees, "
               "fewest leaves) -> sweep toward simpler: num_leaves {7,15} x n_estimators {150,250,400}")
PARENT = "pipeline_16"

X, y, viol = load_xy()
feats = features_v15(X, viol)
Xv = model_features(feats).skb.apply(TableVectorizer(low_cardinality=ToCategorical(), high_cardinality=StringEncoder(random_state=0)))
model = LGBMClassifier(
    learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    min_child_samples=50, reg_lambda=1.0, n_jobs=32, random_state=0, verbose=-1,
    num_leaves=skrub.choose_from([7, 15], name="num_leaves"),
    n_estimators=skrub.choose_from([15, 25, 40], name="n_estimators"),
)
pred = Xv.skb.apply(model, y=y)
