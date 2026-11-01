import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import StringEncoder, TableVectorizer, ToCategorical

from common import base_features, load_xy, model_features, table_block

DESCRIPTION = ("fused-choice HistGB sweep on pipeline_07 features: learning_rate x "
               "max_leaf_nodes x min_samples_leaf (max_iter=1000, early stopping, l2=1)")
PARENT = "pipeline_07"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, ["sr311_2010", "sr311_2020"])
feats = model_features(feats).skb.apply(TableVectorizer(low_cardinality=ToCategorical(), high_cardinality=StringEncoder(random_state=0)))
model = HistGradientBoostingClassifier(
    max_iter=100, l2_regularization=1.0, random_state=0,
    learning_rate=skrub.choose_from([0.03, 0.08], name="lr"),
    max_leaf_nodes=skrub.choose_from([15, 31, 63], name="leaves"),
    min_samples_leaf=skrub.choose_from([20, 200], name="min_leaf"),
)
pred = feats.skb.apply(model, y=y)
