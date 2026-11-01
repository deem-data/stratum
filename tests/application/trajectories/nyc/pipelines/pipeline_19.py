import skrub
from sklearn.linear_model import LogisticRegression

from common import dense_scaled, features_v15, load_xy

DESCRIPTION = ("model family: logistic regression on pipeline_15 features "
               "(impute + indicators + quantile-normal; sweep C)")
PARENT = "pipeline_15"

X, y, viol = load_xy()
feats = features_v15(X, viol)
model = LogisticRegression(max_iter=3000, C=skrub.choose_from([0.01, 0.1, 1.0], name="C"))
pred = dense_scaled(feats).skb.apply(model, y=y)
