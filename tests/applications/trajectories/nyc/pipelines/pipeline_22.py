import skrub
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from skrub import StringEncoder, TableVectorizer, ToCategorical

from common import features_v15, load_xy, make_model, model_features

DESCRIPTION = ("soft-voting ensemble on pipeline_15 features: HistGB (make_model) + "
               "LightGBM (pipeline_21 winner) + logistic regression; fused choice over "
               "weights (does the linear model add diversity?)")
PARENT = "pipeline_21"

LGBM_PARAMS = dict(num_leaves=15, n_estimators=40)

X, y, viol = load_xy()
feats = features_v15(X, viol)
hgb = make_model()
lgbm = make_pipeline(
    TableVectorizer(low_cardinality=ToCategorical(), high_cardinality=StringEncoder(random_state=0)),
    LGBMClassifier(learning_rate=0.03, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, min_child_samples=50, reg_lambda=1.0,
                   n_jobs=32, random_state=0, verbose=-1, **LGBM_PARAMS))
logreg = make_pipeline(
    TableVectorizer(high_cardinality=StringEncoder(random_state=0)), SimpleImputer(strategy="median", add_indicator=True),
    QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                        subsample=200_000, random_state=0),
    LogisticRegression(C=1.0, max_iter=3000))
model = VotingClassifier(
    [("hgb", hgb), ("lgbm", lgbm), ("logreg", logreg)], voting="soft",
    weights=skrub.choose_from({"trees": [1, 1, 0], "equal": [1, 1, 1],
                               "trees2_lin1": [2, 2, 1]}, name="weights"))
pred = model_features(feats).skb.apply(model, y=y)
