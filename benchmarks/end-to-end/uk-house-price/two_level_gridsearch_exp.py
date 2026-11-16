"""Stage 1: evaluate two preprocessing strategies and four regressors."""
import skrub
from sklearn.linear_model import ElasticNet, Ridge
from xgboost import XGBRegressor

from pipeline import SEED, lightgbm, load_data, manual_features, table_features


def exploration(path, threads):
    X, y = load_data(path)
    features = {"Manual": manual_features(X, y), "TableVectorizer": table_features(X)}
    models = {"Ridge": Ridge(random_state=SEED),
              "ElasticNet": ElasticNet(random_state=SEED),
              "XGBoost": XGBRegressor(random_state=SEED, n_jobs=threads, verbosity=0),
              "LightGBM": lightgbm(n_jobs=threads)}
    candidates = {f"{name} + {model_name}": frame.skb.apply(model, y=y, no_wrap=True)
                  for name, frame in features.items() for model_name, model in models.items()}
    return skrub.choose_from(candidates, name="candidate").as_data_op()
