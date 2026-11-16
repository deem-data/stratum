"""Stage 2: tune TableVectorizer + LightGBM over six configurations."""
import skrub

from pipeline import lightgbm, load_data, table_features


def refinement(path, threads):
    X, y = load_data(path)
    features = table_features(X)
    candidates = {
        f"LightGBM lr={rate} depth={depth}": features.skb.apply(
            lightgbm(learning_rate=rate, max_depth=depth, n_jobs=threads), y=y, no_wrap=True)
        for rate in (0.01, 0.05, 0.1) for depth in (5, 7)
    }
    return skrub.choose_from(candidates, name="candidate").as_data_op()
