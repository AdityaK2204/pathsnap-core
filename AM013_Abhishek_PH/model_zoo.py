"""Candidate classifiers compared in the experiments (one per algorithm family)."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# name -> (family, config dict, max training rows or None)
MODEL_SPECS = {
    "logistic_regression": ("Linear", dict(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000), None),
    "gaussian_nb": ("Probabilistic", dict(var_smoothing=1e-9), None),
    "knn": ("Instance-based", dict(n_neighbors=25, weights="distance", metric="euclidean"), 120_000),
    "random_forest": ("Bagging ensemble", dict(n_estimators=200, min_samples_leaf=5, max_features="sqrt"), None),
    "hist_gradient_boosting": ("Boosting", dict(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                                early_stopping=True, validation_fraction=0.1,
                                                n_iter_no_change=20), None),
    "lightgbm": ("Boosting", dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8), None),
    "mlp": ("Neural network", dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                                   learning_rate_init=1e-3, batch_size=512, epochs=40, patience=5), None),
}


def build_model(name, params=None):
    family, cfg, _ = MODEL_SPECS[name]
    cfg = {**cfg, **(params or {})}
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, **cfg))
    if name == "gaussian_nb":
        return make_pipeline(StandardScaler(), GaussianNB(**cfg))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1, **cfg))
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=SEED, **cfg)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=SEED, **cfg)
    if name == "lightgbm":
        return LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **cfg)
    if name == "mlp":
        cfg = {k: v for k, v in cfg.items() if k not in ("epochs", "patience")}
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=SEED, **cfg))
    raise KeyError(name)
