"""Fitting helpers that also record epoch / iteration-wise training histories."""
import copy
import warnings

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

from .model_zoo import MODEL_SPECS, SEED, build_model

# eval_set still works in LightGBM 4.x; the newer eval_X/eval_y form is not in older releases
warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated.*")


def _group_val_split(groups, frac=0.15):
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=SEED)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


def fit_model(name, X, y, groups, params=None, record_history=True):
    """Fit one model. Returns (model, history or None).

    history is a dict of lists for models that train iteratively:
      - mlp: train/validation log loss per epoch (validation = 15 % of trips)
      - lightgbm: train/validation log loss per boosting round
      - hist_gradient_boosting: train/validation loss per iteration (internal split)
    """
    _, cfg, max_rows = MODEL_SPECS[name]
    cfg = {**cfg, **(params or {})}
    rng = np.random.default_rng(SEED)
    if max_rows is not None and len(X) > max_rows:
        sel = np.sort(rng.choice(len(X), max_rows, replace=False))
        X, y, groups = X[sel], y[sel], groups[sel]

    if name == "mlp":
        return _fit_mlp(X, y, groups, cfg, params)

    model = build_model(name, params)
    if name == "lightgbm" and record_history:
        tr, va = _group_val_split(groups)
        model.fit(X[tr], y[tr], eval_set=[(X[tr], y[tr]), (X[va], y[va])],
                  eval_names=["train", "validation"], eval_metric="binary_logloss")
        ev = model.evals_result_
        hist = dict(train=list(ev["train"]["binary_logloss"]),
                    validation=list(ev["validation"]["binary_logloss"]), unit="boosting round")
        # refit on all rows so the final model uses the full training data
        model = build_model(name, params)
        model.fit(X, y)
        return model, hist

    model.fit(X, y)
    hist = None
    if name == "hist_gradient_boosting":
        # with scoring='loss' sklearn stores the negative loss
        hist = dict(train=list(-model.train_score_), validation=list(-model.validation_score_),
                    unit="boosting iteration")
    return model, hist


def _fit_mlp(X, y, groups, cfg, params):
    epochs, patience = int(cfg["epochs"]), int(cfg["patience"])
    pipe = build_model("mlp", params)
    scaler, mlp = pipe.steps[0][1], pipe.steps[1][1]
    tr, va = _group_val_split(groups)
    scaler.fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    rng = np.random.default_rng(SEED)
    hist = dict(train=[], validation=[], unit="epoch")
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        mlp.partial_fit(Xtr[perm], y[tr][perm], classes=np.array([0, 1]))
        hist["train"].append(log_loss(y[tr], mlp.predict_proba(Xtr)[:, 1], labels=[0, 1]))
        vl = log_loss(y[va], mlp.predict_proba(Xva)[:, 1], labels=[0, 1])
        hist["validation"].append(vl)
        if vl < best - 1e-5:
            best, best_state, wait = vl, copy.deepcopy(mlp), 0
        else:
            wait += 1
            if wait >= patience:
                break
    hist["best_epoch"] = int(np.argmin(hist["validation"]) + 1)
    pipe.steps[1] = ("mlpclassifier", best_state)
    # the scaler stays fitted on the training-part rows; that is what the network saw
    return pipe, hist
