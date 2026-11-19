"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected classifier,
tuned-vs-untuned comparison and export of the final model.

Usage (from the repository root):
    python -m AM014_Addanki_Harsha_Rajeswari.bayes_opt [--trials 30] [--model NAME]

Search protocol
  * only the 80 training trips are used
  * objective: mean observation-level Top-1 accuracy over 3-fold GroupKFold
    (trip = group) - cheaper than 5 folds while still group-aware
  * TPE sampler (seed 42) + median pruner on the fold scores
The tuned and untuned models are then compared with the identical 5-fold
protocol and the student-testing partition used in the model comparison.
"""
import argparse
import json
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from AM013_Abhishek_PH.metrics import evaluate
from AM013_Abhishek_PH.model_zoo import MODEL_SPECS, SEED
from AM013_Abhishek_PH.train_compare import load_split, run_model
from AM013_Abhishek_PH.training import fit_model
from AM015_Aditya_Kalloli import config
from AM015_Aditya_Kalloli.features import FEATURES

OUT = config.RESULTS_DIR / "tuning"

SEARCH_SPACES = {
    "lightgbm": lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    ),
    "hist_gradient_boosting": lambda t: dict(
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_leaf_nodes=t.suggest_int("max_leaf_nodes", 15, 255, log=True),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 200, log=True),
        l2_regularization=t.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        max_iter=t.suggest_int("max_iter", 100, 800, step=50),
    ),
    "random_forest": lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 100, 400, step=50),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 50, log=True),
        max_features=t.suggest_float("max_features", 0.2, 0.8),
        max_depth=t.suggest_int("max_depth", 8, 40),
    ),
    "mlp": lambda t: dict(
        hidden_layer_sizes=(t.suggest_int("units_1", 32, 256, log=True), t.suggest_int("units_2", 16, 128, log=True)),
        alpha=t.suggest_float("alpha", 1e-6, 1e-2, log=True),
        learning_rate_init=t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    ),
    "logistic_regression": lambda t: dict(C=t.suggest_float("C", 1e-3, 100.0, log=True)),
}


def objective_factory(name, train, n_folds):
    X, y, g = train[FEATURES].to_numpy(), train.label.to_numpy(), train.trip_id.to_numpy()
    splits = list(GroupKFold(n_splits=n_folds).split(X, y, g))

    def objective(trial):
        params = SEARCH_SPACES[name](trial)
        scores = []
        for k, (tr, va) in enumerate(splits):
            model, _ = fit_model(name, X[tr], y[tr], g[tr], params, record_history=False)
            m = evaluate(train.iloc[va], model.predict_proba(X[va])[:, 1])
            scores.append(m["top1_accuracy"])
            trial.set_user_attr(f"fold{k}_log_loss", m["log_loss"])
            trial.set_user_attr(f"fold{k}_f1_macro", m["f1_macro"])
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def oof_probabilities(name, params, train_sub, train_all):
    """Out-of-fold probabilities for every training candidate row (used to tune the HMM)."""
    X = train_sub[FEATURES].to_numpy()
    y, g = train_sub.label.to_numpy(), train_sub.trip_id.to_numpy()
    trips = np.array(sorted(train_sub.trip_id.unique()))
    out = np.full(len(train_all), np.nan)
    for tr_trips_idx, va_trips_idx in GroupKFold(n_splits=5).split(trips, groups=trips):
        va_trips = set(trips[va_trips_idx])
        tr = ~train_sub.trip_id.isin(va_trips).to_numpy()
        model, _ = fit_model(name, X[tr], y[tr], g[tr], params, record_history=False)
        va = train_all.trip_id.isin(va_trips).to_numpy()
        out[va] = model.predict_proba(train_all.loc[va, FEATURES].to_numpy())[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sel_path = config.RESULTS_DIR / "models" / "selected_model.json"
    name = args.model or json.loads(sel_path.read_text())["selected_model"]
    print(f"tuning {name} with {args.trials} TPE trials, {args.folds}-fold group CV")

    train = load_split("train", subsample_only=True)
    test = load_split("student_test")

    storage = f"sqlite:///{(OUT / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(study_name=f"pathsnap_{name}", direction="maximize", storage=storage,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    # enqueue the untuned configuration as the first trial for reference
    if len(study.trials) == 0:
        base = dict(MODEL_SPECS[name][1])
        if name == "mlp":
            base = dict(units_1=base["hidden_layer_sizes"][0], units_2=base["hidden_layer_sizes"][1],
                        alpha=base["alpha"], learning_rate_init=base["learning_rate_init"])
        study.enqueue_trial({k: v for k, v in base.items() if k in _space_keys(name)})
    t0 = time.perf_counter()
    remaining = max(0, args.trials - len([t for t in study.trials if t.state.is_finished()]))
    study.optimize(objective_factory(name, train, args.folds), n_trials=remaining)
    search_s = time.perf_counter() - t0

    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best_params = _to_model_params(name, study.best_params)
    summary = dict(model=name, n_trials=len(study.trials),
                   n_complete=int((trials.state == "COMPLETE").sum()),
                   n_pruned=int((trials.state == "PRUNED").sum()),
                   objective=f"mean {args.folds}-fold GroupKFold observation-level Top-1 accuracy (training trips)",
                   sampler="TPE (seed 42)", pruner="MedianPruner",
                   best_value=study.best_value, best_params=study.best_params,
                   search_seconds_this_run=search_s,
                   search_space={k: str(v) for k, v in _describe_space(name).items()})
    (OUT / "best_params.json").write_text(json.dumps(summary, indent=2, default=str))
    print("best:", study.best_value, study.best_params)

    # tuned vs untuned with the same 5-fold protocol and student-test partition
    print("evaluating tuned model with the comparison protocol")
    model, hist, folds, tm, cm, prob = run_model(name, train, test, params=best_params)
    fd = pd.DataFrame(folds)
    untuned = pd.read_csv(config.RESULTS_DIR / "models" / "model_comparison.csv").set_index("model").loc[name]
    cmp_rows = [
        dict(variant="untuned", cv_f1_macro=untuned.cv_f1_macro, cv_top1=untuned.cv_top1,
             cv_log_loss=untuned.cv_log_loss, test_top1=untuned.test_top1, test_f1_macro=untuned.test_f1_macro,
             test_log_loss=untuned.test_log_loss, test_route_continuity=untuned.test_route_continuity,
             fit_seconds=untuned.fit_seconds, inference_seconds=untuned.inference_seconds,
             inference_ms_per_1k_fixes=untuned.inference_ms_per_1k_fixes),
        dict(variant="tuned", cv_f1_macro=fd.f1_macro.mean(), cv_top1=fd.top1_accuracy.mean(),
             cv_log_loss=fd.log_loss.mean(), test_top1=tm["top1_accuracy"], test_f1_macro=tm["f1_macro"],
             test_log_loss=tm["log_loss"], test_route_continuity=tm["route_continuity"],
             fit_seconds=tm["fit_seconds"], inference_seconds=tm["inference_seconds"],
             inference_ms_per_1k_fixes=tm["inference_ms_per_1k_fixes"]),
    ]
    pd.DataFrame(cmp_rows).to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "tuned_cv_fold_scores.csv", index=False)
    (OUT / "tuned_confusion_matrix.json").write_text(json.dumps(cm.tolist()))
    if hist is not None:
        (OUT / "tuned_training_history.json").write_text(json.dumps(hist))
    print(pd.DataFrame(cmp_rows).round(4).to_string(index=False))

    # final model artifact
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.ARTIFACTS_DIR / "final_model.joblib", compress=3)
    (config.ARTIFACTS_DIR / "final_model.json").write_text(json.dumps(dict(
        model=name, params=best_params, features=FEATURES, trained_on="training partition (80 trips), subsample",
        candidate_radius_m=config.CAND_RADIUS_M, candidate_max_k=config.CAND_MAX_K,
        label_max_dist_m=config.LABEL_MAX_DIST_M, seed=SEED), indent=2, default=str))

    print("out-of-fold probabilities for decoder tuning")
    train_all = pd.read_parquet(config.CACHE_DIR / "candidates_train.parquet")
    oof = oof_probabilities(name, best_params, train, train_all)
    pd.DataFrame({"trip_id": train_all.trip_id, "fix": train_all.fix, "arc_id": train_all.arc_id,
                  "prob": oof}).to_parquet(config.CACHE_DIR / "oof_train_probabilities.parquet", index=False)
    print("done")


def _space_keys(name):
    return set(_describe_space(name))


def _describe_space(name):
    """Run the search-space function against a recording trial to list the parameters."""
    class Rec:
        def __init__(self):
            self.p = {}

        def suggest_int(self, n, lo, hi, step=1, log=False):
            self.p[n] = f"int[{lo}, {hi}]" + (" log" if log else "") + (f" step {step}" if step != 1 else "")
            return lo

        def suggest_float(self, n, lo, hi, log=False):
            self.p[n] = f"float[{lo}, {hi}]" + (" log" if log else "")
            return lo

    r = Rec()
    SEARCH_SPACES[name](r)
    return r.p


def _to_model_params(name, p):
    p = dict(p)
    if name == "mlp":
        p["hidden_layer_sizes"] = (p.pop("units_1"), p.pop("units_2"))
    return p


if __name__ == "__main__":
    main()
