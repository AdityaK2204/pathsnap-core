"""Train and compare all candidate classifiers.

Usage (from the repository root):
    python -m AM013_Abhishek_PH.train_compare [--models a,b,...]

Protocol
  * model fitting / cross-validation: training partition only (80 trips),
    5-fold GroupKFold with the trip as group, so no trip is split across folds
  * development comparison: the 10 student-testing trips (all fixes)
  * primary selection metric: mean CV observation-level Top-1 accuracy,
    log loss as tie-breaker (probabilities feed the HMM)
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from AM015_Aditya_Kalloli import config
from AM015_Aditya_Kalloli.data_io import load_trip
from AM015_Aditya_Kalloli.features import FEATURES
from AM015_Aditya_Kalloli.route_metrics import route_continuity

from .metrics import confusion, evaluate, top1_per_fix
from .model_zoo import MODEL_SPECS, SEED
from .training import fit_model

OUT = config.RESULTS_DIR / "models"
MODEL_DIR = config.ARTIFACTS_DIR / "models"
MAX_SAVE_MB = 40


def load_split(split, subsample_only=False):
    df = pd.read_parquet(config.CACHE_DIR / f"candidates_{split}.parquet")
    if subsample_only:
        df = df[df.in_subsample].reset_index(drop=True)
    return df


def continuity_on_split(df, prob, split_dir):
    best = top1_per_fix(df, prob)
    vals = []
    for tid, g in best.groupby("trip_id"):
        arcs = load_trip(split_dir, tid).arcs
        g = g.sort_values("fix")
        vals.append(route_continuity(g.pred_arc.to_numpy(), arcs))
    return float(np.mean(vals))


def run_model(name, train, test, n_folds=5, params=None):
    X, y, g = train[FEATURES].to_numpy(), train.label.to_numpy(), train.trip_id.to_numpy()
    folds = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, g)):
        t0 = time.perf_counter()
        model, _ = fit_model(name, X[tr], y[tr], g[tr], params, record_history=False)
        fit_s = time.perf_counter() - t0
        prob = model.predict_proba(X[va])[:, 1]
        m = evaluate(train.iloc[va], prob)
        folds.append(dict(model=name, fold=k, fit_seconds=fit_s, **m))
        print(f"  {name} fold {k}: top1={m['top1_accuracy']:.4f} f1={m['f1_macro']:.4f} "
              f"logloss={m['log_loss']:.4f} ({fit_s:.1f}s)", flush=True)

    t0 = time.perf_counter()
    model, hist = fit_model(name, X, y, g, params)
    fit_s = time.perf_counter() - t0
    Xt = test[FEATURES].to_numpy()
    t0 = time.perf_counter()
    prob = model.predict_proba(Xt)[:, 1]
    inf_s = time.perf_counter() - t0
    test_m = evaluate(test, prob)
    cont = continuity_on_split(test, prob, config.STUDENT_TEST_DIR)
    cm = confusion(test, prob)
    n_fixes_all = test.groupby(["trip_id", "fix"]).ngroups
    return model, hist, folds, dict(fit_seconds=fit_s, inference_seconds=inf_s,
                                    inference_ms_per_1k_fixes=1e6 * inf_s / n_fixes_all,
                                    route_continuity=cont, **test_m), cm, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_SPECS))
    args = ap.parse_args()
    names = args.models.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train", subsample_only=True)
    test = load_split("student_test")
    print(f"train rows={len(train):,} fixes={train.groupby('trip_id').fix.nunique().sum():,} | "
          f"student-test rows={len(test):,}")

    rows, fold_rows, hists, cms = [], [], {}, {}
    test_probs = pd.DataFrame({"trip_id": test.trip_id, "fix": test.fix, "arc_id": test.arc_id})
    for name in names:
        print(f"== {name}", flush=True)
        model, hist, folds, tm, cm, prob = run_model(name, train, test)
        fold_rows += folds
        fd = pd.DataFrame(folds)
        family, cfg, max_rows = MODEL_SPECS[name]
        rows.append(dict(
            model=name, family=family, config=json.dumps(cfg), train_row_cap=max_rows,
            cv_f1_macro=fd.f1_macro.mean(), cv_f1_macro_std=fd.f1_macro.std(),
            cv_top1=fd.top1_accuracy.mean(), cv_top1_std=fd.top1_accuracy.std(),
            cv_log_loss=fd.log_loss.mean(), cv_fit_seconds=fd.fit_seconds.mean(),
            test_top1=tm["top1_accuracy"], test_f1_macro=tm["f1_macro"],
            test_precision_macro=tm["precision_macro"], test_recall_macro=tm["recall_macro"],
            test_accuracy=tm["accuracy"], test_log_loss=tm["log_loss"],
            test_route_continuity=tm["route_continuity"],
            test_candidate_miss_rate=tm["candidate_miss_rate"],
            fit_seconds=tm["fit_seconds"], inference_seconds=tm["inference_seconds"],
            inference_ms_per_1k_fixes=tm["inference_ms_per_1k_fixes"],
        ))
        if hist is not None:
            hists[name] = hist
        cms[name] = cm.tolist()
        test_probs[name] = prob

        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        mb = path.stat().st_size / 1e6
        if mb > MAX_SAVE_MB:
            path.unlink()
            print(f"  {name}: model file {mb:.0f} MB > {MAX_SAVE_MB} MB, not kept (re-create with this script)")

        # save after every model so partial runs are not lost
        pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUT / "cv_fold_scores.csv", index=False)
        (OUT / "training_histories.json").write_text(json.dumps(hists))
        (OUT / "confusion_matrices.json").write_text(json.dumps(cms))
        with open(OUT / "experiment_log.jsonl", "a") as f:
            f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), model=name, seed=SEED,
                                    config=cfg, n_train_rows=len(train), result=rows[-1],
                                    folds=folds), default=float) + "\n")

    test_probs.to_parquet(OUT / "student_test_probabilities.parquet", index=False)
    comp = pd.DataFrame(rows).sort_values(["cv_top1", "cv_log_loss"], ascending=[False, True])
    best = comp.iloc[0]
    sel = dict(selected_model=best.model, rule="highest mean CV top-1 accuracy, then lowest CV log loss",
               cv_top1=best.cv_top1, cv_log_loss=best.cv_log_loss)
    (OUT / "selected_model.json").write_text(json.dumps(sel, indent=2, default=float))
    print(comp[["model", "cv_f1_macro", "cv_top1", "cv_log_loss", "test_top1", "test_f1_macro",
                "test_log_loss", "test_route_continuity", "inference_seconds"]].round(4).to_string(index=False))
    print("selected:", sel)


if __name__ == "__main__":
    main()
