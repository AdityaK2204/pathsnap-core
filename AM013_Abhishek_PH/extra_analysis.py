"""Additional evidence for the model comparison.

  * training (in-sample) vs cross-validation vs student-test scores -> over/under-fitting
  * per-trip student-test Top-1 for every model
  * per-class precision / recall on student-test
  * LightGBM feature importance (final tuned model)

Usage (from the repository root, after train_compare and bayes_opt):
    python -m AM013_Abhishek_PH.extra_analysis
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from AM015_Aditya_Kalloli import config
from AM015_Aditya_Kalloli.features import FEATURES

from .metrics import evaluate, top1_per_fix
from .model_zoo import MODEL_SPECS, SEED
from .train_compare import load_split
from .training import fit_model

OUT = config.RESULTS_DIR / "models"
N_TRAIN_FIXES = 20000   # in-sample scoring on a random sample of training fixes


def load_or_refit(name, train):
    path = config.ARTIFACTS_DIR / "models" / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    print(f"  refitting {name} (model file not kept)")
    model, _ = fit_model(name, train[FEATURES].to_numpy(), train.label.to_numpy(),
                         train.trip_id.to_numpy(), record_history=False)
    return model


def main():
    train = load_split("train", subsample_only=True)
    test = load_split("student_test")
    comp = pd.read_csv(OUT / "model_comparison.csv").set_index("model")
    probs = pd.read_parquet(OUT / "student_test_probabilities.parquet")

    rng = np.random.default_rng(SEED)
    keys = train[["trip_id", "fix"]].drop_duplicates()
    pick = keys.iloc[rng.choice(len(keys), min(N_TRAIN_FIXES, len(keys)), replace=False)]
    sample = train.merge(pick, on=["trip_id", "fix"]).reset_index(drop=True)

    rows, per_trip, per_class = [], {}, []
    models = list(MODEL_SPECS) + ["lightgbm_tuned"]
    for name in models:
        if name == "lightgbm_tuned":
            model = joblib.load(config.ARTIFACTS_DIR / "final_model.joblib")
            tv = pd.read_csv(config.RESULTS_DIR / "tuning" / "tuned_vs_untuned.csv").set_index("variant").loc["tuned"]
            cv_top1, cv_f1, cv_ll = tv.cv_top1, tv.cv_f1_macro, tv.cv_log_loss
            test_prob = model.predict_proba(test[FEATURES].to_numpy())[:, 1]
        else:
            model = load_or_refit(name, train)
            c = comp.loc[name]
            cv_top1, cv_f1, cv_ll = c.cv_top1, c.cv_f1_macro, c.cv_log_loss
            test_prob = probs[name].to_numpy()
        tr = evaluate(sample, model.predict_proba(sample[FEATURES].to_numpy())[:, 1])
        te = evaluate(test, test_prob)
        rows.append(dict(model=name, train_top1=tr["top1_accuracy"], cv_top1=cv_top1, test_top1=te["top1_accuracy"],
                         train_f1_macro=tr["f1_macro"], cv_f1_macro=cv_f1, test_f1_macro=te["f1_macro"],
                         train_log_loss=tr["log_loss"], cv_log_loss=cv_ll, test_log_loss=te["log_loss"],
                         gap_train_minus_cv_top1=tr["top1_accuracy"] - cv_top1))
        m = test.label_valid.to_numpy()
        best = top1_per_fix(test[m], test_prob[m])
        per_trip[name] = best.groupby("trip_id").correct.mean()
        p, r, f, s = precision_recall_fscore_support(test.label.to_numpy()[m], (test_prob[m] >= 0.5).astype(int),
                                                     labels=[0, 1])
        per_class.append(dict(model=name, precision_neg=p[0], recall_neg=r[0], f1_neg=f[0], support_neg=s[0],
                              precision_pos=p[1], recall_pos=r[1], f1_pos=f[1], support_pos=s[1]))
        print(f"{name}: train top1 {tr['top1_accuracy']:.4f} | cv {cv_top1:.4f} | test {te['top1_accuracy']:.4f}", flush=True)

    fit = pd.DataFrame(rows)
    fit.to_csv(OUT / "train_cv_test_scores.csv", index=False)
    pt = pd.DataFrame(per_trip)
    pt.to_csv(OUT / "student_test_top1_per_trip.csv")
    pd.DataFrame(per_class).to_csv(OUT / "student_test_per_class_metrics.csv", index=False)

    final = joblib.load(config.ARTIFACTS_DIR / "final_model.joblib")
    gain = final.booster_.feature_importance(importance_type="gain")
    split = final.booster_.feature_importance(importance_type="split")
    imp = pd.DataFrame({"feature": FEATURES, "gain": gain, "split": split})
    imp["gain_share"] = imp.gain / imp.gain.sum()
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(OUT / "feature_importance_final_model.csv", index=False)

    plots = OUT / "plots"
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(fit))
    for k, (col, lab) in enumerate([("train_top1", "training (in-sample)"), ("cv_top1", "5-fold CV"),
                                    ("test_top1", "student-test")]):
        ax.bar(x + (k - 1) * 0.27, fit[col], 0.27, label=lab)
    ax.set_xticks(x, fit.model, rotation=30)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Training vs cross-validation vs student-test Top-1 (over/under-fitting check)")
    ax.legend()
    fig.tight_layout(); fig.savefig(plots / "train_cv_test_top1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(pt.T.to_numpy(), cmap="viridis", aspect="auto", vmin=pt.min().min(), vmax=1)
    ax.set_xticks(range(len(pt)), pt.index, rotation=45)
    ax.set_yticks(range(len(pt.columns)), pt.columns)
    for i in range(pt.shape[1]):
        for j in range(pt.shape[0]):
            ax.text(j, i, f"{pt.iloc[j, i]:.3f}", ha="center", va="center", fontsize=6.5, color="white")
    fig.colorbar(im, ax=ax, label="Top-1")
    ax.set_title("Student-test Top-1 per trip and model (classifier only)")
    fig.tight_layout(); fig.savefig(plots / "student_test_top1_per_trip.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.feature[::-1], imp.gain_share[::-1])
    ax.set_xlabel("share of total split gain")
    ax.set_title("Feature importance - final tuned LightGBM")
    fig.tight_layout(); fig.savefig(plots / "feature_importance.png", dpi=150); plt.close(fig)
    print(fit.round(4).to_string(index=False))
    print(imp.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
