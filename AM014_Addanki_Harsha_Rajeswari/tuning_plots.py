"""Plots for the Bayesian optimisation run.

Usage (from the repository root):
    python -m AM014_Addanki_Harsha_Rajeswari.tuning_plots
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from AM015_Aditya_Kalloli import config

IN = config.RESULTS_DIR / "tuning"


def main():
    best = json.loads((IN / "best_params.json").read_text())
    study = optuna.load_study(study_name=f"pathsnap_{best['model']}",
                              storage=f"sqlite:///{(IN / 'optuna_study.db').as_posix()}")
    tr = pd.read_csv(IN / "optuna_trials.csv")
    done = tr[tr.state == "COMPLETE"].sort_values("number")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(done.number, done.value, s=18, label="completed trial")
    ax.plot(done.number, done.value.cummax(), color="C3", label="best so far")
    pr = tr[tr.state == "PRUNED"]
    if len(pr):
        ax.scatter(pr.number, np.full(len(pr), done.value.min()), marker="x", color="0.5", label="pruned (position only)")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 accuracy (objective)")
    ax.set_title(f"Bayesian optimisation history ({best['model']}, TPE)")
    ax.legend()
    fig.tight_layout(); fig.savefig(IN / "optimisation_history.png", dpi=150); plt.close(fig)

    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1])
        ax.set_xlabel("importance (fANOVA)"); ax.set_title("Hyperparameter importance")
        fig.tight_layout(); fig.savefig(IN / "param_importance.png", dpi=150); plt.close(fig)
        pd.Series(imp).to_csv(IN / "param_importance.csv", header=["importance"])
    except Exception as e:  # noqa: BLE001
        print("importance skipped:", e)

    params = [c for c in done.columns if c.startswith("params_")]
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), params):
        ax.scatter(done[c], done.value, s=12)
        ax.set_xlabel(c.replace("params_", ""))
        if done[c].min() > 0 and done[c].max() / max(done[c].min(), 1e-12) > 50:
            ax.set_xscale("log")
    np.atleast_1d(axes)[0].set_ylabel("objective")
    fig.suptitle("Objective vs each hyperparameter")
    fig.tight_layout(); fig.savefig(IN / "param_slices.png", dpi=150); plt.close(fig)

    cmp = pd.read_csv(IN / "tuned_vs_untuned.csv").set_index("variant")
    cols = ["cv_f1_macro", "cv_top1", "test_top1", "test_f1_macro"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(cols))
    for k, v in enumerate(cmp.index):
        axes[0].bar(x + (k - 0.5) * 0.4, cmp.loc[v, cols], 0.4, label=v)
    axes[0].set_xticks(x, cols, rotation=20)
    axes[0].set_ylim(cmp[cols].min().min() - 0.02, 1)
    axes[0].legend(); axes[0].set_title("Accuracy / F1")
    for k, v in enumerate(cmp.index):
        axes[1].bar(np.arange(2) + (k - 0.5) * 0.4, cmp.loc[v, ["cv_log_loss", "test_log_loss"]], 0.4, label=v)
    axes[1].set_xticks(range(2), ["cv_log_loss", "test_log_loss"]); axes[1].set_title("Log loss"); axes[1].legend()
    axes[2].bar(cmp.index, cmp.inference_ms_per_1k_fixes)
    axes[2].set_title("Inference ms per 1000 fixes")
    fig.suptitle(f"Untuned vs tuned {best['model']}")
    fig.tight_layout(); fig.savefig(IN / "tuned_vs_untuned.png", dpi=150); plt.close(fig)
    print("plots written to", IN)


if __name__ == "__main__":
    main()
