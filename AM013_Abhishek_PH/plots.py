"""Comparison plots from the saved experiment results.

Usage (from the repository root):
    python -m AM013_Abhishek_PH.plots
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from AM015_Aditya_Kalloli import config

IN = config.RESULTS_DIR / "models"
OUT = IN / "plots"


def _bar(ax, names, vals, err=None, fmt="{:.3f}"):
    bars = ax.bar(names, vals, yerr=err, capsize=4, color="#4C72B0")
    offs = np.zeros(len(vals)) if err is None else np.asarray(err)
    for b, v, o in zip(bars, vals, offs):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height() + o),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=30)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(IN / "model_comparison.csv")
    folds = pd.read_csv(IN / "cv_fold_scores.csv")
    names = comp.model.tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    _bar(ax, names, comp.cv_f1_macro, comp.cv_f1_macro_std)
    ax.set_ylim(comp.cv_f1_macro.min() - 0.05, 1)
    ax.set_ylabel("Macro F1 (5-fold GroupKFold)")
    ax.set_title("Cross-validation Macro F1 by model (training trips)")
    fig.tight_layout(); fig.savefig(OUT / "cv_macro_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, comp.test_top1, 0.4, label="Top-1 accuracy")
    ax.bar(x + 0.2, comp.test_f1_macro, 0.4, label="Macro F1")
    ax.set_xticks(x, names, rotation=30)
    ax.set_ylim(min(comp.test_top1.min(), comp.test_f1_macro.min()) - 0.05, 1)
    ax.set_title("Student-testing partition: Top-1 accuracy and Macro F1")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "student_test_top1_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, comp.cv_log_loss, 0.4, label="CV")
    ax.bar(x + 0.2, comp.test_log_loss, 0.4, label="student-test")
    ax.set_xticks(x, names, rotation=30)
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Log loss by model")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "log_loss.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    _bar(ax, names, comp.inference_ms_per_1k_fixes, fmt="{:.1f}")
    ax.set_yscale("log")
    ax.set_ylabel("ms per 1000 GPS fixes (log scale)")
    ax.set_title("Inference time on student-testing candidates")
    fig.tight_layout(); fig.savefig(OUT / "inference_time.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    data = [folds[folds.model == n].top1_accuracy.values for n in names]
    ax.boxplot(data, tick_labels=names)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i), d, s=12, color="k", zorder=3)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Top-1 accuracy per fold")
    ax.set_title("Stability across CV folds")
    fig.tight_layout(); fig.savefig(OUT / "cv_fold_top1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    _bar(ax, names, comp.test_route_continuity)
    ax.set_ylim(comp.test_route_continuity.min() - 0.05, 1)
    ax.set_title("Route continuity of classifier-only predictions (student-test)")
    fig.tight_layout(); fig.savefig(OUT / "route_continuity.png", dpi=150); plt.close(fig)

    cms = json.loads((IN / "confusion_matrices.json").read_text())
    cols = 4
    rows = int(np.ceil(len(cms) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows))
    for ax, (n, cm) in zip(axes.ravel(), cms.items()):
        cm = np.array(cm, float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(n, fontsize=9)
    for ax in axes.ravel()[len(cms):]:
        ax.axis("off")
    fig.suptitle("Candidate-level confusion matrices (student-test, threshold 0.5)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=150); plt.close(fig)

    hists = json.loads((IN / "training_histories.json").read_text())
    for n, h in hists.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(h["train"]) + 1), h["train"], label="train")
        ax.plot(range(1, len(h["validation"]) + 1), h["validation"], label="validation")
        ax.set_xlabel(h.get("unit", "iteration")); ax.set_ylabel("log loss")
        ax.set_title(f"{n}: training history")
        ax.legend()
        fig.tight_layout(); fig.savefig(OUT / f"history_{n}.png", dpi=150); plt.close(fig)
    print("plots written to", OUT)


if __name__ == "__main__":
    main()
