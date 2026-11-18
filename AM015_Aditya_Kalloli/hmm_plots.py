"""Plots for the HMM decoder tuning.

Usage (from the repository root, after tune_decoder):
    python -m AM015_Aditya_Kalloli.hmm_plots
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

OUT = config.RESULTS_DIR / "hmm"


def main():
    g = pd.read_csv(OUT / "decoder_grid_summary.csv")
    piv = g.pivot(index="emission_weight", columns="beta", values="top1_pooled")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{b:g}" for b in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{w:g}" for w in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iloc[i, j]:.4f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xlabel("beta (transition scale, m)"); ax.set_ylabel("emission weight")
    ax.set_title("HMM grid search: pooled Top-1 on training trips (out-of-fold probabilities)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(OUT / "decoder_grid.png", dpi=150); plt.close(fig)

    p = g.groupby("beta_x_weight").top1_pooled.max().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(p.index, p.values, "o-")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("beta x emission weight"); ax.set_ylabel("pooled Top-1 (training, OOF)")
    ax.set_title("Only the product beta x weight changes the Viterbi path")
    fig.tight_layout(); fig.savefig(OUT / "decoder_product_curve.png", dpi=150); plt.close(fig)

    per = pd.read_csv(OUT / "hmm_vs_classifier_per_trip.csv", dtype={"trip_id": str})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(per.top1_accuracy_classifier, per.top1_accuracy_hmm, s=40)
    for _, r in per.iterrows():
        ax.annotate(r.trip_id[-2:], (r.top1_accuracy_classifier, r.top1_accuracy_hmm), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    lo = min(per.top1_accuracy_classifier.min(), per.top1_accuracy_hmm.min()) - 0.01
    ax.plot([lo, 1], [lo, 1], "k--", lw=1, label="no change")
    ax.set_xlabel("classifier-only Top-1"); ax.set_ylabel("classifier + HMM Top-1")
    ax.set_title("Per-trip effect of HMM decoding (student-test)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "hmm_gain_scatter.png", dpi=150); plt.close(fig)
    print("plots written to", OUT)


if __name__ == "__main__":
    main()
