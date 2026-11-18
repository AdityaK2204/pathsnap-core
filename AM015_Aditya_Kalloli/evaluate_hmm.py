"""Classifier-only vs classifier + HMM/Viterbi on the student-testing trips.

Usage (from the repository root):
    python -m AM015_Aditya_Kalloli.evaluate_hmm

Outputs (results/hmm/)
    hmm_vs_classifier_per_trip.csv, hmm_vs_classifier_summary.csv,
    predictions/<trip>.json (per-fix predictions used by the map scripts),
    hmm_vs_classifier_top1.png, hmm_vs_classifier_continuity.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data_io import list_trips
from .pipeline import MapMatcher, score_against_reference

OUT = config.RESULTS_DIR / "hmm"


def main(split_dir=config.STUDENT_TEST_DIR):
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    mm = MapMatcher()
    print("decoder params:", mm.params)
    rows = []
    for tid in list_trips(split_dir):
        r = mm.match(split_dir, tid)
        s = score_against_reference(r, split_dir, tid)
        rows.append(dict(trip_id=tid, **s, **{k: round(v, 3) for k, v in r["timings"].items()}))
        tr = r["prepared"].track
        (OUT / "predictions" / f"{tid}.json").write_text(json.dumps(dict(
            trip_id=tid, raw_idx=tr.raw_idx.tolist(),
            classifier_arcs=r["clf_arcs"].tolist(), hmm_arcs=r["hmm_arcs"].tolist(),
            path_pieces=[list(map(int, p)) for p in r["path_pieces"]])))
        print(f"{tid}: clf={s['top1_accuracy_classifier']:.4f} hmm={s['top1_accuracy_hmm']:.4f} "
              f"cont {s['route_continuity_classifier']:.3f}->{s['route_continuity_hmm']:.3f} "
              f"routeF1={s['route_f1']:.3f} breaks={s['hmm_breaks']}", flush=True)

    df = pd.DataFrame(rows)
    df["top1_gain"] = df.top1_accuracy_hmm - df.top1_accuracy_classifier
    df.to_csv(OUT / "hmm_vs_classifier_per_trip.csv", index=False)

    w = df.n_labelled_fixes
    summary = pd.DataFrame([
        dict(decoder="classifier_only",
             top1_pooled=np.average(df.top1_accuracy_classifier, weights=w),
             top1_mean_over_trips=df.top1_accuracy_classifier.mean(),
             route_continuity=df.route_continuity_classifier.mean()),
        dict(decoder="classifier+hmm",
             top1_pooled=np.average(df.top1_accuracy_hmm, weights=w),
             top1_mean_over_trips=df.top1_accuracy_hmm.mean(),
             route_continuity=df.route_continuity_hmm.mean(),
             reachable_continuity=df.reachable_continuity_hmm.mean(),
             route_precision=df.route_precision.mean(), route_recall=df.route_recall.mean(),
             route_f1=df.route_f1.mean(), trips_improved=int((df.top1_gain > 0).sum()),
             trips_worse=int((df.top1_gain < 0).sum())),
    ])
    summary.to_csv(OUT / "hmm_vs_classifier_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, df.top1_accuracy_classifier, 0.4, label="classifier only")
    ax.bar(x + 0.2, df.top1_accuracy_hmm, 0.4, label="classifier + HMM/Viterbi")
    ax.set_xticks(x, df.trip_id, rotation=45)
    ax.set_ylim(max(0, df[["top1_accuracy_classifier", "top1_accuracy_hmm"]].min().min() - 0.05), 1)
    ax.set_ylabel("observation-level Top-1 accuracy")
    ax.set_title("Student-testing trips: effect of HMM decoding")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "hmm_vs_classifier_top1.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, df.route_continuity_classifier, 0.4, label="classifier only")
    ax.bar(x + 0.2, df.route_continuity_hmm, 0.4, label="classifier + HMM/Viterbi")
    ax.set_xticks(x, df.trip_id, rotation=45)
    ax.set_ylim(max(0, df[["route_continuity_classifier", "route_continuity_hmm"]].min().min() - 0.05), 1)
    ax.set_ylabel("route continuity (adjacent-arc rate)")
    ax.set_title("Student-testing trips: route continuity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "hmm_vs_classifier_continuity.png", dpi=150)


if __name__ == "__main__":
    main()
