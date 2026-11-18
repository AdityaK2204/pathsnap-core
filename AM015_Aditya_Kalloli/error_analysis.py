"""Categorise the remaining errors of classifier-only and HMM predictions on student-test.

  reverse_twin : predicted arc is the same road in the opposite direction
  adjacent     : predicted arc shares a node with the reference arc (boundary / off-by-one arc)
  other        : any other arc (parallel road, wrong branch, ...)

Usage (from the repository root, after evaluate_hmm):
    python -m AM015_Aditya_Kalloli.error_analysis
"""
import json

import numpy as np
import pandas as pd

from . import config
from .data_io import load_trip

OUT = config.RESULTS_DIR / "hmm"


def categorise(pred, label, arcs):
    pu, pv = arcs[pred, 0], arcs[pred, 1]
    lu, lv = arcs[label, 0], arcs[label, 1]
    twin = (pu == lv) & (pv == lu)
    adjacent = ~twin & ((pu == lu) | (pu == lv) | (pv == lu) | (pv == lv))
    return np.where(pred == label, "correct", np.where(twin, "reverse_twin", np.where(adjacent, "adjacent", "other")))


def main():
    fixes = pd.read_parquet(config.CACHE_DIR / "fixes_student_test.parquet")
    rows = []
    for p in sorted((OUT / "predictions").glob("*.json")):
        pr = json.loads(p.read_text())
        tid = pr["trip_id"]
        arcs = load_trip(config.STUDENT_TEST_DIR, tid).arcs
        f = fixes[fixes.trip_id == tid].sort_values("fix")
        v = f.label_valid.to_numpy()
        lab = f.label_arc.to_numpy()[v]
        for dec, key in (("classifier_only", "classifier_arcs"), ("classifier+hmm", "hmm_arcs")):
            pred = np.asarray(pr[key])[v]
            cats = pd.Series(categorise(pred, lab, arcs)).value_counts()
            rows.append(dict(trip_id=tid, decoder=dec, n=int(v.sum()), **cats.to_dict()))
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(OUT / "error_categories_per_trip.csv", index=False)
    tot = df.groupby("decoder")[["n", "correct", "reverse_twin", "adjacent", "other"]].sum()
    for c in ["correct", "reverse_twin", "adjacent", "other"]:
        tot[c + "_pct"] = 100 * tot[c] / tot.n
    tot.to_csv(OUT / "error_categories_summary.csv")
    print(tot.round(2).T)
    print(df[df.decoder == "classifier+hmm"].to_string(index=False))


if __name__ == "__main__":
    main()
