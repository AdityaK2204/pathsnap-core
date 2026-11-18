"""Build candidate/feature tables for the training and student-testing partitions.

Usage (from the repository root):
    python -m AM015_Aditya_Kalloli.build_dataset

Outputs
    data_cache/candidates_<split>.parquet   one row per (fix, candidate arc)
    data_cache/fixes_<split>.parquet        one row per GPS fix with its derived label
    results/preprocessing/label_quality.csv per-trip label / candidate statistics
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config
from .data_io import list_trips, load_route, load_trip
from .label_alignment import make_labels
from .preprocess import prepare_trip

SPLITS = {"train": config.TRAIN_DIR, "student_test": config.STUDENT_TEST_DIR}


def process(split, split_dir, trip_id):
    t0 = time.perf_counter()
    trip = load_trip(split_dir, trip_id)
    p = prepare_trip(trip)
    route = load_route(split_dir, trip_id)
    lab = make_labels(p.xy, p.net.node_xy, trip.arcs, route)

    fixes = pd.DataFrame({
        "trip_id": trip_id, "fix": np.arange(len(p.xy)), "raw_idx": p.track.raw_idx.to_numpy(),
        "t": p.t, "lon": p.track.lon.to_numpy(), "lat": p.track.lat.to_numpy(),
        "label_arc": lab["label_arc"], "label_dist": lab["label_dist"],
        "label_valid": lab["label_valid"], "route_pos": lab["route_pos"],
    })
    c = p.cands
    c.insert(0, "trip_id", trip_id)
    c["label"] = (c.arc_id.to_numpy() == lab["label_arc"][c.fix.to_numpy()]).astype(np.int8)
    c["label_valid"] = lab["label_valid"][c.fix.to_numpy()]

    hit = c.groupby("fix").label.max().reindex(fixes.fix, fill_value=0).to_numpy().astype(bool)
    fixes["label_in_cands"] = hit

    # evenly spaced subsample of labelled fixes used for model fitting
    valid_idx = np.flatnonzero(lab["label_valid"])
    if len(valid_idx) > config.MAX_FIXES_PER_TRIP:
        valid_idx = valid_idx[np.linspace(0, len(valid_idx) - 1, config.MAX_FIXES_PER_TRIP).astype(int)]
    sub = np.zeros(len(fixes), bool)
    sub[valid_idx] = True
    fixes["in_subsample"] = sub
    c["in_subsample"] = sub[c.fix.to_numpy()]

    v = lab["label_valid"]
    stats = dict(
        split=split, trip_id=trip_id, n_fixes=len(fixes), n_route_arcs=len(route),
        n_arcs_total=len(trip.arcs), n_arcs_cropped=len(p.net.arc_ids),
        label_dist_median=float(np.median(lab["label_dist"])),
        label_dist_p90=float(np.percentile(lab["label_dist"], 90)),
        frac_label_valid=float(v.mean()),
        route_arcs_covered=float(len(np.unique(lab["route_pos"])) / len(route)),
        cands_per_fix=float(len(c) / len(fixes)),
        candidate_miss_rate=float(1 - hit[v].mean()) if v.any() else np.nan,
        nearest_is_label=float(c[c.dist_rank == 0].set_index("fix").label.reindex(np.flatnonzero(v)).mean()) if v.any() else np.nan,
        seconds=time.perf_counter() - t0,
    )
    return fixes, c, stats


def main():
    config.ensure_dirs()
    out_dir = config.RESULTS_DIR / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for split, split_dir in SPLITS.items():
        ids = list_trips(split_dir)
        print(f"[{split}] {len(ids)} trips from {split_dir}")
        res = Parallel(n_jobs=8, verbose=5)(delayed(process)(split, split_dir, t) for t in ids)
        fixes = pd.concat([r[0] for r in res], ignore_index=True)
        cands = pd.concat([r[1] for r in res], ignore_index=True)
        all_stats += [r[2] for r in res]
        fixes.to_parquet(config.CACHE_DIR / f"fixes_{split}.parquet", index=False)
        cands.to_parquet(config.CACHE_DIR / f"candidates_{split}.parquet", index=False)
        print(f"[{split}] fixes={len(fixes):,} candidate rows={len(cands):,} "
              f"positives={int(cands.label.sum()):,}")
    stats = pd.DataFrame(all_stats)
    stats.to_csv(out_dir / "label_quality.csv", index=False)
    print(stats.groupby("split")[["frac_label_valid", "candidate_miss_rate", "nearest_is_label",
                                  "cands_per_fix", "seconds"]].describe().T.round(3))


if __name__ == "__main__":
    main()
