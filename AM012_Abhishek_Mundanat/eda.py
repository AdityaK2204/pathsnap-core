"""Exploratory data analysis of the raw dataset (training + student-testing partitions).

Usage (from the repository root):
    python -m AM012_Abhishek_Mundanat.eda

Outputs (results/eda/)
    dataset_inventory.csv      per-trip files, sizes and counts
    trip_quality.csv           per-trip sampling, gap, speed and network statistics
    partition_summary.csv      aggregate statistics per partition
    missing_invalid_values.csv missing / invalid value counts per column and partition
    data_quality_issues.csv    trips flagged for anomalies
    *.png                      figures
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from AM015_Aditya_Kalloli import config
from AM015_Aditya_Kalloli.data_io import list_trips, load_route, load_trip
from AM015_Aditya_Kalloli.geo import LocalProjection

OUT = config.RESULTS_DIR / "eda"
SPLITS = {"train": config.TRAIN_DIR, "student_test": config.STUDENT_TEST_DIR}
SPEED_LIMIT_MS = 70.0      # ~250 km/h, anything above is treated as a GPS jump


def trip_stats(split, d, tid):
    folder = d / tid
    sizes = {p.suffix[1:]: p.stat().st_size for p in folder.iterdir()}
    trip = load_trip(d, tid)
    route = load_route(d, tid)
    tr = trip.track
    raw = tr.to_numpy()
    t = tr.t.to_numpy()
    dt = np.diff(t)
    proj = LocalProjection.for_track(tr.lon, tr.lat)
    xy = proj.forward(tr.lon, tr.lat)
    step = np.hypot(*np.diff(xy, axis=0).T)
    speed = step / np.where(dt > 0, dt, np.nan)

    n_nodes, n_arcs = len(trip.nodes), len(trip.arcs)
    u, v = trip.arcs[:, 0], trip.arcs[:, 1]
    g = coo_matrix((np.ones(n_arcs), (u, v)), shape=(n_nodes, n_nodes))
    n_cc, lab = connected_components(g, directed=False)
    n_scc, _ = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(np.r_[u, v], minlength=n_nodes)
    keys = u.astype(np.int64) * n_nodes + v
    rev = v.astype(np.int64) * n_nodes + u

    route_ok = route is not None and len(route) and (route >= 0).all() and (route < n_arcs).all()
    breaks = int((trip.arcs[route[:-1], 1] != trip.arcs[route[1:], 0]).sum()) if route_ok else np.nan

    inv = dict(
        nan_values=int(np.isnan(raw).sum()),
        lon_out_of_range=int((~tr.lon.between(-180, 180)).sum()),
        lat_out_of_range=int((~tr.lat.between(-90, 90)).sum()),
        non_increasing_timestamps=int((dt <= 0).sum()),
        duplicate_positions=int((step == 0).sum()),
        implausible_speed_steps=int(np.nansum(speed > SPEED_LIMIT_MS)),
    )
    return dict(
        split=split, trip_id=tid, has_track="track" in sizes, has_nodes="nodes" in sizes,
        has_arcs="arcs" in sizes, has_route="route" in sizes,
        bytes_track=sizes.get("track", 0), bytes_nodes=sizes.get("nodes", 0),
        bytes_arcs=sizes.get("arcs", 0), bytes_route=sizes.get("route", 0),
        n_fixes=len(tr), duration_s=float(t[-1] - t[0]), t_start=float(t[0]),
        dt_median=float(np.median(dt)), dt_max=float(dt.max()), pct_dt_1s=float(100 * np.mean(dt == 1)),
        gaps_over_10s=int((dt > 10).sum()), gaps_over_60s=int((dt > 60).sum()),
        path_km=float(step.sum() / 1000), speed_median=float(np.nanmedian(speed)),
        speed_p99=float(np.nanpercentile(speed, 99)), speed_max=float(np.nanmax(speed)),
        lon_min=float(tr.lon.min()), lon_max=float(tr.lon.max()),
        lat_min=float(tr.lat.min()), lat_max=float(tr.lat.max()),
        extent_x_km=float(np.ptp(xy[:, 0]) / 1000), extent_y_km=float(np.ptp(xy[:, 1]) / 1000),
        n_nodes=n_nodes, n_arcs=n_arcs, self_loops=int((u == v).sum()),
        duplicate_arcs=int(n_arcs - len(np.unique(keys))),
        two_way_arc_frac=float(np.isin(rev, keys).mean()),
        weak_components=int(n_cc), strong_components=int(n_scc),
        largest_component_frac=float(np.bincount(lab).max() / n_nodes),
        isolated_nodes=int((deg == 0).sum()), mean_degree=float(2 * n_arcs / n_nodes),
        route_len=0 if route is None else len(route), route_valid_ids=bool(route_ok),
        route_breaks=breaks, route_repeated_arcs=int(len(route) - len(np.unique(route))) if route_ok else np.nan,
        **inv,
    ), xy, t, speed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, samples = [], {}
    all_dt, all_speed = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    for split, d in SPLITS.items():
        for tid in list_trips(d):
            r, xy, t, sp = trip_stats(split, d, tid)
            rows.append(r)
            all_dt[split].append(np.diff(t))
            all_speed[split].append(sp[np.isfinite(sp)])
            samples[(split, tid)] = xy
    df = pd.DataFrame(rows)

    inv_cols = ["split", "trip_id", "has_track", "has_nodes", "has_arcs", "has_route", "bytes_track",
                "bytes_nodes", "bytes_arcs", "bytes_route", "n_fixes", "n_nodes", "n_arcs", "route_len"]
    df[inv_cols].to_csv(OUT / "dataset_inventory.csv", index=False)
    df.to_csv(OUT / "trip_quality.csv", index=False)

    agg = df.groupby("split").agg(
        trips=("trip_id", "count"), fixes_total=("n_fixes", "sum"), fixes_median=("n_fixes", "median"),
        fixes_min=("n_fixes", "min"), fixes_max=("n_fixes", "max"), duration_median_s=("duration_s", "median"),
        path_km_median=("path_km", "median"), pct_dt_1s_mean=("pct_dt_1s", "mean"),
        trips_with_gap_over_60s=("gaps_over_60s", lambda s: int((s > 0).sum())),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
        two_way_frac_mean=("two_way_arc_frac", "mean"), weak_components_median=("weak_components", "median"),
        route_len_median=("route_len", "median"), trips_with_route_breaks=("route_breaks", lambda s: int((s > 0).sum())),
    )
    agg.T.to_csv(OUT / "partition_summary.csv")

    inv_names = ["nan_values", "lon_out_of_range", "lat_out_of_range", "non_increasing_timestamps",
                 "duplicate_positions", "implausible_speed_steps", "self_loops", "duplicate_arcs", "isolated_nodes"]
    miss = df.groupby("split")[inv_names].sum().T
    miss["trips_affected_train"] = (df[df.split == "train"][inv_names] > 0).sum()
    miss["trips_affected_student_test"] = (df[df.split == "student_test"][inv_names] > 0).sum()
    miss.to_csv(OUT / "missing_invalid_values.csv")

    issues = []
    for _, r in df.iterrows():
        if r.gaps_over_60s:
            issues.append((r.split, r.trip_id, "time gap > 60 s", f"{r.gaps_over_60s} gaps, max {r.dt_max:.0f} s"))
        if r.implausible_speed_steps:
            issues.append((r.split, r.trip_id, f"speed > {SPEED_LIMIT_MS} m/s", f"{r.implausible_speed_steps} steps, max {r.speed_max:.0f} m/s"))
        if r.route_breaks and r.route_breaks > 0:
            issues.append((r.split, r.trip_id, "reference route not contiguous", f"{int(r.route_breaks)} breaks"))
        if r.largest_component_frac < 0.9:
            issues.append((r.split, r.trip_id, "fragmented road network", f"largest component {r.largest_component_frac:.0%}"))
    lq = config.RESULTS_DIR / "preprocessing" / "label_quality.csv"
    if lq.exists():
        q = pd.read_csv(lq, dtype={"trip_id": str})
        for _, r in q[q.frac_label_valid < 0.9].iterrows():
            issues.append((r.split, r.trip_id, "GPS far from reference route",
                           f"median {r.label_dist_median:.0f} m, {r.frac_label_valid:.0%} fixes within 50 m"))
    pd.DataFrame(issues, columns=["split", "trip_id", "issue", "detail"]).to_csv(OUT / "data_quality_issues.csv", index=False)

    _plots(df, all_dt, all_speed, samples)
    print(agg.T.to_string())
    print(miss.to_string())


def _plots(df, all_dt, all_speed, samples):
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SPLITS:
        ax.hist(df[df.split == s].n_fixes, bins=np.logspace(2, 4.5, 30), alpha=0.6, label=s)
    ax.set_xscale("log"); ax.set_xlabel("GPS fixes per trip"); ax.set_ylabel("trips"); ax.legend()
    ax.set_title("Trip length (train + student-test)")
    fig.tight_layout(); fig.savefig(OUT / "fixes_per_trip.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    dt = np.concatenate(all_dt["train"])
    vals, counts = np.unique(np.clip(dt, 0, 31), return_counts=True)
    ax.bar(vals, counts / counts.sum())
    ax.set_yscale("log"); ax.set_xlabel("sampling interval [s] (31 = 31 s or more)"); ax.set_ylabel("share of steps")
    ax.set_title("Sampling intervals (training partition)")
    fig.tight_layout(); fig.savefig(OUT / "sampling_intervals.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sp = np.concatenate(all_speed["train"])
    ax.hist(np.clip(sp, 0, 100), bins=100)
    ax.axvline(SPEED_LIMIT_MS, color="r", ls="--", label=f"{SPEED_LIMIT_MS} m/s")
    ax.set_yscale("log"); ax.set_xlabel("speed between consecutive fixes [m/s]"); ax.legend()
    ax.set_title("Speed distribution (training partition)")
    fig.tight_layout(); fig.savefig(OUT / "speed_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for s, mk in (("train", "o"), ("student_test", "^")):
        d = df[df.split == s]
        ax.scatter((d.lon_min + d.lon_max) / 2, (d.lat_min + d.lat_max) / 2, marker=mk, label=s, s=30)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.grid(alpha=0.3)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
    ax.set_title("Trip locations (each trip has its own local road network)")
    fig.tight_layout(); fig.savefig(OUT / "trip_locations.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(df.n_nodes, df.n_arcs, c=df.split.eq("train").map({True: "C0", False: "C1"}), s=15)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("nodes"); axes[0].set_ylabel("arcs"); axes[0].set_title("Road-network size per trip")
    axes[1].hist(df.largest_component_frac, bins=20)
    axes[1].set_xlabel("share of nodes in largest weak component"); axes[1].set_title("Network connectivity")
    fig.tight_layout(); fig.savefig(OUT / "network_size_connectivity.png", dpi=150); plt.close(fig)

    picks = [("train", "00000001"), ("train", "00000054"), ("train", "00000081"), ("student_test", "00000056")]
    fig, axes = plt.subplots(2, len(picks), figsize=(4.2 * len(picks), 8.4))
    for col, key in enumerate(picks):
        split, tid = key
        d = config.TRAIN_DIR if split == "train" else config.STUDENT_TEST_DIR
        trip = load_trip(d, tid)
        route = load_route(d, tid)
        proj = LocalProjection.for_track(trip.track.lon, trip.track.lat)
        nxy = proj.forward(trip.nodes[:, 0], trip.nodes[:, 1])
        xy = samples[key]
        A, B = nxy[trip.arcs[:, 0]], nxy[trip.arcs[:, 1]]
        rs = np.stack([nxy[trip.arcs[route, 0]], nxy[trip.arcs[route, 1]]], 1)
        cx, cy = xy[len(xy) // 3]
        windows = [(xy[:, 0].min() - 300, xy[:, 0].max() + 300, xy[:, 1].min() - 300, xy[:, 1].max() + 300),
                   (cx - 300, cx + 300, cy - 300, cy + 300)]
        for row, (x0, x1, y0, y1) in enumerate(windows):
            ax = axes[row, col]
            inside = (A[:, 0] > x0) & (A[:, 0] < x1) & (A[:, 1] > y0) & (A[:, 1] < y1)
            ax.add_collection(LineCollection(np.stack([A[inside], B[inside]], 1), colors="0.7", linewidths=0.6))
            ax.add_collection(LineCollection(rs, colors="#1a9850", linewidths=3 if row else 2.5, alpha=0.7))
            ax.plot(xy[:, 0], xy[:, 1], ".", ms=2.5 if row else 1.5, color="#1f4fff")
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{split} {tid}" + (" (600 m zoom)" if row else ""), fontsize=9)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.suptitle("GPS fixes (blue), reference route (green), road network (grey)")
    fig.tight_layout(); fig.savefig(OUT / "trajectory_examples.png", dpi=150); plt.close(fig)

    lq = config.CACHE_DIR / "fixes_train.parquet"
    if lq.exists():
        f = pd.read_parquet(lq, columns=["label_dist"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.clip(f.label_dist, 0, 100), bins=100)
        ax.axvline(config.LABEL_MAX_DIST_M, color="r", ls="--", label="label cut-off")
        ax.set_yscale("log"); ax.set_xlabel("distance GPS fix -> aligned reference arc [m] (clipped at 100)")
        ax.set_title("GPS noise relative to the reference route (training)"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / "gps_to_route_distance.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
