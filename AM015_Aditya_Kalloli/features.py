"""Feature engineering for (GPS fix, candidate arc) pairs."""
import numpy as np
import pandas as pd

from .geo import angle_diff, bearing_xy, point_segment

FEATURES = [
    "dist", "dist_minus_min", "dist_rank", "n_cands",
    "t_proj", "at_endpoint", "log_arc_len",
    "head_diff_short", "head_diff_long", "cos_head_short", "heading_valid",
    "along_disp", "wdist_mean", "wdist_max", "wdist_rank",
    "has_twin", "out_deg_end", "in_deg_start",
    "speed", "dt_prev", "turn_angle",
]

HEADING_MIN_DISP_M = 4.0
WINDOW = 3


def _shift_idx(n, k):
    return np.clip(np.arange(n) + k, 0, n - 1)


def fix_kinematics(xy, t):
    """Per-fix motion descriptors computed only from the GPS track."""
    n = len(xy)
    out = {}
    dt = np.diff(t)
    out["dt_prev"] = np.r_[dt[:1] if n > 1 else [1.0], dt]
    a, b = _shift_idx(n, -1), _shift_idx(n, 1)
    span = np.maximum(t[b] - t[a], 1e-6)
    out["speed"] = np.hypot(*(xy[b] - xy[a]).T) / span

    for name, k in (("short", 2), ("long", 5)):
        a, b = _shift_idx(n, -k), _shift_idx(n, k)
        d = xy[b] - xy[a]
        out[f"disp_{name}"] = d
        out[f"disp_len_{name}"] = np.hypot(*d.T)
        out[f"bearing_{name}"] = bearing_xy(*d.T)

    a, b = _shift_idx(n, -2), _shift_idx(n, 2)
    b_in = bearing_xy(*(xy - xy[a]).T)
    b_out = bearing_xy(*(xy[b] - xy).T)
    moving = (np.hypot(*(xy - xy[a]).T) > 2) & (np.hypot(*(xy[b] - xy).T) > 2)
    out["turn_angle"] = np.where(moving, angle_diff(b_in, b_out), 0.0)
    return out


def build_features(cands, xy, t, net):
    """Add feature columns to the candidate frame (in place copy)."""
    df = cands.copy()
    n = len(xy)
    fix = df.fix.to_numpy()
    arc = df.arc_id.to_numpy()
    kin = fix_kinematics(xy, t)

    grp = df.groupby("fix")
    df["dist_minus_min"] = df.dist - grp.dist.transform("min")
    df["n_cands"] = grp.dist.transform("size")
    df["at_endpoint"] = ((df.t_proj <= 1e-3) | (df.t_proj >= 1 - 1e-3)).astype(np.int8)
    df["log_arc_len"] = np.log1p(net.arc_len[arc])

    arc_b = net.arc_bearing[arc]
    valid_s = kin["disp_len_short"][fix] >= HEADING_MIN_DISP_M
    valid_l = kin["disp_len_long"][fix] >= HEADING_MIN_DISP_M
    hs = angle_diff(kin["bearing_short"][fix], arc_b)
    hl = angle_diff(kin["bearing_long"][fix], arc_b)
    # without a usable heading, use the neutral value 90 degrees
    df["head_diff_short"] = np.where(valid_s, hs, 90.0)
    df["head_diff_long"] = np.where(valid_l, hl, 90.0)
    df["cos_head_short"] = np.where(valid_s, np.cos(np.radians(hs)), 0.0)
    df["heading_valid"] = valid_s.astype(np.int8)

    A, B = net.arc_endpoints_xy(arc)
    u = (B - A) / np.maximum(net.arc_len[arc], 1e-6)[:, None]
    df["along_disp"] = (kin["disp_short"][fix] * u).sum(1)

    # distances of neighbouring fixes to the same arc (local consistency)
    wd = []
    for k in range(-WINDOW, WINDOW + 1):
        if k == 0:
            continue
        P = xy[_shift_idx(n, k)[fix]]
        d, _ = point_segment(P, A, B)
        wd.append(d)
    wd = np.column_stack(wd + [df.dist.to_numpy()])
    df["wdist_mean"] = wd.mean(1)
    df["wdist_max"] = wd.max(1)
    df["wdist_rank"] = df.groupby("fix").wdist_mean.rank(method="first") - 1

    df["has_twin"] = net.has_twin[arc].astype(np.int8)
    df["out_deg_end"] = net.out_deg[net.arcs[arc, 1]]
    df["in_deg_start"] = net.in_deg[net.arcs[arc, 0]]
    df["speed"] = np.minimum(kin["speed"][fix], 80.0)
    df["dt_prev"] = np.minimum(kin["dt_prev"][fix], 600.0)
    df["turn_angle"] = kin["turn_angle"][fix]

    df[FEATURES] = df[FEATURES].astype(np.float32)
    return df
