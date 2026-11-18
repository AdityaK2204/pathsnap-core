"""Candidate road segments for every GPS fix."""
import numpy as np
import pandas as pd

from . import config
from .geo import point_segment


def generate_candidates(xy, net, radius=config.CAND_RADIUS_M, max_k=config.CAND_MAX_K,
                        fallback_k=config.CAND_FALLBACK_K):
    """Return a frame with one row per (fix, candidate arc).

    Columns: fix, arc_id, dist, t_proj. Candidates are the max_k closest arcs
    within `radius`; a fix with nothing in range gets its fallback_k nearest arcs
    so the decoder always has at least one state.
    """
    near = net.arcs_near(xy, radius)
    counts = np.array([len(a) for a in near])
    empty = np.flatnonzero(counts == 0)
    if len(empty):
        fb = net.nearest_arcs(xy[empty], fallback_k)
        for i, arcs in zip(empty, fb):
            near[i] = arcs
        counts = np.array([len(a) for a in near])

    fix = np.repeat(np.arange(len(xy)), counts)
    arc = np.concatenate(near).astype(np.int64)
    A, B = net.arc_endpoints_xy(arc)
    dist, t = point_segment(xy[fix], A, B)

    df = pd.DataFrame({"fix": fix, "arc_id": arc, "dist": dist, "t_proj": t})
    df["in_radius"] = df.dist <= radius
    df = df.sort_values(["fix", "dist"], kind="stable")
    df["dist_rank"] = df.groupby("fix").cumcount()
    # keep top-k inside the radius; for fixes with nothing inside, keep the nearest fallback_k
    any_in = df.groupby("fix").in_radius.transform("any")
    keep = (df.in_radius & (df.dist_rank < max_k)) | (~any_in & (df.dist_rank < fallback_k))
    df = df[keep].drop(columns="in_radius").reset_index(drop=True)
    return df
