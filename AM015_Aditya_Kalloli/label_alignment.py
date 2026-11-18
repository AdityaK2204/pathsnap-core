"""Derive one reference arc per GPS fix from the reference route.

The .route file is only an ordered list of arcs; it does not say which fix
belongs to which arc. We align fixes to route positions with a monotone
dynamic programme: fix i is assigned route position j(i) with
j(0) <= j(1) <= ... so the vehicle never moves backwards along the route,
and the sum of point-to-arc distances is minimised. Fixes whose aligned arc
is further than LABEL_MAX_DIST_M away are marked unreliable and excluded from
training and scoring.
"""
import numpy as np

from . import config
from .geo import point_segment

DIST_CAP_M = 300.0   # caps the influence of outlier fixes on the alignment


def align_fixes_to_route(xy, node_xy, arcs, route, chunk=2048):
    route = np.asarray(route)
    A = node_xy[arcs[route, 0]]
    B = node_xy[arcs[route, 1]]
    n, m = len(xy), len(route)

    back = np.empty((n, m), np.int32)
    prev = None
    dist_rows = np.empty((n, m), np.float32)
    for s in range(0, n, chunk):
        d, _ = point_segment(xy[s:s + chunk, None, :], A[None], B[None])
        dist_rows[s:s + chunk] = np.minimum(d, DIST_CAP_M)

    idx = np.arange(m)
    for i in range(n):
        d = dist_rows[i].astype(np.float64)
        if prev is None:
            cost = d
            back[i] = idx
        else:
            # best previous position j' <= j (monotone constraint) via prefix minimum
            pm = np.minimum.accumulate(prev)
            back[i] = np.maximum.accumulate(np.where(prev <= pm, idx, 0))
            cost = d + pm
        prev = cost

    pos = np.empty(n, np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    label_arc = route[pos]
    label_dist = dist_rows[np.arange(n), pos].astype(np.float64)
    return pos, label_arc, label_dist


def make_labels(xy, node_xy, arcs, route, max_dist=config.LABEL_MAX_DIST_M):
    pos, label_arc, label_dist = align_fixes_to_route(xy, node_xy, arcs, route)
    valid = label_dist <= max_dist
    return dict(route_pos=pos, label_arc=label_arc, label_dist=label_dist, label_valid=valid)
