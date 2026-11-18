"""Route-level evaluation measures."""
import numpy as np


def route_continuity(pred_arcs, arcs):
    """Share of consecutive fix pairs whose predicted arcs are the same arc or
    directly connected (end node of the first == start node of the second)."""
    p = np.asarray(pred_arcs)
    a, b = p[:-1], p[1:]
    ok = (a >= 0) & (b >= 0)
    if not ok.any():
        return np.nan
    a, b = a[ok], b[ok]
    cont = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(cont.mean())


def reachable_continuity(pred_arcs, net, xy, slack=2.0, extra=50.0):
    """Share of consecutive pairs where the second arc is reachable from the first
    through the directed network within slack * straight-line distance + extra."""
    p = np.asarray(pred_arcs)
    good = tot = 0
    for i in range(1, len(p)):
        a, b = int(p[i - 1]), int(p[i])
        if a < 0 or b < 0:
            continue
        tot += 1
        if a == b or net.arcs[a, 1] == net.arcs[b, 0]:
            good += 1
            continue
        lim = slack * float(np.hypot(*(xy[i] - xy[i - 1]))) + extra + net.arc_len[b]
        if net.arcs[b, 0] in net.dist_from(int(net.arcs[a, 1]), lim):
            good += 1
    return good / tot if tot else np.nan


def arc_set_scores(pred_path, route, arc_len):
    """Length-weighted precision / recall / F1 of the set of traversed arcs."""
    P, R = set(map(int, pred_path)), set(map(int, route))
    if not P or not R:
        return dict(route_precision=np.nan, route_recall=np.nan, route_f1=np.nan)
    inter = float(sum(arc_len[a] for a in P & R))
    prec = inter / max(float(sum(arc_len[a] for a in P)), 1e-9)
    rec = inter / max(float(sum(arc_len[a] for a in R)), 1e-9)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return dict(route_precision=prec, route_recall=rec, route_f1=f1)


def top1_accuracy(pred_arcs, label_arcs, valid=None):
    p, l = np.asarray(pred_arcs), np.asarray(label_arcs)
    m = np.ones(len(p), bool) if valid is None else np.asarray(valid)
    return float((p[m] == l[m]).mean()) if m.any() else np.nan
