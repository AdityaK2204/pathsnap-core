"""HMM map matching with Viterbi decoding.

Hidden states   : at fix i, the candidate arcs of fix i (with the projected
                  position t on the arc), so state <-> road segment is kept 1:1.
Emissions       : log of the classifier probability of the candidate,
                  renormalised over the fix's candidates, scaled by
                  `emission_weight`.
Transitions     : from (arc a, t_a) at fix i-1 to (arc b, t_b) at fix i the
                  driving distance d_route through the directed network is
                  compared with the straight-line distance d_gc between the two
                  fixes;  log P = -|d_route - d_gc| / beta  (Newson & Krumm 2009).
                  d_route is searched only up to 2*d_gc + 150 m; unreachable
                  pairs get probability 0.
Gaps / breaks   : the trip is split into independent segments when the time
                  gap exceeds GAP_BREAK_S or the jump exceeds GAP_BREAK_M. If
                  no candidate of fix i is reachable from any surviving state of
                  fix i-1, the chain is restarted at fix i (HMM break) instead
                  of forcing an impossible transition.
"""
from dataclasses import dataclass

import numpy as np

from . import config

ROUTE_SLACK = 2.0
ROUTE_EXTRA_M = 150.0


@dataclass
class DecoderParams:
    beta: float = 10.0
    emission_weight: float = 1.0
    prob_floor: float = 1e-4


def candidate_lists(cands, probs, n_fixes):
    """Split candidate rows into per-fix arrays (arc ids, t_proj, probability, row index)."""
    order = np.argsort(cands.fix.to_numpy(), kind="stable")
    fix = cands.fix.to_numpy()[order]
    bounds = np.searchsorted(fix, np.arange(n_fixes + 1))
    arcs = cands.arc_id.to_numpy()[order]
    tp = cands.t_proj.to_numpy()[order]
    pr = np.asarray(probs)[order]
    out = []
    for i in range(n_fixes):
        s, e = bounds[i], bounds[i + 1]
        out.append((arcs[s:e], tp[s:e], pr[s:e], order[s:e]))
    return out


def segment_starts(xy, t):
    dt = np.diff(t)
    step = np.hypot(*np.diff(xy, axis=0).T)
    brk = (dt > config.GAP_BREAK_S) | (step > config.GAP_BREAK_M)
    return np.r_[True, brk]


def transition_costs(clists, xy, t, net):
    """|d_route - d_gc| for every consecutive pair of fixes (np.inf if unreachable).

    The result does not depend on decoder parameters, so it can be computed once
    and reused while tuning beta / emission weight.
    """
    starts = segment_starts(xy, t)
    costs = [None]
    arcs_from, arcs_to, alen = net.arcs[:, 0], net.arcs[:, 1], net.arc_len
    for i in range(1, len(xy)):
        if starts[i]:
            costs.append(None)
            continue
        a_arcs, a_t = clists[i - 1][0], clists[i - 1][1]
        b_arcs, b_t = clists[i][0], clists[i][1]
        d_gc = float(np.hypot(*(xy[i] - xy[i - 1])))
        limit = ROUTE_SLACK * d_gc + ROUTE_EXTRA_M
        D = np.full((len(a_arcs), len(b_arcs)), np.inf)
        b_start = arcs_from[b_arcs]
        b_off = b_t * alen[b_arcs]
        for r, (a, ta) in enumerate(zip(a_arcs, a_t)):
            rem = (1.0 - ta) * alen[a]
            reach = net.dist_from(int(arcs_to[a]), limit)
            for c, b in enumerate(b_arcs):
                if a == b:
                    D[r, c] = abs(b_t[c] - ta) * alen[a]
                else:
                    sp = reach.get(int(b_start[c]))
                    if sp is not None:
                        D[r, c] = rem + sp + b_off[c]
        costs.append(np.abs(D - d_gc))
    return costs, starts


def viterbi(clists, costs, params: DecoderParams):
    """Return (chosen candidate position per fix, number of HMM breaks)."""
    n = len(clists)
    choice = np.zeros(n, np.int64)
    back = [None] * n
    emis = []
    for arcs, _, pr, _ in clists:
        p = np.maximum(pr, params.prob_floor)
        p = p / p.sum()
        emis.append(params.emission_weight * np.log(p))

    breaks = 0
    seg_start = 0
    score = emis[0].copy()

    def backtrack(end, score_end):
        k = int(np.argmax(score_end))
        for j in range(end, seg_start - 1, -1):
            choice[j] = k
            if j > seg_start:
                k = back[j][k]

    for i in range(1, n):
        C = costs[i]
        if C is not None:
            cand = score[:, None] - C / params.beta
            best_prev = np.argmax(cand, axis=0)
            best = cand[best_prev, np.arange(cand.shape[1])]
            if np.isfinite(best).any():
                back[i] = best_prev
                score = best + emis[i]
                continue
            breaks += 1
        # new segment (time gap / jump) or no reachable state: restart the chain
        backtrack(i - 1, score)
        seg_start = i
        score = emis[i].copy()
    backtrack(n - 1, score)
    return choice, breaks


def reconstruct_path(pred_arcs, net, xy):
    """Connect the per-fix arcs into driven arc sequences using shortest paths.

    Returns a list of pieces (lists of arc ids); a new piece starts where no
    connecting path exists.
    """
    pieces, cur = [], []
    prev = None
    for i, a in enumerate(pred_arcs):
        a = int(a)
        if prev is None:
            cur = [a]
        elif a != prev:
            if net.arcs[prev, 1] == net.arcs[a, 0]:
                cur.append(a)
            else:
                lim = ROUTE_SLACK * float(np.hypot(*(xy[i] - xy[i - 1]))) + ROUTE_EXTRA_M + 500.0
                link = net.path_between(int(net.arcs[prev, 1]), int(net.arcs[a, 0]), lim)
                if link is None:
                    pieces.append(cur)
                    cur = [a]
                else:
                    cur.extend(link)
                    cur.append(a)
        prev = a
    if cur:
        pieces.append(cur)
    return pieces


def decode(prepared, probs, params: DecoderParams, costs=None):
    """Full decoding for one prepared trip. Returns dict with per-fix arcs and path pieces."""
    n = len(prepared.xy)
    clists = candidate_lists(prepared.cands, probs, n)
    if costs is None:
        costs, _ = transition_costs(clists, prepared.xy, prepared.t, prepared.net)
    choice, breaks = viterbi(clists, costs, params)
    hmm_arcs = np.array([clists[i][0][choice[i]] for i in range(n)])
    clf_arcs = np.array([cl[0][int(np.argmax(cl[2]))] for cl in clists])
    pieces = reconstruct_path(hmm_arcs, prepared.net, prepared.xy)
    return dict(hmm_arcs=hmm_arcs, clf_arcs=clf_arcs, path_pieces=pieces, hmm_breaks=breaks)
