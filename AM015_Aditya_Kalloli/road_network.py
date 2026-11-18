"""Per-trip road network: cropping, spatial index and bounded shortest paths."""
import heapq

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .geo import LocalProjection, bearing_xy


class RoadNetwork:
    """Directed road graph of one trip in local metric coordinates.

    Arc ids are always the original 0-based row numbers of the .arcs file, so
    candidate predictions stay comparable with the reference route.
    """

    def __init__(self, nodes_lonlat, arcs, proj: LocalProjection, bbox_xy=None,
                 margin=config.CROP_MARGIN_M, spacing=config.SAMPLE_SPACING_M):
        self.proj = proj
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs_total = len(arcs)
        self.nodes_lonlat = nodes_lonlat
        self.node_xy = proj.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.arcs = arcs

        u, v = arcs[:, 0], arcs[:, 1]
        A, B = self.node_xy[u], self.node_xy[v]
        self.arc_len = np.hypot(*(B - A).T)
        self.arc_bearing = bearing_xy(*(B - A).T)

        # crop to the area around the trajectory
        keep = (u != v)
        if bbox_xy is not None:
            (x0, y0), (x1, y1) = bbox_xy
            x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
            inA = (A[:, 0] >= x0) & (A[:, 0] <= x1) & (A[:, 1] >= y0) & (A[:, 1] <= y1)
            inB = (B[:, 0] >= x0) & (B[:, 0] <= x1) & (B[:, 1] >= y0) & (B[:, 1] <= y1)
            keep &= inA | inB
        self.arc_ids = np.flatnonzero(keep)

        # degree and reverse-twin information (computed on the kept arcs)
        ku, kv = u[self.arc_ids], v[self.arc_ids]
        self.out_deg = np.bincount(ku, minlength=self.n_nodes)
        self.in_deg = np.bincount(kv, minlength=self.n_nodes)
        pair_key = ku.astype(np.int64) * self.n_nodes + kv
        rev_key = kv.astype(np.int64) * self.n_nodes + ku
        self.has_twin = np.zeros(self.n_arcs_total, bool)
        self.has_twin[self.arc_ids] = np.isin(rev_key, pair_key)

        # CSR adjacency over kept arcs for Dijkstra
        order = np.argsort(ku, kind="stable")
        self._adj_arc = self.arc_ids[order]
        self._adj_to = kv[order]
        self._adj_w = self.arc_len[self._adj_arc]
        self._indptr = np.zeros(self.n_nodes + 1, np.int64)
        np.add.at(self._indptr, ku + 1, 1)
        self._indptr = np.cumsum(self._indptr)
        self._sp_cache = {}

        # densified sample points for the spatial index
        n_s = np.maximum(1, np.ceil(self.arc_len[self.arc_ids] / spacing).astype(int)) + 1
        rep = np.repeat(np.arange(len(self.arc_ids)), n_s)
        offs = np.arange(n_s.sum()) - np.repeat(np.cumsum(n_s) - n_s, n_s)
        frac = offs / np.repeat(n_s - 1, n_s)
        a_ids = self.arc_ids[rep]
        pts = self.node_xy[u[a_ids]] + frac[:, None] * (self.node_xy[v[a_ids]] - self.node_xy[u[a_ids]])
        self._sample_arc = a_ids
        self._tree = cKDTree(pts)
        self.spacing = spacing

    @classmethod
    def for_trip(cls, trip, proj, xy_track):
        bbox = (xy_track.min(0), xy_track.max(0))
        return cls(trip.nodes, trip.arcs, proj, bbox_xy=bbox)

    # ---- spatial queries -------------------------------------------------
    def arcs_near(self, xy, radius):
        """List of arrays: arc ids with a sample point within radius of each xy."""
        hits = self._tree.query_ball_point(xy, r=radius + self.spacing / 2)
        return [np.unique(self._sample_arc[h]) if len(h) else np.empty(0, np.int64) for h in hits]

    def nearest_arcs(self, xy, k):
        kk = min(len(self._sample_arc), 16 * k)
        _, idx = self._tree.query(xy, k=kk)
        idx = np.atleast_2d(idx)
        return [np.unique(self._sample_arc[row]) for row in idx]

    def arc_endpoints_xy(self, arc_ids):
        return self.node_xy[self.arcs[arc_ids, 0]], self.node_xy[self.arcs[arc_ids, 1]]

    # ---- shortest paths --------------------------------------------------
    def dist_from(self, src, limit):
        """Bounded Dijkstra from node src. Returns {node: distance} for nodes within limit."""
        cached = self._sp_cache.get(src)
        if cached is not None and cached[0] >= limit:
            return cached[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        indptr, to, w = self._indptr, self._adj_to, self._adj_w
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf):
                continue
            for e in range(indptr[n], indptr[n + 1]):
                m = int(to[e])
                nd = d + w[e]
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    heapq.heappush(heap, (nd, m))
        self._sp_cache[src] = (limit, dist)
        return dist

    def path_between(self, src, dst, limit):
        """Arc sequence of the shortest path src -> dst (empty if src == dst, None if unreachable)."""
        if src == dst:
            return []
        dist = {src: 0.0}
        prev = {}
        heap = [(0.0, src)]
        indptr, to, w, arc = self._indptr, self._adj_to, self._adj_w, self._adj_arc
        while heap:
            d, n = heapq.heappop(heap)
            if n == dst:
                break
            if d > dist.get(n, np.inf):
                continue
            for e in range(indptr[n], indptr[n + 1]):
                m = int(to[e])
                nd = d + w[e]
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    prev[m] = (n, int(arc[e]))
                    heapq.heappush(heap, (nd, m))
        if dst not in prev:
            return None
        out, n = [], dst
        while n != src:
            n, a = prev[n]
            out.append(a)
        return out[::-1]
