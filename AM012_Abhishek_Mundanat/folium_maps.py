"""Folium maps of trips: GPS fixes, nearby road network, reference route, predicted route.

Usage (from the repository root):
    python -m AM012_Abhishek_Mundanat.folium_maps                 # EDA maps (GPS + network + reference)
    python -m AM012_Abhishek_Mundanat.folium_maps --predictions   # add predicted routes from results/hmm/predictions
"""
import argparse
import json

import folium
import numpy as np
from branca.element import Element
from scipy.spatial import cKDTree

from AM015_Aditya_Kalloli import config
from AM015_Aditya_Kalloli.data_io import load_route, load_trip
from AM015_Aditya_Kalloli.geo import LocalProjection

OUT = config.RESULTS_DIR / "maps"

COLORS = dict(gps="#1f4fff", network="#9a9a9a", reference="#1a9850", predicted="#d7301f",
              classifier="#ff9f1c", start="green", end="black")


def _arc_lines(trip, arc_ids):
    a = trip.arcs[np.asarray(arc_ids, int)]
    return [[[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]] for u, v in a]


def _route_lines(trip, arc_seq):
    """Merge a chain of arcs into polylines (breaks where consecutive arcs do not touch)."""
    lines, cur = [], []
    for a in arc_seq:
        u, v = trip.arcs[a]
        if not cur or cur[-1] != [trip.nodes[u, 1], trip.nodes[u, 0]]:
            if len(cur) > 1:
                lines.append(cur)
            cur = [[trip.nodes[u, 1], trip.nodes[u, 0]]]
        cur.append([trip.nodes[v, 1], trip.nodes[v, 0]])
    if len(cur) > 1:
        lines.append(cur)
    return lines


def nearby_arcs(trip, buffer_m=150.0):
    """Arcs with an endpoint within buffer_m of any GPS fix (keeps the HTML small)."""
    tr = trip.track
    proj = LocalProjection.for_track(tr.lon.to_numpy(), tr.lat.to_numpy())
    pxy = proj.forward(tr.lon.to_numpy(), tr.lat.to_numpy())
    nxy = proj.forward(trip.nodes[:, 0], trip.nodes[:, 1])
    tree = cKDTree(pxy)
    d, _ = tree.query(nxy, distance_upper_bound=buffer_m)
    near_node = np.isfinite(d)
    return np.flatnonzero(near_node[trip.arcs[:, 0]] | near_node[trip.arcs[:, 1]])


def add_legend(m, items):
    rows = "".join(f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{label}</div>' for label, c in items)
    html = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
            'padding:8px 10px;border:1px solid #999;border-radius:4px;font:12px sans-serif">'
            f'<b>Layers</b>{rows}</div>')
    m.get_root().html.add_child(Element(html))


def trip_map(trip, route=None, hmm_path=None, clf_arcs=None, title=None, max_gps=4000):
    tr = trip.track
    lat, lon = tr.lat.to_numpy(), tr.lon.to_numpy()
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=13,
                   tiles=None, control_scale=True)
    # Esri first: OpenStreetMap's own tile server refuses requests without a Referer
    # (e.g. when the HTML file is opened directly from disk)
    folium.TileLayer("Esri.WorldStreetMap", name="Esri street map").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    legend = []

    fg = folium.FeatureGroup(name="Road network (near track)", show=True)
    folium.PolyLine(_arc_lines(trip, nearby_arcs(trip)), color=COLORS["network"], weight=2, opacity=0.6).add_to(fg)
    fg.add_to(m)
    legend.append(("Road network", COLORS["network"]))

    if route is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_route_lines(trip, route), color=COLORS["reference"], weight=7, opacity=0.6,
                        tooltip="reference route").add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))

    # GPS below the predicted layers so the predicted route stays visible
    fg = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(tr) // max_gps)
    for i in range(0, len(tr), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.7,
                            weight=0, tooltip=f"fix {i}, t={tr.t.iloc[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    legend.append(("GPS observations", COLORS["gps"]))

    if clf_arcs is not None:
        fg = folium.FeatureGroup(name="Classifier-only arcs", show=False)
        uniq = list(dict.fromkeys(int(a) for a in clf_arcs))
        folium.PolyLine(_arc_lines(trip, uniq), color=COLORS["classifier"], weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)
        legend.append(("Classifier-only arcs", COLORS["classifier"]))

    if hmm_path is not None:
        fg = folium.FeatureGroup(name="Predicted route (HMM)", show=True)
        for piece in hmm_path:
            folium.PolyLine(_route_lines(trip, piece), color=COLORS["predicted"], weight=3.5, opacity=0.95,
                            tooltip="predicted route").add_to(fg)
        fg.add_to(m)
        legend.append(("Predicted route (HMM)", COLORS["predicted"]))

    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    add_legend(m, legend)
    if title:
        m.get_root().html.add_child(Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:4px 8px;'
            f'border:1px solid #999;font:13px sans-serif"><b>{title}</b></div>'))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--trips", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.predictions:
        pred_dir = config.RESULTS_DIR / "hmm" / "predictions"
        for p in sorted(pred_dir.glob("*.json")):
            pr = json.loads(p.read_text())
            tid = pr["trip_id"]
            trip = load_trip(config.STUDENT_TEST_DIR, tid)
            m = trip_map(trip, route=load_route(config.STUDENT_TEST_DIR, tid), hmm_path=pr["path_pieces"],
                         clf_arcs=pr["classifier_arcs"], title=f"Student-test trip {tid}: reference vs predicted")
            m.save(str(OUT / f"prediction_{tid}.html"))
            print("saved", OUT / f"prediction_{tid}.html")
        return

    picks = args.trips.split(",") if args.trips else ["train:00000001", "train:00000054", "student_test:00000056"]
    for item in picks:
        split, tid = item.split(":")
        d = config.TRAIN_DIR if split == "train" else config.STUDENT_TEST_DIR
        trip = load_trip(d, tid)
        m = trip_map(trip, route=load_route(d, tid), title=f"{split} trip {tid}: GPS, road network, reference route")
        m.save(str(OUT / f"eda_{split}_{tid}.html"))
        print("saved", OUT / f"eda_{split}_{tid}.html")


if __name__ == "__main__":
    main()
