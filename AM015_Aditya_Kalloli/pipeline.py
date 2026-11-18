"""End-to-end inference: raw trip files -> per-fix road segments and route.

`MapMatcher.match` never reads the reference route. Scoring is a separate
step (`score_against_reference`) that the API runs after prediction, using
reference data that stays on the server.
"""
import json
import time

import joblib
import numpy as np

from . import config
from .data_io import load_route, load_trip
from .features import FEATURES
from .hmm_viterbi import DecoderParams, decode
from .label_alignment import make_labels
from .preprocess import prepare_trip
from .route_metrics import arc_set_scores, reachable_continuity, route_continuity


class MapMatcher:
    def __init__(self, model_path=None, decoder_path=None):
        model_path = model_path or config.ARTIFACTS_DIR / "final_model.joblib"
        decoder_path = decoder_path or config.ARTIFACTS_DIR / "decoder_config.json"
        self.model = joblib.load(model_path)
        meta = config.ARTIFACTS_DIR / "final_model.json"
        self.model_info = json.loads(meta.read_text()) if meta.exists() else {}
        dp = json.loads(decoder_path.read_text()) if decoder_path.exists() else {}
        self.params = DecoderParams(**{k: v for k, v in dp.items() if k in DecoderParams.__annotations__})

    def match(self, split_dir, trip_id):
        timings = {}
        t0 = time.perf_counter()
        trip = load_trip(split_dir, trip_id)
        prepared = prepare_trip(trip)
        timings["preprocess_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        probs = self.model.predict_proba(prepared.cands[FEATURES].to_numpy())[:, 1]
        timings["classifier_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dec = decode(prepared, probs, self.params)
        timings["hmm_s"] = time.perf_counter() - t0
        return dict(trip=trip, prepared=prepared, probs=probs, timings=timings, **dec)

    @staticmethod
    def path_coordinates(result):
        """[ [ [lat, lon], ... ], ... ] polylines of the reconstructed route pieces."""
        trip = result["trip"]
        lines = []
        for piece in result["path_pieces"]:
            if not piece:
                continue
            nodes = [trip.arcs[piece[0], 0]] + [trip.arcs[a, 1] for a in piece]
            lines.append([[float(trip.nodes[n, 1]), float(trip.nodes[n, 0])] for n in nodes])
        return lines


def score_against_reference(result, split_dir, trip_id):
    """Server-side evaluation of a prediction against the reference route."""
    route = load_route(split_dir, trip_id)
    if route is None:
        return None
    p, trip = result["prepared"], result["trip"]
    lab = make_labels(p.xy, p.net.node_xy, trip.arcs, route)
    v = lab["label_valid"]
    la = lab["label_arc"]
    path = [a for piece in result["path_pieces"] for a in piece]
    out = dict(
        n_fixes=int(len(v)), n_labelled_fixes=int(v.sum()),
        top1_accuracy_hmm=float((result["hmm_arcs"] == la)[v].mean()),
        top1_accuracy_classifier=float((result["clf_arcs"] == la)[v].mean()),
        route_continuity_hmm=route_continuity(result["hmm_arcs"], trip.arcs),
        route_continuity_classifier=route_continuity(result["clf_arcs"], trip.arcs),
        reachable_continuity_hmm=reachable_continuity(result["hmm_arcs"], p.net, p.xy),
        hmm_breaks=int(result["hmm_breaks"]),
        path_pieces=len(result["path_pieces"]),
    )
    out.update(arc_set_scores(path, route, p.net.arc_len))
    return out
