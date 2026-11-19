"""PathSnap inference API (runs on laptop 1).

Start from the repository root:
    uvicorn AM014_Addanki_Harsha_Rajeswari.api.main:app --host 0.0.0.0 --port 8000

Environment variables
    PATHSNAP_EVAL_DIR   folder with the evaluation trips (one sub-folder per trip,
                        same file layout as the raw data). Defaults to the
                        student-testing partition for development.

The reference .route files in that folder are read only by the server-side
scoring step, after prediction. They are never returned to the client.
"""
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from AM015_Aditya_Kalloli import config
from AM015_Aditya_Kalloli.data_io import list_trips
from AM015_Aditya_Kalloli.pipeline import MapMatcher, score_against_reference

from .schemas import PredictRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", config.STUDENT_TEST_DIR))
MAX_GPS_POINTS_RETURNED = 5000

app = FastAPI(title="PathSnap map-matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

matcher = MapMatcher()
log.info("model loaded: %s | decoder: %s | evaluation folder: %s (%d trips)",
         matcher.model_info.get("model", "?"), matcher.params, EVAL_DIR, len(list_trips(EVAL_DIR)))


@app.get("/health")
def health():
    return dict(status="ok", model=matcher.model_info.get("model"), decoder=vars(matcher.params),
                eval_dir=str(EVAL_DIR), n_trips=len(list_trips(EVAL_DIR)))


@app.get("/trips")
def trips(request: Request):
    ids = list_trips(EVAL_DIR)
    log.info("GET /trips from %s -> %d trips", request.client.host if request.client else "?", len(ids))
    out = []
    for tid in ids:
        with open(EVAL_DIR / tid / f"{tid}.track", "rb") as f:
            n = sum(1 for _ in f)
        out.append(dict(trip_id=tid, n_fixes=n))
    return dict(trips=out)


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    client = request.client.host if request.client else "?"
    if req.trip_id not in set(list_trips(EVAL_DIR)):
        log.warning("POST /predict from %s: unknown trip %r", client, req.trip_id)
        raise HTTPException(status_code=404, detail=f"unknown trip id {req.trip_id}")

    log.info("POST /predict from %s: trip %s received", client, req.trip_id)
    t0 = time.perf_counter()
    result = matcher.match(EVAL_DIR, req.trip_id)
    t_pred = time.perf_counter() - t0
    # evaluation happens after prediction, with reference data that stays on this machine
    metrics = score_against_reference(result, EVAL_DIR, req.trip_id)
    total = time.perf_counter() - t0

    p, trip = result["prepared"], result["trip"]
    step = max(1, len(p.track) // MAX_GPS_POINTS_RETURNED)
    gps = p.track[["lat", "lon"]].to_numpy()[::step].round(7).tolist()
    cand_arcs = sorted(set(int(a) for a in p.cands.arc_id))
    network = [[[float(trip.nodes[u, 1]), float(trip.nodes[u, 0])], [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]]
               for u, v in trip.arcs[cand_arcs]]

    if metrics:
        log.info("trip %s: %d fixes | top-1 HMM %.4f (classifier only %.4f) | continuity %.4f | "
                 "route F1 %.4f | prediction %.2fs total %.2fs", req.trip_id, metrics["n_fixes"],
                 metrics["top1_accuracy_hmm"], metrics["top1_accuracy_classifier"],
                 metrics["route_continuity_hmm"], metrics["route_f1"], t_pred, total)
    else:
        log.info("trip %s: %d fixes | no reference available | prediction %.2fs", req.trip_id, len(p.xy), t_pred)

    return dict(
        trip_id=req.trip_id,
        n_fixes=int(len(p.xy)),
        model=matcher.model_info.get("model"),
        timings_s={**{k: round(v, 3) for k, v in result["timings"].items()},
                   "prediction_total_s": round(t_pred, 3), "request_total_s": round(total, 3)},
        metrics=metrics,
        predicted_arcs=[int(a) for a in result["hmm_arcs"]],
        predicted_route=MapMatcher.path_coordinates(result),
        gps=gps,
        candidate_network=network,
    )
