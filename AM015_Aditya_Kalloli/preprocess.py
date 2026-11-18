"""One preprocessing path shared by dataset building, evaluation and the API."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .data_io import Trip, clean_track
from .features import build_features
from .geo import LocalProjection
from .road_network import RoadNetwork


@dataclass
class PreparedTrip:
    trip_id: str
    track: pd.DataFrame      # cleaned track with raw_idx
    xy: np.ndarray
    t: np.ndarray
    proj: LocalProjection
    net: RoadNetwork
    cands: pd.DataFrame      # candidate rows with features


def prepare_trip(trip: Trip) -> PreparedTrip:
    track = clean_track(trip.track)
    proj = LocalProjection.for_track(track.lon.to_numpy(), track.lat.to_numpy())
    xy = proj.forward(track.lon.to_numpy(), track.lat.to_numpy())
    t = track.t.to_numpy()
    net = RoadNetwork.for_trip(trip, proj, xy)
    cands = generate_candidates(xy, net)
    cands = build_features(cands, xy, t, net)
    return PreparedTrip(trip.trip_id, track, xy, t, proj, net, cands)
