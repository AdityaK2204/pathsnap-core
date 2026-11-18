"""Reading the raw per-trip files (.track, .nodes, .arcs, .route)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame          # lon, lat, t
    nodes: np.ndarray            # (n_nodes, 2) lon, lat
    arcs: np.ndarray             # (n_arcs, 2) from_node, to_node


def list_trips(split_dir):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(p.name for p in split_dir.iterdir()
                  if p.is_dir() and (p / f"{p.name}.track").exists())


def _read_table(path, dtype, ncols):
    arr = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c").to_numpy()
    return arr.reshape(-1, ncols)


def load_trip(split_dir, trip_id):
    """Load everything needed for prediction. The reference route is NOT read here."""
    d = Path(split_dir) / trip_id
    tr = _read_table(d / f"{trip_id}.track", np.float64, 3)
    track = pd.DataFrame(tr, columns=["lon", "lat", "t"])
    nodes = _read_table(d / f"{trip_id}.nodes", np.float64, 2)
    arcs = _read_table(d / f"{trip_id}.arcs", np.int64, 2)
    return Trip(trip_id, track, nodes, arcs)


def load_route(split_dir, trip_id):
    """Reference arc sequence. Only used for label derivation and scoring."""
    p = Path(split_dir) / trip_id / f"{trip_id}.route"
    if not p.exists():
        return None
    return _read_table(p, np.int64, 1).ravel()


def clean_track(track):
    """Drop invalid / duplicate fixes and enforce increasing timestamps.

    Returns the cleaned frame and the original row index of every kept fix,
    so predictions can be mapped back to the raw file.
    """
    df = track.copy()
    df["raw_idx"] = np.arange(len(df))
    ok = np.isfinite(df[["lon", "lat", "t"]]).all(axis=1)
    ok &= df.lon.between(-180, 180) & df.lat.between(-90, 90)
    df = df[ok]
    df = df.sort_values("t", kind="stable")
    df = df[~df.t.duplicated(keep="first")]
    return df.reset_index(drop=True)
