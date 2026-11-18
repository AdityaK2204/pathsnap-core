"""Geometry helpers.

Trips are spread over the whole globe, so a single projected CRS does not fit.
Each trip gets its own local equirectangular projection centred on the median
GPS fix. For a trip 30 km across the east-west scale error at its edge is
about 0.2 % at 45 deg and 0.4 % at 60 deg latitude - centimetres on a 1 s step,
far below GPS noise - and it keeps distances in metres.
"""
import numpy as np

EARTH_R = 6371008.8


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = np.radians(1.0) * EARTH_R * np.cos(np.radians(self.lat0))
        self.ky = np.radians(1.0) * EARTH_R

    @classmethod
    def for_track(cls, lon, lat):
        return cls(np.median(lon), np.median(lat))

    def forward(self, lon, lat):
        lon = np.asarray(lon, float)
        dlon = (lon - self.lon0 + 180.0) % 360.0 - 180.0   # safe across the antimeridian
        return np.stack([dlon * self.kx, (np.asarray(lat, float) - self.lat0) * self.ky], -1)

    def inverse(self, xy):
        xy = np.asarray(xy, float)
        return xy[..., 0] / self.kx + self.lon0, xy[..., 1] / self.ky + self.lat0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def bearing_xy(dx, dy):
    """Compass bearing in degrees (0 = north, clockwise) from planar deltas."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff(a, b):
    """Absolute smallest difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment(P, A, B):
    """Distance from points P to segments AB (broadcasting over leading dims).

    Returns (distance, t) where t in [0, 1] is the position of the projection.
    """
    AB = B - A
    L2 = (AB * AB).sum(-1)
    t = ((P - A) * AB).sum(-1) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    C = A + t[..., None] * AB
    return np.sqrt(((P - C) ** 2).sum(-1)), t
