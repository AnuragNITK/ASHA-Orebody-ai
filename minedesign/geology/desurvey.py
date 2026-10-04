"""Minimum-curvature desurvey.

Standard directional-drilling formula, implemented directly. Between two survey
stations the hole is assumed to follow a circular arc; the ratio factor RF
corrects the straight-line (balanced tangential) step for that curvature.

Convention used throughout the pipeline:
  azimuth  degrees clockwise from grid north (0 = +Y, 90 = +X)
  dip      degrees, negative downwards (Surpac convention)
  x        easting, y        northing, z elevation (increases upwards)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


def _direction_cosines(azimuth_deg: np.ndarray, dip_deg: np.ndarray):
    """Unit direction vector for each station."""
    az = np.radians(azimuth_deg)
    # inclination measured from horizontal, negative down
    incl = np.radians(dip_deg)
    cos_incl = np.cos(incl)
    dx = cos_incl * np.sin(az)
    dy = cos_incl * np.cos(az)
    dz = np.sin(incl)
    return dx, dy, dz


def _ratio_factor(dogleg: np.ndarray) -> np.ndarray:
    """RF = (2/DL) * tan(DL/2), with the small-angle limit handled explicitly."""
    rf = np.ones_like(dogleg)
    big = dogleg > 1e-9
    rf[big] = (2.0 / dogleg[big]) * np.tan(dogleg[big] / 2.0)
    return rf


def desurvey_hole(collar_xyz, depths, azimuths, dips) -> pd.DataFrame:
    """Desurvey one hole. Returns a trace with depth, x, y, z per station."""
    depths = np.asarray(depths, dtype=float)
    azimuths = np.asarray(azimuths, dtype=float)
    dips = np.asarray(dips, dtype=float)

    order = np.argsort(depths)
    depths, azimuths, dips = depths[order], azimuths[order], dips[order]

    if depths.size == 0:  # no survey: vertical hole
        depths = np.array([0.0])
        azimuths = np.array([0.0])
        dips = np.array([-90.0])

    if depths[0] > 1e-9:  # extend the first station up to the collar
        depths = np.concatenate([[0.0], depths])
        azimuths = np.concatenate([[azimuths[0]], azimuths])
        dips = np.concatenate([[dips[0]], dips])

    n = depths.size
    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)
    x[0], y[0], z[0] = collar_xyz

    if n == 1:
        return pd.DataFrame({"depth": depths, "x": x, "y": y, "z": z,
                             "azimuth": azimuths, "dip": dips})

    dx, dy, dz = _direction_cosines(azimuths, dips)
    d1 = np.stack([dx[:-1], dy[:-1], dz[:-1]], axis=1)
    d2 = np.stack([dx[1:], dy[1:], dz[1:]], axis=1)

    cos_dl = np.clip((d1 * d2).sum(axis=1), -1.0, 1.0)
    dogleg = np.arccos(cos_dl)
    rf = _ratio_factor(dogleg)
    dmd = np.diff(depths)
    half = dmd * rf / 2.0

    steps = (d1 + d2) * half[:, None]
    x[1:] = x[0] + np.cumsum(steps[:, 0])
    y[1:] = y[0] + np.cumsum(steps[:, 1])
    z[1:] = z[0] + np.cumsum(steps[:, 2])

    return pd.DataFrame({"depth": depths, "x": x, "y": y, "z": z,
                         "azimuth": azimuths, "dip": dips})


def desurvey_minimum_curvature(collar: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    """Desurvey every hole. Returns the concatenated 3D trace, one row per station."""
    traces = []
    # dict(groupby) does not yield (name, group) pairs in pandas 3, so the
    # explicit comprehension is the correct form here despite what C416 suggests.
    survey_by_hole = {hid: g for hid, g in survey.groupby("hole_id", sort=False)}  # noqa: C416

    for row in collar.itertuples(index=False):
        g = survey_by_hole.get(row.hole_id)
        if g is None or g.empty:
            depths, az, dip = np.array([]), np.array([]), np.array([])
        else:
            depths = g["depth"].to_numpy(float)
            az = g["azimuth"].to_numpy(float)
            dip = g["dip"].to_numpy(float)

        max_depth = getattr(row, "max_depth", np.nan)
        if max_depth is not None and np.isfinite(max_depth) and (
                depths.size == 0 or max_depth > depths.max() + 1e-9):
            depths = np.concatenate([depths, [float(max_depth)]])
            az = np.concatenate([az, [az[-1] if az.size else 0.0]])
            dip = np.concatenate([dip, [dip[-1] if dip.size else -90.0]])

        trace = desurvey_hole((row.x, row.y, row.z), depths, az, dip)
        trace.insert(0, "hole_id", row.hole_id)
        traces.append(trace)

    out = pd.concat(traces, ignore_index=True)
    log.info("desurveyed %d holes into %d trace stations", collar["hole_id"].nunique(), len(out))
    return out


def interpolate_along_trace(traces: pd.DataFrame, hole_id: str,
                            depths: np.ndarray) -> np.ndarray:
    """Linear interpolation of xyz at arbitrary downhole depths of one hole.

    Interpolating between desurveyed stations is the standard way to place assay
    interval midpoints; the curvature is already captured in the station spacing.
    """
    t = traces[traces["hole_id"] == hole_id]
    if t.empty:
        return np.full((len(depths), 3), np.nan)
    td = t["depth"].to_numpy(float)
    return np.column_stack([
        np.interp(depths, td, t["x"].to_numpy(float)),
        np.interp(depths, td, t["y"].to_numpy(float)),
        np.interp(depths, td, t["z"].to_numpy(float)),
    ])


def attach_coordinates(intervals: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    """Add midpoint x/y/z (and endpoint coordinates) to a downhole interval table."""
    out = intervals.copy()
    mid = (out["from"].to_numpy(float) + out["to"].to_numpy(float)) / 2.0
    xyz = np.full((len(out), 3), np.nan)
    xyz_from = np.full((len(out), 3), np.nan)
    xyz_to = np.full((len(out), 3), np.nan)

    for hole_id, idx in out.groupby("hole_id", sort=False).indices.items():
        idx = np.asarray(idx)
        xyz[idx] = interpolate_along_trace(traces, hole_id, mid[idx])
        xyz_from[idx] = interpolate_along_trace(traces, hole_id, out["from"].to_numpy(float)[idx])
        xyz_to[idx] = interpolate_along_trace(traces, hole_id, out["to"].to_numpy(float)[idx])

    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["x_from"], out["y_from"], out["z_from"] = xyz_from[:, 0], xyz_from[:, 1], xyz_from[:, 2]
    out["x_to"], out["y_to"], out["z_to"] = xyz_to[:, 0], xyz_to[:, 1], xyz_to[:, 2]
    return out
