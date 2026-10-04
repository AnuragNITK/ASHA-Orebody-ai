"""Block grid construction and the topographic surface.

Block size follows the standard rule of thumb of one half to one third of the
average drill spacing: smaller than that and the model claims resolution the
drilling does not support; larger and selectivity is lost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

MAX_BLOCKS = 400_000  # keeps kriging and the browser scene interactive


def _round_block_size(value: float) -> float:
    """Snap to a conventional block dimension."""
    for candidate in (2.5, 5.0, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0):
        if value <= candidate * 1.15:
            return candidate
    return float(np.round(value / 10.0) * 10.0)


def recommend_block_size(drill_spacing: float, divisor: float = 2.0) -> dict:
    raw = drill_spacing / divisor
    size = _round_block_size(raw)
    return {
        "block_size": size,
        "drill_spacing": float(drill_spacing),
        "rationale": (f"average drill spacing is {drill_spacing:.1f} m; blocks are set to "
                      f"{size:.1f} m, within the conventional one half to one third of "
                      f"drill spacing"),
    }


def build_block_grid(bbox: np.ndarray, block_size: float | tuple[float, float, float],
                     padding: float = 0.0) -> pd.DataFrame:
    """Regular centroid grid covering ``bbox`` (2x3 array of min/max xyz)."""
    if np.isscalar(block_size):
        bx = by = bz = float(block_size)
    else:
        bx, by, bz = (float(v) for v in block_size)

    lo = np.asarray(bbox[0], float) - padding
    hi = np.asarray(bbox[1], float) + padding

    axes = []
    for dim, size in zip(range(3), (bx, by, bz)):
        n = max(int(np.ceil((hi[dim] - lo[dim]) / size)), 1)
        start = lo[dim] + size / 2.0
        axes.append(start + np.arange(n) * size)

    n_total = int(np.prod([len(a) for a in axes]))
    if n_total > MAX_BLOCKS:
        scale = (n_total / MAX_BLOCKS) ** (1 / 3)
        bx, by, bz = bx * scale, by * scale, bz * scale
        log.warning("grid of %d blocks exceeds the %d cap; block size relaxed to %.1f m",
                    n_total, MAX_BLOCKS, bx)
        return build_block_grid(bbox, (bx, by, bz), padding)

    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    blocks = pd.DataFrame({
        "x": gx.ravel(), "y": gy.ravel(), "z": gz.ravel(),
    })
    blocks["block_id"] = np.arange(len(blocks))
    blocks.attrs["block_size"] = (bx, by, bz)
    blocks.attrs["block_volume"] = bx * by * bz
    log.info("built block grid: %d blocks of %.1f x %.1f x %.1f m",
             len(blocks), bx, by, bz)
    return blocks


class Topography:
    """Topographic surface interpolated from collar elevations.

    Blocks above it are air and are excluded from the resource and from the pit,
    which is what stops the optimiser mining the sky.
    """

    def __init__(self, collar: pd.DataFrame):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        pts = collar[["x", "y"]].to_numpy(float)
        z = collar["z"].to_numpy(float)
        self.mean_z = float(z.mean())
        self._nearest = NearestNDInterpolator(pts, z)
        self._linear = None
        if len(np.unique(pts, axis=0)) >= 3:
            try:
                self._linear = LinearNDInterpolator(pts, z)
            except Exception:  # pragma: no cover - degenerate collar layout
                self._linear = None

    def elevation(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self._linear is not None:
            z = self._linear(xy)
            gaps = ~np.isfinite(z)
            if gaps.any():
                z[gaps] = self._nearest(xy[gaps])
            return z
        return self._nearest(xy)

    def below_surface(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        return xyz[:, 2] <= self.elevation(xyz[:, :2])


def data_density_summary(samples: pd.DataFrame, frame, block_size: float) -> dict:
    """Evidence pack on data support and anisotropy for LLM Call 1.

    This is measurement, not judgement: nearest-neighbour spacing, sample counts,
    and the extent of the sampled cloud along each principal axis.
    """
    from scipy.spatial import cKDTree

    xyz = samples[["x", "y", "z"]].to_numpy(float)
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=min(2, len(xyz)))
    nn = d[:, 1] if d.ndim > 1 and d.shape[1] > 1 else np.array([0.0])

    local = frame.to_local(xyz)
    extents = {axis: float(local[:, i].max() - local[:, i].min())
               for i, axis in enumerate(("along_strike", "across_strike", "vertical"))}

    # sample spacing along strike vs across strike, the practical anisotropy cue
    return {
        "n_samples": int(len(samples)),
        "mean_nearest_neighbour_m": float(np.mean(nn)),
        "median_nearest_neighbour_m": float(np.median(nn)),
        "extent_m": extents,
        "anisotropy_ratio_strike_over_across": float(
            extents["along_strike"] / max(extents["across_strike"], 1e-6)),
        "anisotropy_ratio_strike_over_vertical": float(
            extents["along_strike"] / max(extents["vertical"], 1e-6)),
        "block_size_m": float(block_size),
        "samples_per_block_volume": float(
            len(samples) / max(np.prod(list(extents.values())) / block_size ** 3, 1e-9)),
    }
