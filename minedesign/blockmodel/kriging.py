"""Grade estimation: ordinary kriging and inverse distance weighting.

Pure execution. The variogram parameters and the choice between kriging and IDW
arrive from LLM Call 1; nothing here re-decides them. Blocks are estimated in
neighbourhood batches: a KD-tree finds the nearest composites to each block, and
each block gets its own small kriging system, which is the standard search
neighbourhood approach rather than one intractable global system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..logging_utils import get_logger
from .variogram import variogram_model

log = get_logger(__name__)


def _anisotropic_scale(anisotropy: dict | None, frame) -> np.ndarray | None:
    """Build a 3x3 matrix mapping world offsets into isotropic variogram space."""
    if not anisotropy or frame is None:
        return None
    along = float(anisotropy.get("along_strike", 1.0)) or 1.0
    across = float(anisotropy.get("across_strike", 1.0)) or 1.0
    vertical = float(anisotropy.get("vertical", 1.0)) or 1.0

    axes = np.vstack([frame.strike_dir, frame.across_dir, np.array([0.0, 0.0, 1.0])])
    scale = np.diag([1.0 / along, 1.0 / across, 1.0 / vertical])
    return scale @ axes


def _transform(offsets: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    if matrix is None:
        return offsets
    return offsets @ matrix.T


def ordinary_kriging(block_xyz: np.ndarray, sample_xyz: np.ndarray, sample_values: np.ndarray,
                     model: str, nugget: float, sill: float, rng: float,
                     max_samples: int = 24, min_samples: int = 4,
                     search_radius: float | None = None,
                     anisotropy_matrix: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordinary kriging with a moving neighbourhood.

    Returns ``(estimate, kriging_variance, n_samples_used)``. Blocks with fewer
    than ``min_samples`` neighbours inside the search radius are left as NaN
    rather than silently extrapolated.
    """
    block_xyz = np.asarray(block_xyz, float).reshape(-1, 3)
    sample_xyz = np.asarray(sample_xyz, float).reshape(-1, 3)
    sample_values = np.asarray(sample_values, float)

    if search_radius is None:
        search_radius = rng * 1.5

    tree = cKDTree(sample_xyz)
    neighbours = tree.query_ball_point(block_xyz, r=search_radius)

    est = np.full(len(block_xyz), np.nan)
    var = np.full(len(block_xyz), np.nan)
    used = np.zeros(len(block_xyz), dtype=int)

    total_sill = nugget + sill

    for b, idx in enumerate(neighbours):
        if len(idx) < min_samples:
            continue
        idx = np.asarray(idx)
        if len(idx) > max_samples:  # keep the closest samples
            d = np.linalg.norm(sample_xyz[idx] - block_xyz[b], axis=1)
            idx = idx[np.argsort(d)[:max_samples]]

        pts = sample_xyz[idx]
        vals = sample_values[idx]
        n = len(idx)

        # sample-to-sample covariance, expressed through the variogram
        offsets = pts[:, None, :] - pts[None, :, :]
        h = np.linalg.norm(_transform(offsets.reshape(-1, 3), anisotropy_matrix), axis=1).reshape(n, n)
        gamma = variogram_model(h, model, nugget, sill, rng)

        # block-to-sample
        h0 = np.linalg.norm(_transform(pts - block_xyz[b], anisotropy_matrix), axis=1)
        gamma0 = variogram_model(h0, model, nugget, sill, rng)

        # ordinary kriging system with the unbiasedness (Lagrange) constraint
        a = np.ones((n + 1, n + 1))
        a[:n, :n] = gamma
        a[n, n] = 0.0
        b_vec = np.ones(n + 1)
        b_vec[:n] = gamma0

        try:
            solution = np.linalg.solve(a, b_vec)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(a, b_vec, rcond=None)[0]

        weights = solution[:n]
        lagrange = solution[n]
        est[b] = float(weights @ vals)
        var[b] = float(max(weights @ gamma0 + lagrange, 0.0))
        used[b] = n

    filled = int(np.isfinite(est).sum())
    log.info("ordinary kriging estimated %d of %d blocks (search radius %.0f m)",
             filled, len(block_xyz), search_radius)
    if total_sill <= 0:
        log.warning("variogram total sill is zero; kriging variance is not meaningful")
    return est, var, used


def inverse_distance(block_xyz: np.ndarray, sample_xyz: np.ndarray, sample_values: np.ndarray,
                     power: float = 2.0, max_samples: int = 24, min_samples: int = 2,
                     search_radius: float = 100.0,
                     anisotropy_matrix: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse distance weighting, the safe fallback when data is too sparse to krige."""
    block_xyz = np.asarray(block_xyz, float).reshape(-1, 3)
    sample_xyz = np.asarray(sample_xyz, float).reshape(-1, 3)
    sample_values = np.asarray(sample_values, float)

    tree = cKDTree(sample_xyz)
    neighbours = tree.query_ball_point(block_xyz, r=search_radius)

    est = np.full(len(block_xyz), np.nan)
    spread = np.full(len(block_xyz), np.nan)
    used = np.zeros(len(block_xyz), dtype=int)

    for b, idx in enumerate(neighbours):
        if len(idx) < min_samples:
            continue
        idx = np.asarray(idx)
        d = np.linalg.norm(_transform(sample_xyz[idx] - block_xyz[b], anisotropy_matrix), axis=1)
        if len(idx) > max_samples:
            order = np.argsort(d)[:max_samples]
            idx, d = idx[order], d[order]

        if np.any(d < 1e-6):  # a sample sits in the block: take it directly
            est[b] = float(sample_values[idx[np.argmin(d)]])
            spread[b] = 0.0
            used[b] = 1
            continue

        w = 1.0 / d ** power
        w /= w.sum()
        vals = sample_values[idx]
        est[b] = float(w @ vals)
        # weighted variance stands in for an estimation-confidence measure
        spread[b] = float(w @ (vals - est[b]) ** 2)
        used[b] = len(idx)

    log.info("IDW (power %.1f) estimated %d of %d blocks",
             power, int(np.isfinite(est).sum()), len(block_xyz))
    return est, spread, used


def run_kriging(block_centroids: pd.DataFrame, composited: pd.DataFrame,
                variogram_params: dict, method: str = "ordinary_kriging",
                frame=None, anisotropy: dict | None = None,
                grade_column: str = "grade",
                max_samples: int = 24, min_samples: int = 4,
                search_radius: float | None = None) -> pd.DataFrame:
    """Estimate a grade and an estimation variance for every block.

    ``variogram_params`` and ``method`` come from LLM Call 1; this function only
    executes them.
    """
    block_xyz = block_centroids[["x", "y", "z"]].to_numpy(float)

    samples = composited.dropna(subset=[grade_column, "x", "y", "z"])
    sample_xyz = samples[["x", "y", "z"]].to_numpy(float)
    sample_values = samples[grade_column].to_numpy(float)
    if len(sample_values) == 0:
        raise ValueError("no composited samples with coordinates available for estimation")

    model = variogram_params.get("model", "spherical")
    nugget = float(variogram_params.get("nugget", 0.0))
    sill = float(variogram_params.get("sill", float(np.var(sample_values))))
    rng_ = float(variogram_params.get("range", 100.0))

    aniso_matrix = _anisotropic_scale(anisotropy, frame)
    radius = search_radius if search_radius is not None else rng_ * 1.5

    out = block_centroids.copy()
    if method == "idw":
        est, var, used = inverse_distance(
            block_xyz, sample_xyz, sample_values, max_samples=max_samples,
            min_samples=max(2, min_samples - 2), search_radius=radius,
            anisotropy_matrix=aniso_matrix)
        out["estimation_variance"] = var
    else:
        est, var, used = ordinary_kriging(
            block_xyz, sample_xyz, sample_values, model, nugget, sill, rng_,
            max_samples=max_samples, min_samples=min_samples, search_radius=radius,
            anisotropy_matrix=aniso_matrix)
        out["kriging_variance"] = var
        out["estimation_variance"] = var

    out["grade"] = est
    out["n_samples_used"] = used
    out["estimation_method"] = method
    out["search_radius"] = radius

    # classification by sample support, the standard confidence proxy
    out["confidence"] = pd.cut(
        out["n_samples_used"], bins=[-1, 3, 8, 1_000_000],
        labels=["inferred", "indicated", "measured"]).astype(str)
    out.loc[~np.isfinite(out["grade"]), "confidence"] = "unestimated"
    return out


def cross_validate(composited: pd.DataFrame, variogram_params: dict, method: str,
                   grade_column: str = "grade", frame=None,
                   anisotropy: dict | None = None, sample_size: int = 400,
                   seed: int = 0) -> dict:
    """Leave-one-out cross validation on a random subset of the composites.

    This is the honest check on whether the chosen variogram and method actually
    predict grades better than the naive mean, and it goes in the report.
    """
    samples = composited.dropna(subset=[grade_column, "x", "y", "z"])
    xyz = samples[["x", "y", "z"]].to_numpy(float)
    values = samples[grade_column].to_numpy(float)
    n = len(values)
    if n < 20:
        return {"performed": False, "reason": "fewer than 20 samples with coordinates"}

    rs = np.random.default_rng(seed)
    test_idx = rs.choice(n, size=min(sample_size, n), replace=False)

    model = variogram_params.get("model", "spherical")
    nugget = float(variogram_params.get("nugget", 0.0))
    sill = float(variogram_params.get("sill", float(np.var(values))))
    rng_ = float(variogram_params.get("range", 100.0))
    aniso = _anisotropic_scale(anisotropy, frame)

    preds = np.full(len(test_idx), np.nan)
    for k, i in enumerate(test_idx):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if method == "idw":
            e, _, _ = inverse_distance(xyz[i:i + 1], xyz[mask], values[mask],
                                       search_radius=rng_ * 1.5, anisotropy_matrix=aniso)
        else:
            e, _, _ = ordinary_kriging(xyz[i:i + 1], xyz[mask], values[mask],
                                       model, nugget, sill, rng_,
                                       search_radius=rng_ * 1.5, anisotropy_matrix=aniso)
        preds[k] = e[0]

    actual = values[test_idx]
    ok = np.isfinite(preds)
    if ok.sum() < 5:
        return {"performed": False, "reason": "too few cross-validation points could be estimated"}

    err = preds[ok] - actual[ok]
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual[ok] - actual[ok].mean()) ** 2))
    return {
        "performed": True,
        "n_tested": int(ok.sum()),
        "mean_error": float(err.mean()),
        "mean_absolute_error": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "correlation": float(np.corrcoef(preds[ok], actual[ok])[0, 1]) if ok.sum() > 2 else 0.0,
    }
