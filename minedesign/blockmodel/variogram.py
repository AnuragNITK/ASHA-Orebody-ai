"""Experimental variography and model fitting.

Computing the experimental variogram and least-squares fitting the three standard
models is arithmetic, so it runs here in code. Which model to adopt, and whether
the data density supports kriging at all, is the judgement call handed to LLM
Call 1 -- with these fitted numbers as the evidence it reasons over.
"""
from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

MODELS = ("spherical", "exponential", "gaussian")


def variogram_model(h: np.ndarray, model: str, nugget: float, sill: float,
                    rng: float) -> np.ndarray:
    """Gamma(h) for the three standard models. ``sill`` is the partial sill."""
    h = np.asarray(h, float)
    rng = max(float(rng), 1e-9)
    if model == "spherical":
        ratio = np.minimum(h / rng, 1.0)
        gamma = sill * (1.5 * ratio - 0.5 * ratio ** 3)
    elif model == "exponential":
        gamma = sill * (1.0 - np.exp(-3.0 * h / rng))
    elif model == "gaussian":
        gamma = sill * (1.0 - np.exp(-3.0 * (h / rng) ** 2))
    else:
        raise ValueError(f"unknown variogram model: {model}")
    # gamma(0) is 0 by definition: the nugget is the discontinuity just off zero,
    # not a value at zero. Adding it at h=0 would also put the nugget on the
    # diagonal of the kriging matrix, which silently changes the estimator.
    return np.where(h > 0, nugget + gamma, 0.0)


def experimental_variogram(points: np.ndarray, values: np.ndarray,
                           lag: float | None = None, n_lags: int = 15,
                           max_pairs: int = 4_000_000,
                           direction: np.ndarray | None = None,
                           tolerance_deg: float = 22.5,
                           rng_seed: int = 0) -> dict:
    """Experimental semivariogram, omnidirectional or along one direction.

    Large sample sets are randomly subsampled to keep the pair count bounded;
    the subsample is seeded so a run is reproducible.
    """
    points = np.asarray(points, float)
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    points, values = points[ok], values[ok]

    n = len(points)
    if n < 10:
        raise ValueError("too few samples for a variogram")

    max_n = int(np.sqrt(2 * max_pairs)) + 1
    if n > max_n:
        idx = np.random.default_rng(rng_seed).choice(n, max_n, replace=False)
        points, values = points[idx], values[idx]
        n = max_n

    i, j = np.triu_indices(n, k=1)
    diff = points[i] - points[j]
    h = np.linalg.norm(diff, axis=1)
    gamma_pairs = 0.5 * (values[i] - values[j]) ** 2

    if direction is not None:
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_a = np.abs(diff @ d) / np.where(h > 0, h, np.nan)
        keep = cos_a >= np.cos(np.radians(tolerance_deg))
        keep &= np.isfinite(cos_a)
        h, gamma_pairs = h[keep], gamma_pairs[keep]

    if h.size == 0:
        raise ValueError("no sample pairs fall inside the search direction")

    max_h = float(np.percentile(h, 60))  # beyond ~half the field the variogram is noise
    if lag is None:
        lag = max_h / n_lags
    lag = max(float(lag), 1e-6)

    bins = np.arange(0, max_h + lag, lag)
    idx = np.digitize(h, bins) - 1
    valid = (idx >= 0) & (idx < len(bins) - 1)

    lag_h, lag_g, lag_n = [], [], []
    for b in range(len(bins) - 1):
        sel = valid & (idx == b)
        count = int(sel.sum())
        if count < 10:  # under-populated lags are unreliable, standard practice
            continue
        lag_h.append(float(h[sel].mean()))
        lag_g.append(float(gamma_pairs[sel].mean()))
        lag_n.append(count)

    return {
        "lag_distance": float(lag),
        "h": lag_h,
        "gamma": lag_g,
        "n_pairs": lag_n,
        "sample_variance": float(np.var(values, ddof=1)),
        "n_samples": int(n),
        "direction": None if direction is None else [float(v) for v in np.asarray(direction, float)],
    }


def fit_variogram_model(experimental: dict, model: str) -> dict:
    """Least-squares fit of one model, weighted by pair count.

    A coarse grid search followed by local refinement; the parameter space is
    three-dimensional and well behaved, so this is more robust here than a
    gradient optimiser that can walk into a negative sill.
    """
    h = np.asarray(experimental["h"], float)
    g = np.asarray(experimental["gamma"], float)
    w = np.asarray(experimental["n_pairs"], float)
    if h.size < 3:
        raise ValueError("not enough populated lags to fit a variogram model")

    var = experimental["sample_variance"]
    total_max = max(float(g.max()), var) * 1.5

    best = None
    nuggets = np.linspace(0.0, total_max * 0.6, 13)
    sills = np.linspace(total_max * 0.05, total_max, 20)
    ranges = np.linspace(h.min() if h.min() > 0 else h.max() / 10, h.max() * 2.0, 25)

    for nugget in nuggets:
        for sill in sills:
            for rng_ in ranges:
                pred = variogram_model(h, model, nugget, sill, rng_)
                sse = float(np.sum(w * (pred - g) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, nugget, sill, rng_)

    sse, nugget, sill, rng_ = best

    # local refinement around the grid optimum
    for _ in range(4):
        step_n = max(total_max * 0.02, 1e-9)
        step_s = max(total_max * 0.02, 1e-9)
        step_r = max(h.max() * 0.05, 1e-6)
        improved = True
        while improved:
            improved = False
            for dn, ds, dr in ((step_n, 0, 0), (-step_n, 0, 0), (0, step_s, 0),
                               (0, -step_s, 0), (0, 0, step_r), (0, 0, -step_r)):
                cand = (max(nugget + dn, 0.0), max(sill + ds, 1e-9), max(rng_ + dr, 1e-6))
                pred = variogram_model(h, model, *cand)
                cand_sse = float(np.sum(w * (pred - g) ** 2))
                if cand_sse < sse - 1e-12:
                    sse, (nugget, sill, rng_) = cand_sse, cand
                    improved = True
        step_n, step_s, step_r = step_n / 2, step_s / 2, step_r / 2

    pred = variogram_model(h, model, nugget, sill, rng_)
    ss_tot = float(np.sum(w * (g - np.average(g, weights=w)) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0
    total_sill = nugget + sill

    return {
        "model": model,
        "nugget": float(nugget),
        "sill": float(sill),              # partial sill
        "total_sill": float(total_sill),
        "range": float(rng_),
        "nugget_effect_ratio": float(nugget / total_sill) if total_sill > 0 else 0.0,
        "sse": float(sse),
        "r_squared": float(r2),
    }


def fit_all_models(experimental: dict) -> dict:
    """Fit all three standard models and report which fits best by weighted SSE."""
    fits = {}
    for model in MODELS:
        try:
            fits[model] = fit_variogram_model(experimental, model)
        except Exception as exc:  # noqa: BLE001
            log.warning("variogram fit failed for %s: %s", model, exc)
    if not fits:
        raise ValueError("no variogram model could be fitted")
    best = min(fits.values(), key=lambda f: f["sse"])
    log.info("best-fitting variogram: %s, range %.0f m, nugget ratio %.2f, R2 %.3f",
             best["model"], best["range"], best["nugget_effect_ratio"], best["r_squared"])
    return {"fits": fits, "best_fit": best}


def directional_analysis(points: np.ndarray, values: np.ndarray, frame,
                         lag: float | None = None) -> dict:
    """Variograms along strike, across strike and vertically, to expose anisotropy."""
    directions = {
        "along_strike": frame.strike_dir,
        "across_strike": frame.across_dir,
        "vertical": np.array([0.0, 0.0, 1.0]),
    }
    out = {}
    for name, d in directions.items():
        try:
            exp = experimental_variogram(points, values, lag=lag, direction=d)
            out[name] = {"experimental": exp, "fit": fit_all_models(exp)["best_fit"]}
        except Exception as exc:  # noqa: BLE001
            log.warning("directional variogram failed for %s: %s", name, exc)
    ranges = {k: v["fit"]["range"] for k, v in out.items()}
    if ranges:
        longest = max(ranges.values())
        out["anisotropy"] = {k: float(v / longest) for k, v in ranges.items()}
        out["ranges"] = ranges
    return out
