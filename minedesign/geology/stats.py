"""Descriptive grade statistics and the percentile colour ramp.

Both are formulas, not judgement calls, so neither goes near an LLM. The stats
block computed here is also the evidence package handed to LLM Call 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]

# A perceptually ordered ramp, cold to hot, used for grade everywhere in the
# report so the 3D scene, the legend and the charts agree. The hot end is kept
# deliberately dark rather than bright, so the highest-grade blocks read as
# visually heavier -- the eye should land on high grade, not on bright yellow.
GRADE_RAMP = ["#1a2a5e", "#1f4a7a", "#2a6f9a", "#3d9c8a", "#6ba84a",
              "#d4a03a", "#d4702a", "#b8402a", "#8b1a1a"]


def compute_grade_stats(assay: pd.DataFrame, column: str = "grade",
                        weight_column: str | None = None) -> dict:
    """Descriptive statistics, optionally length-weighted.

    Length weighting matters: unequal sample supports otherwise bias the mean.
    """
    values = pd.to_numeric(assay[column], errors="coerce").to_numpy(float)
    mask = np.isfinite(values)
    values = values[mask]
    if values.size == 0:
        raise ValueError(f"column {column} has no finite values")

    weights = None
    if weight_column and weight_column in assay.columns:
        w = pd.to_numeric(assay[weight_column], errors="coerce").to_numpy(float)[mask]
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
        if np.isfinite(w).any():
            weights = np.nan_to_num(w, nan=np.nanmedian(w))

    n = int(values.size)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    positive = values[values > 0]
    geo_mean = float(np.exp(np.mean(np.log(positive)))) if positive.size else 0.0

    centred = values - mean
    m2 = float(np.mean(centred ** 2))
    skew = float(np.mean(centred ** 3) / m2 ** 1.5) if m2 > 0 else 0.0
    kurt = float(np.mean(centred ** 4) / m2 ** 2 - 3.0) if m2 > 0 else 0.0

    pct = {f"p{p:g}": float(np.percentile(values, p)) for p in PERCENTILES}

    stats = {
        "column": column,
        "n": n,
        "n_missing": int((~mask).sum()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": mean,
        "median": float(np.median(values)),
        "geometric_mean": geo_mean,
        "std_dev": std,
        "variance": float(std ** 2),
        "cv": float(std / mean) if mean else 0.0,
        "skewness": skew,
        "kurtosis_excess": kurt,
        "percentiles": pct,
        "n_zero": int((values == 0).sum()),
    }
    if weights is not None:
        stats["length_weighted_mean"] = float(np.average(values, weights=weights))
    return stats


def choose_color_breakpoints(grade_stats: dict, n_classes: int = 9) -> dict:
    """Percentile-based colour ramp breakpoints.

    Formula, not judgement: equal-count classes between the 5th and 99th
    percentile keep a skewed grade distribution readable, which equal-interval
    classes on a lognormal population do not.
    """
    n_classes = max(3, min(n_classes, len(GRADE_RAMP)))
    pct = grade_stats["percentiles"]
    lo, hi = pct["p5"], pct["p99"]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = grade_stats["min"], max(grade_stats["max"], grade_stats["min"] + 1e-9)

    qs = np.linspace(5, 99, n_classes + 1)
    breaks = [float(np.interp(q, PERCENTILES, [pct[f"p{p:g}"] for p in PERCENTILES])) for q in qs]
    breaks[0], breaks[-1] = float(lo), float(hi)
    breaks = sorted({round(b, 6) for b in breaks})
    while len(breaks) < n_classes + 1:  # degenerate distribution guard
        breaks.append(breaks[-1] + 1e-6)

    colors = GRADE_RAMP[:len(breaks) - 1]
    return {
        "breaks": breaks,
        "colors": colors,
        "min": float(grade_stats["min"]),
        "max": float(grade_stats["max"]),
        "method": "equal-count (percentile) classes between P5 and P99",
    }


def color_for_grade(value: float, ramp: dict) -> str:
    breaks, colors = ramp["breaks"], ramp["colors"]
    if not np.isfinite(value):
        return "#8a8a8a"
    for i in range(len(colors)):
        if value <= breaks[i + 1]:
            return colors[i]
    return colors[-1]


ROCK_PALETTE = ["#8d6e63", "#4db6ac", "#7986cb", "#ffb74d", "#a1887f",
                "#90a4ae", "#ba68c8", "#4dd0e1", "#aed581", "#e57373"]


def rock_type_colors(rock_types) -> dict[str, str]:
    """Stable, deterministic colour per rock code (sorted, so runs are repeatable)."""
    codes = sorted({str(r) for r in rock_types if pd.notna(r)})
    return {code: ROCK_PALETTE[i % len(ROCK_PALETTE)] for i, code in enumerate(codes)}


def histogram_data(values: np.ndarray, bins: int = 30) -> dict:
    """Histogram plus a fitted lognormal curve and the cumulative distribution."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    cumulative = np.cumsum(counts) / max(counts.sum(), 1) * 100.0

    positive = values[values > 0]
    fitted = None
    if positive.size > 10:
        mu = float(np.mean(np.log(positive)))
        sigma = float(np.std(np.log(positive), ddof=1))
        if sigma > 0:
            width = float(edges[1] - edges[0])
            safe = np.where(centres > 0, centres, np.nan)
            pdf = (1.0 / (safe * sigma * np.sqrt(2 * np.pi))) * np.exp(
                -((np.log(safe) - mu) ** 2) / (2 * sigma ** 2))
            scaled = np.nan_to_num(pdf * positive.size * width)
            fitted = {"model": "lognormal", "mu": mu, "sigma": sigma,
                      "values": [float(v) for v in scaled]}

    return {
        "bin_centres": [float(c) for c in centres],
        "bin_edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "cumulative_pct": [float(c) for c in cumulative],
        "fitted_curve": fitted,
    }
