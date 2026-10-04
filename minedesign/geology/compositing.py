"""Downhole compositing, top-cut (grade capping) and geology back-tagging.

All conventional geostatistics with a settled formula:
  * composite length  = mode of the raw sample interval lengths
  * composite grade   = length-weighted average of the contributing samples
  * top cut           = 99th percentile of the composited population
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


def recommend_composite_length(assay: pd.DataFrame) -> dict:
    """Mode of the raw sample lengths, rounded to a sensible working length."""
    lengths = (assay["to"] - assay["from"]).to_numpy(float)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    if lengths.size == 0:
        return {"length": 1.0, "rationale": "no valid sample lengths; defaulted to 1.0 m"}

    rounded = np.round(lengths, 2)
    values, counts = np.unique(rounded, return_counts=True)
    mode = float(values[counts.argmax()])
    share = float(counts.max() / counts.sum())

    # snap to a clean working length when the mode is close to one
    for candidate in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0):
        if abs(mode - candidate) < 0.1 * candidate:
            mode = candidate
            break

    return {
        "length": mode,
        "modal_share": share,
        "mean_sample_length": float(lengths.mean()),
        "min_sample_length": float(lengths.min()),
        "max_sample_length": float(lengths.max()),
        "rationale": (f"modal raw sample length is {values[counts.argmax()]:.2f} m "
                      f"({share:.0%} of samples); composites are set to {mode:.2f} m"),
    }


def composite(assay: pd.DataFrame, length: float, min_recovery: float = 0.5) -> pd.DataFrame:
    """Fixed-length downhole compositing, length-weighted, per hole.

    Composites are cut on a regular grid from the top of each hole. A trailing
    composite shorter than ``min_recovery`` of the target length is merged into
    the previous one rather than left as an under-supported sample.
    """
    if length <= 0:
        raise ValueError("composite length must be positive")

    records: list[dict] = []
    for hole_id, grp in assay.groupby("hole_id", sort=False):
        grp = grp.sort_values("from")
        f = grp["from"].to_numpy(float)
        t = grp["to"].to_numpy(float)
        g = grp["grade"].to_numpy(float)
        if f.size == 0:
            continue

        start, end = float(f.min()), float(t.max())
        edges = np.arange(start, end + length * 0.999, length)
        if edges[-1] < end - 1e-9:
            edges = np.append(edges, end)
        edges[-1] = min(edges[-1], end)

        # merge a stub trailing composite into its predecessor
        if len(edges) > 2 and (edges[-1] - edges[-2]) < min_recovery * length:
            edges = np.delete(edges, -2)

        for i in range(len(edges) - 1):
            c_from, c_to = float(edges[i]), float(edges[i + 1])
            if c_to - c_from <= 1e-9:
                continue
            overlap = np.minimum(t, c_to) - np.maximum(f, c_from)
            overlap = np.where(overlap > 0, overlap, 0.0)
            valid = overlap > 0
            sampled = float(overlap[valid].sum())
            graded = overlap * np.isfinite(g)
            weight = float(graded.sum())
            grade = float(np.nansum(np.where(np.isfinite(g), g, 0.0) * overlap) / weight) if weight > 0 else np.nan
            records.append({
                "hole_id": hole_id,
                "from": c_from,
                "to": c_to,
                "length": c_to - c_from,
                "grade": grade,
                "sampled_length": sampled,
                "recovery": sampled / (c_to - c_from),
                "n_samples": int(valid.sum()),
            })

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise ValueError("compositing produced no intervals")

    dropped = int((out["recovery"] < min_recovery).sum())
    if dropped:
        log.info("%d composite(s) have under %.0f%% sample coverage and are flagged",
                 dropped, min_recovery * 100)
    out["low_recovery"] = out["recovery"] < min_recovery
    log.info("composited %d raw samples into %d intervals of %.2f m",
             len(assay), len(out), length)
    return out.reset_index(drop=True)


def recommend_top_cut(grade_stats: dict, percentile: float = 99.0) -> dict:
    """Top-cut at the 99th percentile, the standard first-pass convention.

    The recommendation also reports how much metal the cap removes, because a cap
    that strips a large share of contained metal is the signal to revisit it.
    """
    pct = grade_stats["percentiles"]
    value = float(pct.get(f"p{percentile:g}", grade_stats["max"]))
    return {
        "value": value,
        "percentile": percentile,
        "max_grade": grade_stats["max"],
        "rationale": (f"top cut set at the {percentile:g}th percentile "
                      f"({value:.3f}) following standard practice; the uncapped maximum is "
                      f"{grade_stats['max']:.3f}"),
    }


def apply_top_cut(composited: pd.DataFrame, value: float,
                  column: str = "grade") -> tuple[pd.DataFrame, dict]:
    """Cap grades at ``value``; returns the capped table and a capping summary."""
    out = composited.copy()
    raw = out[column].to_numpy(float)
    capped_mask = np.isfinite(raw) & (raw > value)

    out[f"{column}_uncapped"] = raw
    out[column] = np.where(capped_mask, value, raw)
    out["capped"] = capped_mask

    lengths = out["length"].to_numpy(float)
    finite = np.isfinite(raw)
    metal_before = float(np.sum(raw[finite] * lengths[finite]))
    metal_after = float(np.sum(out[column].to_numpy(float)[finite] * lengths[finite]))

    summary = {
        "top_cut_value": float(value),
        "n_capped": int(capped_mask.sum()),
        "pct_samples_capped": float(capped_mask.sum() / max(finite.sum(), 1) * 100.0),
        "metal_removed_pct": float((metal_before - metal_after) / metal_before * 100.0)
        if metal_before > 0 else 0.0,
        "mean_before": float(np.nanmean(raw)),
        "mean_after": float(np.nanmean(out[column].to_numpy(float))),
    }
    log.info("top cut %.3f capped %d composite(s), removing %.2f%% of contained metal",
             value, summary["n_capped"], summary["metal_removed_pct"])
    return out, summary


def tag_rock_type(intervals: pd.DataFrame, geology: pd.DataFrame) -> pd.DataFrame:
    """Assign each interval the rock type with the largest downhole overlap."""
    out = intervals.copy()
    out["rock_type"] = "UNKNOWN"
    if geology.empty:
        return out

    # dict(groupby) does not yield (name, group) pairs in pandas 3
    geo_by_hole = {hid: g for hid, g in geology.groupby("hole_id", sort=False)}  # noqa: C416
    codes = out["rock_type"].to_numpy(dtype=object)

    for hole_id, idx in out.groupby("hole_id", sort=False).indices.items():
        g = geo_by_hole.get(hole_id)
        if g is None or g.empty:
            continue
        gf = g["from"].to_numpy(float)
        gt = g["to"].to_numpy(float)
        gr = g["rock_type"].to_numpy(dtype=object)
        idx = np.asarray(idx)
        f = out["from"].to_numpy(float)[idx]
        t = out["to"].to_numpy(float)[idx]
        for k, (a, b) in enumerate(zip(f, t)):
            overlap = np.minimum(gt, b) - np.maximum(gf, a)
            if overlap.size and overlap.max() > 0:
                codes[idx[k]] = gr[int(overlap.argmax())]

    out["rock_type"] = codes
    return out
