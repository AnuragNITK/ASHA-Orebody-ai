"""2D chart payloads for the report body.

Statistical figures stay 2D on purpose. A grade histogram or a variogram forced
into a 3D scene is harder to read, not better, so distributions live in charts
and only genuinely spatial objects go into the viewer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geology.stats import histogram_data


def grade_histogram_chart(values: np.ndarray, bins: int = 30) -> dict:
    """Histogram with a fitted lognormal curve overlaid."""
    data = histogram_data(np.asarray(values, float), bins=bins)
    return {
        "labels": [round(c, 4) for c in data["bin_centres"]],
        "counts": data["counts"],
        "fitted": data["fitted_curve"],
        "bin_edges": data["bin_edges"],
    }


def cumulative_frequency_chart(values: np.ndarray, bins: int = 30) -> dict:
    """Frequency bars with the cumulative percentage line."""
    data = histogram_data(np.asarray(values, float), bins=bins)
    return {
        "labels": [round(c, 4) for c in data["bin_centres"]],
        "counts": data["counts"],
        "cumulative_pct": [round(v, 3) for v in data["cumulative_pct"]],
    }


def variogram_chart(variography: dict) -> dict:
    """Experimental points with the fitted model curve, per direction."""
    from ..blockmodel.variogram import variogram_model

    series = []
    for name, entry in variography.items():
        if not isinstance(entry, dict) or "experimental" not in entry:
            continue
        exp = entry["experimental"]
        fit = entry.get("fit") or {}
        if not exp.get("h"):
            continue
        h = np.asarray(exp["h"], float)
        curve_h = np.linspace(0, float(h.max()) * 1.1, 60)
        curve = []
        if fit:
            curve = variogram_model(curve_h, fit["model"], fit["nugget"],
                                    fit["sill"], fit["range"]).tolist()
        series.append({
            "name": name.replace("_", " "),
            "experimental": [{"x": round(float(a), 2), "y": round(float(b), 6)}
                             for a, b in zip(exp["h"], exp["gamma"])],
            "n_pairs": exp["n_pairs"],
            "model_curve": [{"x": round(float(a), 2), "y": round(float(b), 6)}
                            for a, b in zip(curve_h, curve)],
            "fit": fit,
        })
    return {"series": series}


def grade_tonnage_chart(curve: dict) -> dict:
    return {
        "cutoffs": [round(c, 4) for c in curve.get("cutoffs", [])],
        "tonnes": [round(t, 1) for t in curve.get("tonnes", [])],
        "mean_grade": [round(g, 5) for g in curve.get("mean_grade", [])],
        "contained_metal_oz": [round(m, 1) for m in curve.get("contained_metal_oz", [])],
    }


def nested_shells_chart(shells: list[dict]) -> dict:
    return {
        "revenue_factors": [s["revenue_factor"] for s in shells],
        "ore_tonnes": [round(s["ore_tonnes"], 1) for s in shells],
        "waste_tonnes": [round(s["waste_tonnes"], 1) for s in shells],
        "strip_ratio": [None if not np.isfinite(s["strip_ratio"]) else round(s["strip_ratio"], 3)
                        for s in shells],
        "pit_value": [round(s["pit_value"], 1) for s in shells],
        "mean_ore_grade": [round(s["mean_ore_grade"], 5) for s in shells],
    }


def topcut_comparison_chart(capped: pd.DataFrame) -> dict:
    """Uncapped against capped grades, so the effect of the cap is visible."""
    if "grade_uncapped" not in capped.columns:
        return {"labels": [], "before": [], "after": []}
    before = pd.to_numeric(capped["grade_uncapped"], errors="coerce").dropna().to_numpy(float)
    after = pd.to_numeric(capped["grade"], errors="coerce").dropna().to_numpy(float)
    if before.size == 0:
        return {"labels": [], "before": [], "after": []}

    edges = np.histogram_bin_edges(before, bins=30)
    b_counts, _ = np.histogram(before, bins=edges)
    a_counts, _ = np.histogram(after, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    return {
        "labels": [round(float(c), 4) for c in centres],
        "before": [int(c) for c in b_counts],
        "after": [int(c) for c in a_counts],
    }


def cross_validation_chart(cv: dict) -> dict:
    return {k: v for k, v in (cv or {}).items() if k != "performed"} if cv else {}
