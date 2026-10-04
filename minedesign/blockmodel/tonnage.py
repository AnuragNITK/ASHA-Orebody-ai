"""Tonnage, contained metal, and grade-tonnage curves.

Straight arithmetic on the block model: tonnes = volume x density, metal =
tonnes x grade. Reported per rock type and per confidence class so the numbers
can be audited rather than taken on trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

# grams per tonne to ounces: 1 troy ounce = 31.1034768 g
G_PER_TONNE_TO_OZ = 1.0 / 31.1034768


def assign_density(blocks: pd.DataFrame, density: float | dict,
                   rock_column: str = "rock_type") -> pd.DataFrame:
    """Attach a bulk density per block: a single value, or one per rock type."""
    out = blocks.copy()
    if isinstance(density, dict):
        default = float(density.get("default", np.mean(
            [v for v in density.values() if isinstance(v, (int, float))])))
        codes = out[rock_column] if rock_column in out.columns else pd.Series("UNKNOWN", index=out.index)
        out["density"] = codes.map(lambda c: float(density.get(str(c), default)))
    else:
        out["density"] = float(density)
    return out


def compute_tonnage(blocks: pd.DataFrame, density: float | dict,
                    block_volume: float | None = None,
                    ore_column: str = "is_ore", grade_column: str = "grade") -> dict:
    """Tonnage and contained-metal summary for the block model."""
    if block_volume is None:
        block_volume = float(blocks.attrs.get("block_volume", 0.0))
    if not block_volume:
        raise ValueError("block volume is required to compute tonnage")

    b = assign_density(blocks, density)
    b["volume"] = block_volume
    b["tonnes"] = b["volume"] * b["density"]

    grade = b[grade_column].to_numpy(float)
    b["contained_metal_g"] = np.where(np.isfinite(grade), grade, 0.0) * b["tonnes"]

    ore = b[b[ore_column]] if ore_column in b.columns else b
    waste = b[~b[ore_column]] if ore_column in b.columns else b.iloc[0:0]

    ore_tonnes = float(ore["tonnes"].sum())
    ore_metal_g = float(ore["contained_metal_g"].sum())
    mean_grade = ore_metal_g / ore_tonnes if ore_tonnes > 0 else 0.0

    summary = {
        "block_volume_m3": float(block_volume),
        "n_blocks_total": int(len(b)),
        "n_blocks_ore": int(len(ore)),
        "n_blocks_waste": int(len(waste)),
        "ore_volume_m3": float(ore["volume"].sum()),
        "ore_tonnes": ore_tonnes,
        "waste_tonnes": float(waste["tonnes"].sum()),
        "total_tonnes": float(b["tonnes"].sum()),
        "mean_ore_grade": float(mean_grade),
        "contained_metal_g": ore_metal_g,
        "contained_metal_oz": float(ore_metal_g * G_PER_TONNE_TO_OZ),
        "contained_metal_t": float(ore_metal_g / 1e6),
        "density_used": density if isinstance(density, dict) else float(density),
        "by_rock_type": {},
        "by_confidence": {},
    }

    if "rock_type" in ore.columns and not ore.empty:
        grouped = ore.groupby("rock_type", dropna=False)
        for code, grp in grouped:
            t = float(grp["tonnes"].sum())
            m = float(grp["contained_metal_g"].sum())
            summary["by_rock_type"][str(code)] = {
                "n_blocks": int(len(grp)), "tonnes": t,
                "mean_grade": float(m / t) if t > 0 else 0.0,
                "contained_metal_oz": float(m * G_PER_TONNE_TO_OZ),
            }

    if "confidence" in ore.columns and not ore.empty:
        for code, grp in ore.groupby("confidence", dropna=False):
            t = float(grp["tonnes"].sum())
            m = float(grp["contained_metal_g"].sum())
            summary["by_confidence"][str(code)] = {
                "n_blocks": int(len(grp)), "tonnes": t,
                "mean_grade": float(m / t) if t > 0 else 0.0,
                "contained_metal_oz": float(m * G_PER_TONNE_TO_OZ),
            }

    log.info("tonnage: %.2f Mt of ore at %.3f mean grade, %.0f oz contained",
             ore_tonnes / 1e6, mean_grade, summary["contained_metal_oz"])
    return summary


def grade_tonnage_curve(blocks: pd.DataFrame, density: float | dict,
                        block_volume: float, cutoffs: np.ndarray | None = None,
                        grade_column: str = "grade") -> dict:
    """Tonnes and mean grade above a sweep of cutoffs: the classic sensitivity view."""
    b = assign_density(blocks, density)
    grade = b[grade_column].to_numpy(float)
    tonnes = block_volume * b["density"].to_numpy(float)

    finite = np.isfinite(grade)
    grade, tonnes = grade[finite], tonnes[finite]
    if grade.size == 0:
        return {"cutoffs": [], "tonnes": [], "mean_grade": [], "contained_metal_oz": []}

    if cutoffs is None:
        top = float(np.percentile(grade, 99.5))
        cutoffs = np.linspace(0.0, max(top, 1e-6), 25)

    rows = {"cutoffs": [], "tonnes": [], "mean_grade": [], "contained_metal_oz": []}
    for c in cutoffs:
        sel = grade >= c
        t = float(tonnes[sel].sum())
        m = float((grade[sel] * tonnes[sel]).sum())
        rows["cutoffs"].append(float(c))
        rows["tonnes"].append(t)
        rows["mean_grade"].append(float(m / t) if t > 0 else 0.0)
        rows["contained_metal_oz"].append(float(m * G_PER_TONNE_TO_OZ))
    return rows
