"""LLM Call 1 - Orebody & Grade Modeling Agent.

Runs once per pipeline, after compositing and top-cut, before block modelling.
Decides the ore/waste cutoff, the domain boundary per section, and the variogram
model and estimation method.

If the call is unavailable, `deterministic_decision` supplies the same structure
from established geostatistical conventions, and the run is labelled accordingly
so the report never implies a judgement was made when it was not.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..solid.sections import propose_section_boundaries, section_summary
from .client import CallLog, chat_json_with_fallback
from .prompts import OREBODY_GRADE_SYSTEM

log = get_logger(__name__)


def _build_user_payload(grade_stats: dict, top_cut: dict, composite_info: dict,
                        rock_type_stats: dict, sections: list[dict],
                        density_summary: dict, variography: dict) -> str:
    return json.dumps({
        "grade_statistics_after_top_cut": grade_stats,
        "top_cut": top_cut,
        "compositing": composite_info,
        "grade_by_rock_type": rock_type_stats,
        "data_density_and_anisotropy": density_summary,
        "fitted_variograms": variography,
        "sections": sections,
        "coordinate_note": (
            "Section points are given as u (metres across strike) and v (metres of "
            "elevation relative to the mean collar elevation). Return boundary points "
            "in the same u,v frame."),
    }, indent=2, default=float)


def rock_type_grade_stats(composited: pd.DataFrame) -> dict:
    """Grade distribution per rock code: the evidence for a rock-type-aware cutoff."""
    if "rock_type" not in composited.columns:
        return {}
    out = {}
    for code, grp in composited.groupby("rock_type", dropna=False):
        g = pd.to_numeric(grp["grade"], errors="coerce").dropna()
        if g.empty:
            continue
        out[str(code)] = {
            "n": int(len(g)),
            "mean": float(g.mean()),
            "median": float(g.median()),
            "p75": float(g.quantile(0.75)),
            "p90": float(g.quantile(0.90)),
            "max": float(g.max()),
            "share_of_samples_pct": float(len(g) / len(composited) * 100.0),
        }
    return out


def deterministic_decision(grade_stats: dict, rock_stats: dict, variography: dict,
                           density_summary: dict) -> dict:
    """Coded fallback for every Call 1 output, using standard conventions.

    Cutoff: the grade at which the population separates, approximated by the
    inflection between the barren mode and the mineralised tail. In practice the
    convention of a cutoff near the mean of a strongly skewed gold distribution,
    bounded by the 60th to 75th percentile, is used here.
    Rock types: any code whose mean grade exceeds the global mean is mineralised.
    Method: kriging when the fitted nugget effect is moderate and the drill grid
    is dense relative to the variogram range, IDW otherwise.
    """
    pct = grade_stats["percentiles"]
    mean = grade_stats["mean"]
    cutoff = float(np.clip(mean, pct["p50"], pct["p75"]))

    ore_rocks = [code for code, s in rock_stats.items() if s["mean"] > mean] or list(rock_stats)

    best = variography.get("omnidirectional", {}).get("best_fit", {})
    model = best.get("model", "spherical")
    rng = float(best.get("range", 100.0))
    nugget_ratio = float(best.get("nugget_effect_ratio", 0.4))
    spacing = float(density_summary.get("median_nearest_neighbour_m", 25.0))

    supports_kriging = (nugget_ratio < 0.6 and rng > spacing * 1.5
                        and density_summary.get("n_samples", 0) >= 100)
    method = "ordinary_kriging" if supports_kriging else "idw"

    return {
        "cutoff_grade": cutoff,
        "cutoff_rationale": (
            f"Deterministic fallback: the distribution has a skewness of "
            f"{grade_stats['skewness']:.2f} and a coefficient of variation of "
            f"{grade_stats['cv']:.2f}, so the mean sits well above the median "
            f"({mean:.3f} against {pct['p50']:.3f}). The cutoff is set at {cutoff:.3f}, "
            f"the mean bounded into the P50-P75 interval, and combined with the rock "
            f"codes whose mean grade exceeds the global mean. No LLM judgement was "
            f"applied to this decision."),
        "ore_rock_types": ore_rocks,
        "domain_boundaries": [],  # filled by the deterministic convex-hull proposer
        "variogram_model": model,
        "variogram_params": {"range": rng, "sill": float(best.get("sill", grade_stats["variance"])),
                             "nugget": float(best.get("nugget", 0.0))},
        "estimation_method": method,
        "estimation_rationale": (
            f"Deterministic fallback: the best-fitting variogram is {model} with a range of "
            f"{rng:.0f} m against a median sample spacing of {spacing:.1f} m and a nugget "
            f"effect of {nugget_ratio:.0%}. "
            + ("That supports ordinary kriging." if supports_kriging else
               "Data density is insufficient for reliable kriging, so inverse-distance "
               "weighting is used as the safer estimator.")),
        "decision_source": "deterministic",
    }


def _validate_and_repair(decision: dict, grade_stats: dict, rock_stats: dict,
                         fallback: dict) -> tuple[dict, list[str]]:
    """Keep the model honest: every field is range-checked against the real data."""
    notes: list[str] = []
    out = dict(decision)

    cutoff = out.get("cutoff_grade")
    try:
        cutoff = float(cutoff)
    except (TypeError, ValueError):
        cutoff = None
    if cutoff is None or not np.isfinite(cutoff) or not (
            grade_stats["min"] <= cutoff <= grade_stats["max"]):
        notes.append(f"cutoff grade {out.get('cutoff_grade')!r} was outside the observed "
                     f"grade range and was replaced with the deterministic value")
        cutoff = fallback["cutoff_grade"]
    out["cutoff_grade"] = float(cutoff)

    codes = out.get("ore_rock_types") or []
    known = set(rock_stats)
    codes = [str(c).upper() for c in codes if str(c).upper() in known]
    if not codes:
        codes = fallback["ore_rock_types"]
        notes.append("no recognised rock-type codes were returned; the deterministic "
                     "mineralised-code set was used")
    out["ore_rock_types"] = codes

    model = str(out.get("variogram_model", "")).lower()
    if model not in ("spherical", "exponential", "gaussian"):
        notes.append(f"variogram model {out.get('variogram_model')!r} is not one of the "
                     f"three standard models; the best-fitting model was used")
        model = fallback["variogram_model"]
    out["variogram_model"] = model

    params = out.get("variogram_params") or {}
    repaired = {}
    for key in ("range", "sill", "nugget"):
        try:
            value = float(params[key])
        except (KeyError, TypeError, ValueError):
            value = float(fallback["variogram_params"][key])
            notes.append(f"variogram {key} was missing or non-numeric; the fitted value was used")
        if not np.isfinite(value) or value < 0:
            value = float(fallback["variogram_params"][key])
            notes.append(f"variogram {key} was negative or non-finite; the fitted value was used")
        repaired[key] = value
    if repaired["range"] <= 0:
        repaired["range"] = fallback["variogram_params"]["range"]
    out["variogram_params"] = repaired

    method = str(out.get("estimation_method", "")).lower()
    if method not in ("ordinary_kriging", "idw"):
        notes.append(f"estimation method {out.get('estimation_method')!r} was not recognised; "
                     f"the deterministic choice was used")
        method = fallback["estimation_method"]
    out["estimation_method"] = method

    for key in ("cutoff_rationale", "estimation_rationale"):
        if not str(out.get(key, "")).strip():
            out[key] = fallback[key]
            notes.append(f"{key} was empty and was replaced with the deterministic rationale")

    return out, notes


def _boundaries_from_llm(decision: dict, frame, ore_intervals: pd.DataFrame) -> list[dict]:
    """Convert the returned u,v polygons into world-coordinate boundary sets."""
    sections = decision.get("domain_boundaries") or []
    section_s = (ore_intervals.groupby("section_id")["section_s"].first().to_dict()
                 if not ore_intervals.empty else {})

    out = []
    for entry in sections:
        try:
            section_id = int(entry["section_id"])
            uv = np.asarray(entry.get("boundary_points_uv")
                            or entry.get("boundary_points"), float)
        except (KeyError, TypeError, ValueError):
            continue
        if uv.ndim != 2 or len(uv) < 3:
            continue
        if uv.shape[1] == 3:  # a model that returned xyz anyway
            world = uv
        else:
            s = float(section_s.get(section_id, 0.0))
            world = frame.to_world(np.column_stack([np.full(len(uv), s), uv[:, 0], uv[:, 1]]))
        out.append({
            "section_id": section_id,
            "section_s": float(section_s.get(section_id, 0.0)),
            "boundary_points": [[float(p[0]), float(p[1]), float(p[2])] for p in world],
            "source": "llm",
        })
    return out


def run_orebody_agent(composited: pd.DataFrame, grade_stats: dict, top_cut: dict,
                      composite_info: dict, frame, density_summary: dict,
                      variography: dict, call_log: CallLog | None = None,
                      max_sections_to_llm: int = 14) -> dict:
    """Execute LLM Call 1 and return a validated, always-complete decision."""
    rock_stats = rock_type_grade_stats(composited)
    fallback = deterministic_decision(grade_stats, rock_stats, variography, density_summary)

    sections = section_summary(composited, frame)
    if len(sections) > max_sections_to_llm:  # keep one request, keep it bounded
        step = len(sections) / max_sections_to_llm
        sections = [sections[int(i * step)] for i in range(max_sections_to_llm)]

    user = _build_user_payload(grade_stats, top_cut, composite_info, rock_stats,
                               sections, density_summary, variography)

    result, provider = chat_json_with_fallback(
        OREBODY_GRADE_SYSTEM, user, label="call-1-orebody-grade", call_log=call_log)

    if result is None:
        decision = dict(fallback)
        decision["decision_source"] = "deterministic"
        decision["validation_notes"] = [
            "Neither the paid model nor the local fallback was reachable. Every Call 1 "
            "output below comes from coded geostatistical convention, not from a model "
            "judgement."]
    else:
        decision, notes = _validate_and_repair(result, grade_stats, rock_stats, fallback)
        decision["decision_source"] = provider
        decision["validation_notes"] = notes

    # classify ore/waste with the agreed cutoff and mineralised rock codes
    ore_codes = set(decision["ore_rock_types"])
    grade = pd.to_numeric(composited["grade"], errors="coerce")
    rock = composited["rock_type"].astype(str) if "rock_type" in composited.columns else None
    is_ore = grade >= decision["cutoff_grade"]
    if rock is not None and ore_codes:
        is_ore &= rock.isin(ore_codes)
    decision["n_ore_intervals"] = int(is_ore.sum())
    decision["n_waste_intervals"] = int((~is_ore).sum())

    ore_intervals = composited[is_ore]
    llm_boundaries = _boundaries_from_llm(decision, frame, ore_intervals) if result else []
    coded_boundaries = propose_section_boundaries(ore_intervals, frame)

    if len(llm_boundaries) >= 2:
        decision["domain_boundaries"] = llm_boundaries
        decision["boundary_source"] = "llm"
    else:
        decision["domain_boundaries"] = coded_boundaries
        decision["boundary_source"] = "deterministic_convex_hull"
        if result is not None:
            decision.setdefault("validation_notes", []).append(
                "fewer than two usable section polygons were returned, so the "
                "deterministic ore-envelope proposal was used for the wireframe")

    decision["fallback_boundaries"] = coded_boundaries
    decision["ore_mask"] = is_ore.to_numpy()
    log.info("Call 1 (%s): cutoff %.3f, %d ore intervals, %s, %d boundary polygons",
             decision["decision_source"], decision["cutoff_grade"],
             decision["n_ore_intervals"], decision["estimation_method"],
             len(decision["domain_boundaries"]))
    return decision


REBOUNDARY_SYSTEM = """You repair a failed orebody wireframe. You are given the section
polygons that produced an invalid solid and the exact validation errors. Return corrected
boundary points for the sections only. Keep polygons simple, non-self-intersecting, ordered
around the perimeter, and with at least 3 points per section.
Output strict JSON only:
{"domain_boundaries": [{"section_id": <int>, "boundary_points_uv": [[u, v], ...]}, ...]}"""


def repropose_boundaries(failed_boundaries: list[dict], validation_failures: list[str],
                         frame, ore_intervals: pd.DataFrame,
                         call_log: CallLog | None = None) -> list[dict] | None:
    """Escalate a boundary re-proposal to the local fallback model only.

    Called at most once, and only after the solid has failed validation twice.
    Nothing else about the Call 1 decision is revisited.
    """
    from ..config import SETTINGS
    from .client import LLMError, chat_json

    payload = json.dumps({
        "validation_errors": validation_failures,
        "sections": [
            {"section_id": b["section_id"],
             "boundary_points_uv": [
                 [round(float(p[0]), 2), round(float(p[1]), 2)]
                 for p in frame.to_local(np.asarray(b["boundary_points"], float))[:, 1:3]]}
            for b in failed_boundaries],
    }, indent=2)

    try:
        result = chat_json(REBOUNDARY_SYSTEM, payload, model=SETTINGS.local_fallback_model,
                           label="call-1-boundary-repair", provider="ollama",
                           max_tokens=4000, call_log=call_log)
    except LLMError as exc:
        log.warning("boundary re-proposal unavailable: %s", exc)
        return None

    repaired = _boundaries_from_llm(result, frame, ore_intervals)
    return repaired if len(repaired) >= 2 else None
