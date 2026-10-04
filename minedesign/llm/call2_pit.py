"""LLM Call 2 - Pit Design & Report Agent.

Runs once per pipeline, after the block model is finalized. Decides the pit
economics and writes the engineering narrative. The pit itself is then optimized
in code with those economics, and every figure in the report tables comes from
that tool output rather than from the model.
"""
from __future__ import annotations

import json

import numpy as np

from ..logging_utils import get_logger
from ..pit.lerchs_grossmann import EconomicParameters
from .client import CallLog, chat_json_with_fallback
from .prompts import PIT_REPORT_SYSTEM

log = get_logger(__name__)

# Neutral first-pass economics. Deliberately generic: these are the values the
# preliminary shell is computed with so the agent has something real to reason
# about, and every one of them is reported as an assumption if it survives.
DEFAULT_ECONOMICS = {
    "commodity": "gold",
    "commodity_price": 2000.0,
    "price_unit": "USD/oz",
    "mining_cost_per_tonne": 3.0,
    "processing_cost_per_tonne": 25.0,
    "ga_cost_per_tonne": 4.0,
    "recovery_pct": 90.0,
    "overall_slope_angle_deg": 45.0,
    "discount_rate_pct": 8.0,
}

_NUMERIC_BOUNDS = {
    "commodity_price": (0.01, 1e7),
    "mining_cost_per_tonne": (0.01, 500.0),
    "processing_cost_per_tonne": (0.01, 2000.0),
    "ga_cost_per_tonne": (0.0, 500.0),
    "recovery_pct": (1.0, 100.0),
    "overall_slope_angle_deg": (10.0, 85.0),
    "discount_rate_pct": (0.0, 40.0),
}

REPORT_SECTIONS = ("executive_summary", "methodology", "data_quality_notes",
                   "orebody_geometry", "grade_statistics_discussion",
                   "domaining_rationale", "block_model_summary",
                   "pit_optimization_discussion", "risks_and_limitations",
                   "recommendations")


def _deterministic_report(context: dict, econ: dict) -> dict:
    """Factual narrative assembled from tool outputs when no model is reachable.

    Deliberately plain: it states what was computed and by what convention, and
    it says outright that no engineering judgement was applied.
    """
    ingest = context.get("ingestion", {})
    stats = context.get("grade_statistics_after_top_cut", {})
    solid = context.get("solid_validation", {})
    bm = context.get("block_model", {})
    prelim = context.get("preliminary_pit", {})
    call1 = context.get("call_1_decision", {})

    def num(value, fmt="{:,.2f}", default="not available"):
        try:
            return fmt.format(float(value))
        except (TypeError, ValueError):
            return default

    return {
        "executive_summary": (
            f"This report was generated without the judgement steps of the pipeline: no "
            f"language model was reachable, so every decision below comes from coded "
            f"geostatistical convention. The campaign comprises {ingest.get('n_holes', 'an unknown number of')} "
            f"drillholes with {ingest.get('n_assay_intervals', 0):,} assay intervals. A cutoff of "
            f"{num(call1.get('cutoff_grade'), '{:.3f}')} separates ore from waste, producing an "
            f"orebody solid of {num(solid.get('volume_m3'), '{:,.0f}')} cubic metres. The block "
            f"model holds {bm.get('n_blocks_total', 0):,} blocks, of which "
            f"{bm.get('n_blocks_ore', 0):,} are classified as ore. A preliminary pit at default "
            f"economics contains {num(prelim.get('ore_tonnes', 0) / 1e6 if prelim.get('ore_tonnes') else 0, '{:,.2f}')} "
            f"million tonnes of ore at a strip ratio of {num(prelim.get('strip_ratio'), '{:.2f}')}. "
            f"These figures are a screening estimate and require review by a competent person."),
        "methodology": (
            "The workflow follows the standard open-pit resource sequence. Collar, survey, assay "
            "and geology files were loaded and schema-checked. Hole traces were desurveyed by "
            "minimum curvature. Raw samples were composited to the modal sample length by "
            "length-weighted averaging, then top-cut at the 99th percentile. Ore intervals were "
            "flagged by grade and rock type, an ore envelope was digitised per drill section, and "
            "the sections were lofted into a closed triangulated solid which was validated for "
            "watertightness, connectivity, duplicates, non-manifold edges, self-intersection and "
            "consistent winding. A regular block grid at roughly half the drill spacing was "
            "populated by the chosen estimator, blocks were classified against the solid by ray "
            "casting, and the ultimate pit was found as the maximum-weight closure of the block "
            "precedence graph."),
        "data_quality_notes": (
            "Data-quality flags raised during ingestion: "
            + ("; ".join(f"[{f['severity']}] {f['file']}: {f['message']}"
                         + (f" ({f['count']} records)" if f.get("count") else "")
                         for f in ingest.get("flags", []))
               or "none raised.")),
        "orebody_geometry": (
            f"The wireframe encloses {num(solid.get('volume_m3'), '{:,.0f}')} cubic metres with a "
            f"surface area of {num(solid.get('surface_area_m2'), '{:,.0f}')} square metres, built "
            f"from {solid.get('n_vertices', 0):,} vertices and {solid.get('n_faces', 0):,} "
            f"triangles. Validation "
            + ("passed every check." if solid.get("is_valid")
               else "failed: " + "; ".join(solid.get("failures", [])))),
        "grade_statistics_discussion": (
            f"After compositing and top-cut the population has a mean of "
            f"{num(stats.get('mean'), '{:.3f}')} against a median of {num(stats.get('median'), '{:.3f}')}, "
            f"a coefficient of variation of {num(stats.get('cv'), '{:.2f}')}, and a skewness of "
            f"{num(stats.get('skewness'), '{:.2f}')}. A mean well above the median with high positive "
            f"skewness is the signature of a lognormal grade distribution, which is why the top cut "
            f"and the percentile-based colour classes are applied."),
        "domaining_rationale": str(call1.get("cutoff_rationale", "not available")),
        "block_model_summary": (
            f"The model contains {bm.get('n_blocks_total', 0):,} blocks, "
            f"{bm.get('n_blocks_ore', 0):,} of them inside the orebody solid, holding "
            f"{num(bm.get('ore_tonnes', 0) / 1e6 if bm.get('ore_tonnes') else 0, '{:,.2f}')} million "
            f"tonnes at a mean grade of {num(bm.get('mean_ore_grade'), '{:.3f}')}. Grades were "
            f"estimated by {call1.get('estimation_method', 'the selected estimator')}."),
        "pit_optimization_discussion": (
            "The ultimate pit is the maximum-weight closure of the block precedence graph, solved "
            "exactly by minimum cut, which returns the same optimum as the Lerchs-Grossmann "
            "algorithm. Because no model was reachable, the economics used are the neutral "
            "defaults listed in the assumptions table, and they are not calibrated to this deposit, "
            "this jurisdiction or any real cost study. The nested shells show how the pit responds "
            "to price and are the more informative result at this stage."),
        "risks_and_limitations": (
            "This run had no access to the judgement steps of the pipeline, so the cutoff grade, "
            "the estimation method and the pit economics are all conventions rather than "
            "engineering decisions. The estimate is not compliant with JORC or NI 43-101 and "
            "carries no competent-person sign-off. Grade estimation quality is bounded by drill "
            "spacing, and the block model reports an estimation variance per block which should be "
            "read alongside the grades."),
        "recommendations": (
            "Have a mining engineer review the cutoff grade and the mineralised rock-type set. "
            "Replace the default economics with a costed study before drawing any investment "
            "conclusion. Confirm the bulk density by measurement rather than assumption. "
            "Review the variogram and search neighbourhood against the drill spacing. "
            "Infill drill where the block model reports low sample support."),
    }


def _validate_economics(raw: dict, user_supplied: dict | None) -> tuple[dict, list[str]]:
    """Range-check the returned economics; user-supplied values always win."""
    notes: list[str] = []
    econ = dict(DEFAULT_ECONOMICS)

    for key, value in (raw or {}).items():
        if key not in econ and key not in ("slope_angles_by_sector", "price_unit", "commodity"):
            continue
        econ[key] = value

    for key, (lo, hi) in _NUMERIC_BOUNDS.items():
        try:
            value = float(econ.get(key))
        except (TypeError, ValueError):
            value = float(DEFAULT_ECONOMICS[key])
            notes.append(f"{key} was missing or non-numeric; the default was used")
        if not np.isfinite(value) or not (lo <= value <= hi):
            notes.append(f"{key} of {econ.get(key)!r} is outside the plausible range "
                         f"[{lo}, {hi}]; the default {DEFAULT_ECONOMICS[key]} was used")
            value = float(DEFAULT_ECONOMICS[key])
        econ[key] = value

    econ["commodity"] = str(econ.get("commodity") or "gold")
    econ["price_unit"] = str(econ.get("price_unit") or "USD/oz")

    sectors = econ.get("slope_angles_by_sector") or {}
    if isinstance(sectors, dict):
        clean = {}
        for name, angle in sectors.items():
            try:
                a = float(angle)
            except (TypeError, ValueError):
                continue
            if 10.0 <= a <= 85.0:
                clean[str(name)] = a
        econ["slope_angles_by_sector"] = clean
    else:
        econ["slope_angles_by_sector"] = {}

    # user-supplied values are facts, not proposals
    for key, value in (user_supplied or {}).items():
        if key in econ and value is not None:
            econ[key] = value
            notes.append(f"{key} was set from user input, overriding the proposed value")

    return econ, notes


def _build_assumptions(econ: dict, raw_assumptions, user_supplied: dict | None) -> list[dict]:
    """Every economic input, labelled by where it actually came from."""
    user_keys = set((user_supplied or {}).keys())
    basis_by_param = {}
    for item in raw_assumptions or []:
        if isinstance(item, dict) and item.get("parameter"):
            basis_by_param[str(item["parameter"])] = str(item.get("basis", ""))

    rows = []
    for key in ("commodity", "commodity_price", "mining_cost_per_tonne",
                "processing_cost_per_tonne", "ga_cost_per_tonne", "recovery_pct",
                "overall_slope_angle_deg", "discount_rate_pct"):
        rows.append({
            "parameter": key,
            "value": econ.get(key),
            "source": "user_supplied" if key in user_keys else "assumed_default",
            "basis": basis_by_param.get(key, "" if key in user_keys
                                        else "proposed value, not a costed study"),
        })
    return rows


def run_pit_report_agent(context: dict, user_economics: dict | None = None,
                         call_log: CallLog | None = None) -> dict:
    """Execute LLM Call 2 and return validated economics plus the report prose."""
    payload = dict(context)
    payload["user_supplied_economics"] = user_economics or {}
    user = json.dumps(payload, indent=2, default=_json_default)

    result, provider = chat_json_with_fallback(
        PIT_REPORT_SYSTEM, user, label="call-2-pit-report", call_log=call_log,
        max_tokens=12000)

    if result is None:
        econ, notes = _validate_economics({}, user_economics)
        report = _deterministic_report(context, econ)
        assumptions = _build_assumptions(econ, [], user_economics)
        notes.insert(0, "Neither the paid model nor the local fallback was reachable. "
                        "The economics are neutral defaults and the narrative is "
                        "assembled from tool outputs only.")
        source = "deterministic"
    else:
        econ, notes = _validate_economics(result.get("economics") or {}, user_economics)
        assumptions = _build_assumptions(econ, result.get("assumptions"), user_economics)
        report = result.get("report") or {}
        if not isinstance(report, dict):
            report = {}
        fallback_report = _deterministic_report(context, econ)
        missing = [s for s in REPORT_SECTIONS if not str(report.get(s, "")).strip()]
        for section in missing:
            report[section] = fallback_report[section]
        if missing:
            notes.append(f"report section(s) {missing} were empty and were filled from "
                         f"tool outputs")
        source = provider

    economics = EconomicParameters(
        commodity=econ["commodity"],
        commodity_price=float(econ["commodity_price"]),
        price_unit=econ["price_unit"],
        mining_cost_per_tonne=float(econ["mining_cost_per_tonne"]),
        processing_cost_per_tonne=float(econ["processing_cost_per_tonne"]),
        ga_cost_per_tonne=float(econ["ga_cost_per_tonne"]),
        recovery_pct=float(econ["recovery_pct"]),
        overall_slope_angle_deg=float(econ["overall_slope_angle_deg"]),
        slope_angles_by_sector=econ.get("slope_angles_by_sector") or {},
        discount_rate_pct=float(econ["discount_rate_pct"]),
    )

    log.info("Call 2 (%s): %s at %.2f %s, slope %.0f deg, %d assumption(s)",
             source, economics.commodity, economics.commodity_price,
             economics.price_unit, economics.overall_slope_angle_deg, len(assumptions))

    return {
        "economics": economics,
        "economics_dict": econ,
        "assumptions": assumptions,
        "report": report,
        "decision_source": source,
        "validation_notes": notes,
    }


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
