"""End-to-end pipeline: four CSVs in, one self-contained HTML report out.

Control flow, matching the design:

    [code] ingest -> validate -> desurvey -> colour -> composite -> top-cut
       -> [LLM CALL 1: cutoff + domain boundaries + variogram/method]
       -> [code] triangulate -> validate_solid (retry: local fallback, max 1x)
       -> [code] block grid -> estimate -> point_in_solid -> tonnage
       -> [code] preliminary pit + nested shells at default economics
       -> [LLM CALL 2: pit economics + report narrative]
       -> [code] final Lerchs-Grossmann pit -> scene, charts, tables -> HTML

Two paid API calls per run regardless of dataset size. Everything else is
deterministic Python.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .blockmodel.grid import (
    Topography,
    build_block_grid,
    data_density_summary,
    recommend_block_size,
)
from .blockmodel.kriging import cross_validate, run_kriging
from .blockmodel.tonnage import compute_tonnage, grade_tonnage_curve
from .blockmodel.variogram import directional_analysis, experimental_variogram, fit_all_models
from .config import OUTPUT_DIR, SETTINGS
from .errors import MineDesignError
from .geology.compositing import (
    apply_top_cut,
    composite,
    recommend_composite_length,
    recommend_top_cut,
    tag_rock_type,
)
from .geology.desurvey import attach_coordinates, desurvey_minimum_curvature
from .geology.stats import choose_color_breakpoints, compute_grade_stats
from .ingest.loader import load_and_validate
from .llm.call1_orebody import repropose_boundaries, run_orebody_agent
from .llm.call2_pit import DEFAULT_ECONOMICS, run_pit_report_agent
from .llm.client import CallLog
from .logging_utils import get_logger, step
from .pit.lerchs_grossmann import (
    EconomicParameters,
    bench_schedule,
    nested_pit_shells,
    run_lerchs_grossmann,
)
from .report import charts as chart_builders
from .report.render import dataframe_to_table, render_report
from .report.scene import build_scene
from .solid.point_in_solid import point_in_solid
from .solid.sections import assign_sections, build_section_frame, drill_spacing
from .solid.triangulate import triangulate_solid
from .solid.validate import validate_solid

log = get_logger(__name__)

VERSION = "1.0.0"


@dataclass
class PipelineConfig:
    """Everything a caller may want to steer, with sensible defaults."""

    project_name: str = "Untitled Deposit"
    density: float | dict = None  # falls back to SETTINGS.default_density
    commodity: str = "gold"
    grade_unit: str = "g/t"
    # Which assayed element is the grade. Required for a multi-element file
    # (a WAMEX assay table carries dozens of elements); ignored for a simple
    # table that already has one grade column.
    grade_element: str | None = None
    composite_length: float | None = None      # None -> modal sample length
    top_cut_percentile: float = 99.0
    block_size: float | None = None            # None -> half the drill spacing
    n_sections: int | None = None
    user_economics: dict = field(default_factory=dict)
    max_samples_per_block: int = 24
    min_samples_per_block: int = 4
    run_cross_validation: bool = True
    allow_llm: bool = True
    output_path: str | Path | None = None

    # --- GemPy implicit orebody model (no LLM involved at any point)
    gempy_model: bool = True
    gempy_target_cells: int = 120_000
    gempy_max_contacts: int = 400

    def resolved_density(self) -> float | dict:
        return self.density if self.density is not None else SETTINGS.default_density


@dataclass
class PipelineResult:
    report_path: Path
    summary: dict
    call_log: list
    duration_s: float
    artefacts: dict = field(default_factory=dict)


def _stats_rows(before: dict, after: dict) -> dict:
    """Side-by-side descriptive statistics for the report table."""
    def fmt(value, nd=4):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return f"{value:,.{nd}f}" if np.isfinite(value) else "n/a"

    labels = [
        ("Samples", "n", 0), ("Minimum", "min", 4), ("Maximum", "max", 4),
        ("Mean", "mean", 4), ("Median", "median", 4),
        ("Geometric mean", "geometric_mean", 4), ("Standard deviation", "std_dev", 4),
        ("Variance", "variance", 5), ("Coefficient of variation", "cv", 3),
        ("Skewness", "skewness", 3), ("Kurtosis (excess)", "kurtosis_excess", 3),
    ]
    rows = [{"label": label, "before": fmt(before.get(key), nd), "after": fmt(after.get(key), nd)}
            for label, key, nd in labels]
    for p in ("p10", "p25", "p50", "p75", "p90", "p95", "p99"):
        rows.append({
            "label": f"Percentile {p[1:]}",
            "before": fmt(before.get("percentiles", {}).get(p)),
            "after": fmt(after.get("percentiles", {}).get(p)),
        })
    return {"rows": rows}


def run_pipeline(collar_path, survey_path, assay_path, geology_path,
                 config: PipelineConfig | None = None) -> PipelineResult:
    """Run the whole workflow and write the HTML report."""
    config = config or PipelineConfig()
    started = time.perf_counter()
    call_log = CallLog()
    density = config.resolved_density()

    # ---------------------------------------------------------------- ingest
    with step("ingest and validate source files"):
        ingested = load_and_validate(collar_path, survey_path, assay_path, geology_path,
                                     allow_llm_fallback=config.allow_llm,
                                     grade_element=config.grade_element)
        # a WAMEX file states its own assay unit; honour it over the default
        if ingested.grade_unit and config.grade_unit == "g/t":
            config.grade_unit = ingested.grade_unit

    with step("desurvey by minimum curvature"):
        traces = desurvey_minimum_curvature(ingested.collar, ingested.survey)
        raw_assay = attach_coordinates(
            tag_rock_type(ingested.assay, ingested.geology), traces)

    # ----------------------------------------------- composite and top cut
    with step("composite and apply top cut"):
        comp_info = recommend_composite_length(ingested.assay)
        length = config.composite_length or comp_info["length"]
        composited = composite(ingested.assay, length)
        composited = tag_rock_type(composited, ingested.geology)
        composited = attach_coordinates(composited, traces)

        stats_before = compute_grade_stats(composited, weight_column="length")
        top_cut = recommend_top_cut(stats_before, config.top_cut_percentile)
        capped, top_cut_summary = apply_top_cut(composited, top_cut["value"])
        stats_after = compute_grade_stats(capped, weight_column="length")
        ramp = choose_color_breakpoints(stats_after)

    # ------------------------------------------------ sections and variography
    with step("build the section frame and variograms"):
        frame = build_section_frame(ingested.collar, config.n_sections)
        capped = assign_sections(capped, frame)
        spacing = drill_spacing(ingested.collar)

        block_rec = recommend_block_size(spacing)
        block_size = config.block_size or block_rec["block_size"]

        estimation_input = capped.dropna(subset=["x", "y", "z", "grade"])
        points = estimation_input[["x", "y", "z"]].to_numpy(float)
        values = estimation_input["grade"].to_numpy(float)

        variography: dict = {}
        try:
            omni = experimental_variogram(points, values)
            variography["omnidirectional"] = {"experimental": omni,
                                              **fit_all_models(omni)}
        except Exception as exc:  # noqa: BLE001
            log.warning("omnidirectional variogram failed: %s", exc)
        try:
            variography.update(directional_analysis(points, values, frame))
        except Exception as exc:  # noqa: BLE001
            log.warning("directional variography failed: %s", exc)

        density_summary = data_density_summary(estimation_input, frame, block_size)

    # --------------------------------------------------------- LLM CALL 1
    with step("LLM Call 1: cutoff, domaining and estimation method"):
        decision = run_orebody_agent(
            capped, stats_after, top_cut, comp_info, frame,
            density_summary, variography, call_log=call_log)
        ore_mask = decision.pop("ore_mask")
        capped["is_ore_sample"] = ore_mask

    # ------------------------------------------- triangulate and validate
    with step("triangulate and validate the orebody solid"):
        solid, validation, boundaries = _build_validated_solid(
            decision, frame, capped[ore_mask], call_log)

    # -------------------------------------------------------- block model
    with step("build the block grid and estimate grades"):
        bbox = solid.bbox
        blocks = build_block_grid(bbox, block_size, padding=block_size)
        actual_block_size = blocks.attrs["block_size"]

        topo = Topography(ingested.collar)
        blocks["is_air"] = ~topo.below_surface(blocks[["x", "y", "z"]].to_numpy(float))

        anisotropy = variography.get("anisotropy")
        blocks = run_kriging(
            blocks, capped, decision["variogram_params"], decision["estimation_method"],
            frame=frame, anisotropy=anisotropy,
            max_samples=config.max_samples_per_block,
            min_samples=config.min_samples_per_block)
        blocks.attrs["block_size"] = actual_block_size
        blocks.attrs["block_volume"] = float(np.prod(actual_block_size))

    with step("classify blocks against the solid and compute tonnage"):
        inside = point_in_solid(blocks[["x", "y", "z"]].to_numpy(float), solid)
        blocks["in_solid"] = inside
        blocks["is_ore"] = (inside & np.isfinite(blocks["grade"].to_numpy(float))
                            & ~blocks["is_air"].to_numpy(bool)
                            & (blocks["grade"].to_numpy(float) >= decision["cutoff_grade"]))
        blocks["rock_type"] = np.where(blocks["is_ore"], "ORE", "WASTE")

        tonnage = compute_tonnage(blocks, density, blocks.attrs["block_volume"])
        gt_curve = grade_tonnage_curve(blocks[blocks["in_solid"]], density,
                                       blocks.attrs["block_volume"])

    cross_val = {}
    if config.run_cross_validation:
        with step("cross-validate the estimator"):
            cross_val = cross_validate(capped, decision["variogram_params"],
                                       decision["estimation_method"], frame=frame,
                                       anisotropy=anisotropy)

    # ------------------- GemPy implicit orebody model, entirely deterministic
    gempy_result = None
    if config.gempy_model:
        with step("GemPy implicit orebody model"):
            from .gempy_model.builder import build_gempy_model
            from .gempy_model.contacts import build_contact_set

            try:
                contact_set = build_contact_set(capped, ore_mask,
                                                thin=config.gempy_max_contacts)
                ore_xyz = (capped.loc[ore_mask, ["x", "y", "z"]].dropna()
                           .to_numpy(float))
                gempy_result = build_gempy_model(
                    contact_set, project_name=_slug(config.project_name),
                    target_cells=config.gempy_target_cells,
                    density=density if not isinstance(density, dict)
                    else SETTINGS.default_density,
                    blocks=blocks, wireframe_volume=validation.volume,
                    ore_sample_xyz=ore_xyz)
            except Exception as exc:  # noqa: BLE001
                log.error("GemPy model failed: %s", exc, exc_info=True)
                from .gempy_model.builder import GemPyResult

                gempy_result = GemPyResult(available=False, error=str(exc))

    # -------------------------------- preliminary pit, so Call 2 sees reality
    with step("preliminary pit and nested shells at default economics"):
        pit_blocks = blocks.copy()
        pit_blocks["tonnes"] = (blocks.attrs["block_volume"]
                                * (density if not isinstance(density, dict)
                                   else SETTINGS.default_density))
        # The wireframe constrains the resource, so a block outside the solid is
        # waste no matter what the estimator interpolated into it. Without this
        # the optimizer sells grade smeared into the padding around the orebody
        # and reports more ore in the pit than exists in the deposit.
        outside = ~blocks["in_solid"].to_numpy(bool) | blocks["is_air"].to_numpy(bool)
        pit_blocks.loc[outside, "grade"] = 0.0
        pit_blocks["grade"] = pit_blocks["grade"].fillna(0.0)
        pit_blocks.attrs["block_size"] = actual_block_size
        log.info("pit model: %d of %d blocks carry a grade; the rest are waste or air",
                 int((~outside).sum()), len(pit_blocks))

        default_econ = EconomicParameters(**{k: v for k, v in DEFAULT_ECONOMICS.items()
                                             if k != "commodity"},
                                          commodity=config.commodity)
        default_econ.grade_unit = config.grade_unit
        preliminary = run_lerchs_grossmann(pit_blocks, default_econ)
        shells = nested_pit_shells(pit_blocks, default_econ)

    # --------------------------------------------------------- LLM CALL 2
    with step("LLM Call 2: pit economics and the engineering report"):
        context = _build_call2_context(
            ingested, stats_before, stats_after, top_cut_summary, comp_info,
            decision, validation, tonnage, gt_curve, preliminary, shells,
            cross_val, blocks, config)
        pit_agent = run_pit_report_agent(context, config.user_economics, call_log=call_log)

    # --------------------------------------------- final pit with those economics
    with step("final Lerchs-Grossmann ultimate pit"):
        economics = pit_agent["economics"]
        economics.grade_unit = config.grade_unit
        final_pit = run_lerchs_grossmann(
            pit_blocks, economics,
            slope_angles=economics.slope_angles_by_sector or None)
        final_shells = nested_pit_shells(pit_blocks, economics)

        bench_height = float(actual_block_size[2])
        schedule_density = density if not isinstance(density, dict) else SETTINGS.default_density
        pit_bench_schedule = bench_schedule(final_pit.pit_surface, bench_height, schedule_density)

    # -------------------------------------------------------------- report
    with step("assemble the 3D scene, charts and tables"):
        scene = build_scene(
            traces=traces, composited_raw=raw_assay, composited=composited,
            capped=capped, boundaries=boundaries, frame=frame, solid=solid,
            blocks=blocks, in_pit=final_pit.in_pit,
            pit_surface_data=final_pit.pit_surface, ramp=ramp,
            block_size=actual_block_size, topo=topo, bbox=bbox,
            gempy_result=gempy_result)

        charts = {
            "histogram": chart_builders.grade_histogram_chart(
                capped["grade"].to_numpy(float)),
            "cumulative": chart_builders.cumulative_frequency_chart(
                capped["grade"].to_numpy(float)),
            "topcut": chart_builders.topcut_comparison_chart(capped),
            "variogram": chart_builders.variogram_chart(variography),
            "grade_tonnage": chart_builders.grade_tonnage_chart(gt_curve),
            "nested_shells": chart_builders.nested_shells_chart(final_shells or shells),
        }

        tables = {
            "composites": dataframe_to_table(
                capped,
                ["hole_id", "from", "to", "length", "x", "y", "z", "grade",
                 "grade_uncapped", "capped", "rock_type", "section_id"],
                rounding={"x": 2, "y": 2, "z": 2, "grade": 4, "grade_uncapped": 4,
                          "from": 2, "to": 2, "length": 2}),
            "blocks": dataframe_to_table(
                blocks[blocks["in_solid"]].assign(
                    tonnes=blocks.attrs["block_volume"] * (
                        density if not isinstance(density, dict) else SETTINGS.default_density),
                    in_pit=final_pit.in_pit[blocks["in_solid"].to_numpy(bool)]),
                ["block_id", "x", "y", "z", "grade", "estimation_variance", "tonnes",
                 "rock_type", "confidence", "n_samples_used", "in_pit"],
                rounding={"x": 2, "y": 2, "z": 2, "grade": 4, "estimation_variance": 5,
                          "tonnes": 1}),
        }

        gempy_context = _build_gempy_context(gempy_result, capped, frame, boundaries)

        summary = _build_summary(
            ingested, comp_info, length, decision, validation, tonnage,
            final_pit, blocks, actual_block_size, cross_val, config,
            top_cut_summary, len(capped))

        report_context = {
            "meta": {
                "project_name": config.project_name,
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "version": VERSION,
            },
            "report": pit_agent["report"],
            "economics": pit_agent["economics_dict"],
            "assumptions": pit_agent["assumptions"],
            "flags": [f.as_dict() for f in ingested.flags],
            "provenance": {
                "llm_used": decision["decision_source"] != "deterministic"
                            or pit_agent["decision_source"] != "deterministic",
                "call_1_source": decision["decision_source"],
                "call_2_source": pit_agent["decision_source"],
                "calls": call_log.as_list(),
            },
            "summary": summary,
            "validation": validation.as_dict(),
            "stats": _stats_rows(stats_before, stats_after),
            "scene": scene,
            "charts": charts,
            "tables": tables,
            "gempy": gempy_context,
            "bench_schedule": pit_bench_schedule,
        }

    output_path = Path(config.output_path) if config.output_path else (
        OUTPUT_DIR / f"{_slug(config.project_name)}_report.html")
    with step("render the self-contained HTML report"):
        report_path = render_report(report_context, output_path)

    duration = time.perf_counter() - started
    log.info("pipeline complete in %.1fs -> %s", duration, report_path)

    return PipelineResult(
        report_path=report_path,
        summary=summary,
        call_log=call_log.as_list(),
        duration_s=duration,
        artefacts={
            "blocks": blocks, "composites": capped, "solid": solid,
            "validation": validation.as_dict(), "pit": final_pit,
            "traces": traces, "decision": decision, "variography": variography,
            "frame": frame, "ramp": ramp, "boundaries": boundaries,
            "topo": topo, "density": density, "block_size": actual_block_size,
            "gempy": gempy_result,
        },
    )


def _build_validated_solid(decision, frame, ore_intervals, call_log):
    """Triangulate, validate, and escalate a re-proposal only after two failures."""
    boundaries = decision["domain_boundaries"]
    attempts: list[tuple] = []

    for attempt in range(1, 3):
        try:
            solid = triangulate_solid(boundaries, frame)
            validation = validate_solid(solid)
        except Exception as exc:  # noqa: BLE001
            log.warning("triangulation attempt %d failed: %s", attempt, exc)
            attempts.append((None, None, str(exc)))
            boundaries = decision.get("fallback_boundaries") or boundaries
            continue

        if validation.is_valid:
            return solid, validation, boundaries
        attempts.append((solid, validation, "; ".join(validation.failures)))
        log.warning("solid validation attempt %d failed: %s", attempt,
                    "; ".join(validation.failures))
        if attempt == 1:
            boundaries = decision.get("fallback_boundaries") or boundaries

    # two failures: escalate the boundary re-proposal to the local fallback model
    failures = [a[2] for a in attempts if a[2]]
    repaired = repropose_boundaries(boundaries, failures, frame, ore_intervals,
                                    call_log=call_log)
    if repaired:
        try:
            solid = triangulate_solid(repaired, frame)
            validation = validate_solid(solid)
            log.info("boundary re-proposal produced a %s solid",
                     "valid" if validation.is_valid else "still-invalid")
            return solid, validation, repaired
        except Exception as exc:  # noqa: BLE001
            log.warning("re-proposed boundaries also failed to triangulate: %s", exc)

    # keep the best solid we have and report its failures honestly
    for solid, validation, _ in reversed(attempts):
        if solid is not None:
            return solid, validation, boundaries

    raise MineDesignError(
        "could not build an orebody solid from any boundary set: "
        + "; ".join(failures))


def _build_call2_context(ingested, stats_before, stats_after, top_cut_summary,
                         comp_info, decision, validation, tonnage, gt_curve,
                         preliminary, shells, cross_val, blocks, config) -> dict:
    """The evidence pack for LLM Call 2. Summaries only, never whole tables."""
    return {
        "project": {"name": config.project_name, "commodity": config.commodity,
                    "grade_unit": config.grade_unit},
        "ingestion": ingested.summary(),
        "compositing": comp_info,
        "top_cut": top_cut_summary,
        "grade_statistics_before_top_cut": stats_before,
        "grade_statistics_after_top_cut": stats_after,
        "call_1_decision": {
            "cutoff_grade": decision["cutoff_grade"],
            "cutoff_rationale": decision["cutoff_rationale"],
            "ore_rock_types": decision["ore_rock_types"],
            "estimation_method": decision["estimation_method"],
            "estimation_rationale": decision["estimation_rationale"],
            "variogram_model": decision["variogram_model"],
            "variogram_params": decision["variogram_params"],
            "decision_source": decision["decision_source"],
            "n_ore_intervals": decision["n_ore_intervals"],
        },
        "solid_validation": validation.as_dict(),
        "block_model": {
            "n_blocks_total": tonnage["n_blocks_total"],
            "n_blocks_ore": tonnage["n_blocks_ore"],
            "ore_tonnes": tonnage["ore_tonnes"],
            "mean_ore_grade": tonnage["mean_ore_grade"],
            "contained_metal_oz": tonnage["contained_metal_oz"],
            "by_rock_type": tonnage["by_rock_type"],
            "by_confidence": tonnage["by_confidence"],
            "block_size": list(blocks.attrs["block_size"]),
            "density_used": tonnage["density_used"],
        },
        "grade_tonnage_curve": {
            "cutoffs": gt_curve["cutoffs"][::3],
            "tonnes": gt_curve["tonnes"][::3],
            "mean_grade": gt_curve["mean_grade"][::3],
        },
        "cross_validation": cross_val,
        "preliminary_pit": preliminary.summary,
        "preliminary_economics": preliminary.economics.as_dict(),
        "nested_shells": shells,
    }


def _build_summary(ingested, comp_info, length, decision, validation, tonnage,
                   final_pit, blocks, block_size, cross_val, config,
                   top_cut_summary, n_composites) -> dict:
    s = final_pit.summary
    slope_gap = abs(s.get("slope_angle_achieved_deg", 0) - s.get("slope_angle_requested_deg", 0))
    return {
        "n_holes": int(ingested.collar["hole_id"].nunique()),
        "n_assay_intervals": int(len(ingested.assay)),
        "n_composites": int(n_composites),
        "composite_length": float(length),
        "grade_unit": config.grade_unit,
        "cutoff_grade": decision["cutoff_grade"],
        "n_ore_intervals": decision["n_ore_intervals"],
        "ore_rock_types": decision["ore_rock_types"],
        "boundary_source": decision.get("boundary_source", "unknown"),
        "estimation_method": decision["estimation_method"],
        "estimation_rationale": decision["estimation_rationale"],
        "variogram_model": decision["variogram_model"],
        "variogram_range": decision["variogram_params"]["range"],
        "variogram_nugget": decision["variogram_params"]["nugget"],
        "cross_validation": cross_val,
        "n_blocks_total": int(len(blocks)),
        "n_blocks_ore": int(blocks["is_ore"].sum()),
        "block_size_label": " x ".join(f"{v:g}" for v in block_size) + " m",
        "resource_tonnes": tonnage["ore_tonnes"],
        "resource_grade": tonnage["mean_ore_grade"],
        "top_cut_value": top_cut_summary["top_cut_value"],
        "pit_ore_tonnes": s["ore_tonnes"],
        "pit_waste_tonnes": s["waste_tonnes"],
        "pit_strip_ratio": s["strip_ratio"],
        "pit_mean_grade": s["mean_ore_grade"],
        "pit_contained_oz": s["contained_metal_oz"],
        "pit_recovered_oz": s["recovered_metal_oz"],
        "pit_cashflow": s["undiscounted_cashflow"],
        "pit_depth_m": s["pit_depth_m"],
        "pit_floor_rl": s["pit_floor_rl"],
        "internal_cutoff": s["internal_cutoff_grade"],
        "slope_requested": s.get("slope_angle_requested_deg"),
        "slope_achieved": s.get("slope_angle_achieved_deg"),
        "slope_warning": (
            f"A {s.get('slope_angle_requested_deg', 0):.1f} degree wall cannot be represented "
            f"exactly on a {block_size[0]:g} m block, so the pit is cut at "
            f"{s.get('slope_angle_achieved_deg', 0):.1f} degrees. A steeper wall than requested "
            f"understates waste, so this result is optimistic; use a smaller block or an "
            f"achievable angle." if slope_gap > 1.5 else ""),
    }


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_") or "report"


def _build_gempy_context(result, composites, frame, boundaries,
                         max_sections: int = 3) -> dict:
    """Everything the GemPy section of the report needs."""
    if result is None:
        return {"enabled": False}

    context = {"enabled": True, "summary": result.summary(),
               "comparison": result.comparison, "sections": [], "rose": None,
               "orientations": [], "available": result.available,
               "error": result.error, "warnings": result.warnings}

    if not result.available:
        return context

    from .gempy_model import plots as gempy_plots

    section_ids = sorted(composites["section_id"].unique())
    if len(section_ids) > max_sections:
        step_size = len(section_ids) / max_sections
        section_ids = [section_ids[int(i * step_size)] for i in range(max_sections)]

    for sid in section_ids:
        try:
            svg = gempy_plots.implicit_section(result, int(sid), composites, frame,
                                              boundaries=boundaries)
        except Exception as exc:  # noqa: BLE001
            log.warning("GemPy section %s failed to draw: %s", sid, exc)
            continue
        if svg:
            context["sections"].append({"id": int(sid), "svg": svg})

    try:
        context["rose"] = gempy_plots.orientation_rose(result)
    except Exception as exc:  # noqa: BLE001
        log.warning("orientation rose failed to draw: %s", exc)

    if result.contacts is not None and not result.contacts.orientations.empty:
        ori = result.contacts.orientations
        context["orientations"] = [
            {"x": round(float(r.x), 1), "y": round(float(r.y), 1),
             "z": round(float(r.z), 1), "dip": round(float(r.dip), 1),
             "azimuth": round(float(r.azimuth), 1),
             "points_up": bool(float(r.G_z) >= 0), "kind": str(r.kind)}
            for r in ori.itertuples(index=False)]

    return context
