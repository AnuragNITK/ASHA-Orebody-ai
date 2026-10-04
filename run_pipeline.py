#!/usr/bin/env python
"""Command-line entry point for the AI Mine Design System.

Examples
--------
Run the bundled synthetic deposit end to end:

    python run_pipeline.py --sample

Run your own data with user-supplied economics (CSV or Excel, any mix):

    python run_pipeline.py \\
        --collar data/collar.csv --survey data/survey.csv \\
        --assay data/assay.csv --geology data/geology.csv \\
        --project "Mount Isa South" --price 2400 --mining-cost 2.8 --slope 48

One Excel workbook with a sheet per table (collar/survey/assay/geology):

    python run_pipeline.py --workbook data/deposit.xlsx --project "Mount Isa South"

Add --offline to skip both paid calls and use the coded conventions instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# =============================================================================
# EDIT THESE to point at your own data, then just run:
#
#     python run_pipeline.py
#
# Accepted formats, detected automatically:
#   * CSV
#   * Excel .xlsx / .xlsm / .xls
#   * WAMEX / GSWA statutory .txt (the H0002../H1000/D-row lodgement layout)
#
# Leave all four blank ("") to use --collar/--survey/--assay/--geology/
# --workbook command-line flags instead, or --sample for the bundled demo.
# A command-line flag always overrides the matching line below.
# =============================================================================
_WAMEX = r"C:\Users\a4anu\Downloads\Drilling Data (1)\Appendix I - Drillhole Datasets"

collar_file_path = _WAMEX + r"\PT_WASLS_COLL2016A.txt"
assay_file_path = _WAMEX + r"\PT_WADG3_ASS2016A.txt"
geology_file_path = _WAMEX + r"\PT_WADL3_GEO2016A.txt"
survey_file_path = _WAMEX + r"\PT_WADS3_SURV2016A.txt"

# Which assayed element is the grade, e.g. "U", "Au", "Cu".
# Required when the assay file carries many elements (a WAMEX/GSWA table
# has dozens); leave blank for a simple table with a single grade column.
grade_element = r"U"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_argument_group("input files (CSV, Excel .xlsx/.xlsm/.xls, or WAMEX .txt)")
    src.add_argument("--collar", default=collar_file_path or None,
                     help="collar table: hole id, easting, northing, elevation")
    src.add_argument("--survey", default=survey_file_path or None,
                     help="survey table: hole id, depth, azimuth, dip")
    src.add_argument("--assay", default=assay_file_path or None,
                     help="assay table: hole id, from, to, grade")
    src.add_argument("--grade-element", default=grade_element or None,
                     help="which assayed element is the grade (e.g. U, Au, Cu); "
                          "required for a multi-element assay file")
    src.add_argument("--geology", default=geology_file_path or None,
                     help="geology table: hole id, from, to, rock type")
    src.add_argument("--workbook",
                     help="one Excel file holding all four tables, one sheet per kind "
                          "(sheet names containing 'collar'/'survey'/'assay'/'geology' "
                          "are matched automatically; a single-sheet file is used as-is). "
                          "Used for any of --collar/--survey/--assay/--geology left unset, "
                          "so sources can also be mixed, e.g. --workbook plus one override.")
    src.add_argument("--sample", action="store_true",
                     help="use the bundled synthetic dataset in data/sample")

    job = p.add_argument_group("job settings")
    job.add_argument("--project", default="Untitled Deposit", help="project name for the report")
    job.add_argument("--commodity", default="gold")
    job.add_argument("--grade-unit", default="g/t")
    job.add_argument("--density", type=float, help="in-situ bulk density, t/m3")
    job.add_argument("--composite-length", type=float,
                     help="override the modal composite length, in metres")
    job.add_argument("--block-size", type=float, help="override the block size, in metres")
    job.add_argument("--top-cut-percentile", type=float, default=99.0)
    job.add_argument("--no-cross-validation", action="store_true",
                     help="skip leave-one-out cross validation (faster)")
    job.add_argument("--output", help="path for the HTML report")

    econ = p.add_argument_group("economics (any value given here is used as fact, "
                                "not as a model proposal)")
    econ.add_argument("--price", type=float, help="commodity price")
    econ.add_argument("--price-unit", help="price unit, e.g. USD/oz")
    econ.add_argument("--mining-cost", type=float, help="mining cost per tonne")
    econ.add_argument("--processing-cost", type=float, help="processing cost per tonne")
    econ.add_argument("--ga-cost", type=float, help="general and administrative cost per tonne")
    econ.add_argument("--recovery", type=float, help="metallurgical recovery, percent")
    econ.add_argument("--slope", type=float, help="overall pit slope angle, degrees")
    econ.add_argument("--discount-rate", type=float, help="discount rate, percent")

    gempy = p.add_argument_group("GemPy implicit orebody model (no LLM involved)")
    gempy.add_argument("--no-gempy", action="store_true",
                       help="skip the GemPy implicit orebody model")
    gempy.add_argument("--gempy-cells", type=int, default=120_000,
                       help="target voxel count for the interpolation grid")
    gempy.add_argument("--gempy-contacts", type=int, default=400,
                       help="maximum ore contact points fed to the interpolator")

    misc = p.add_argument_group("behaviour")
    misc.add_argument("--offline", action="store_true",
                      help="skip both paid LLM calls and use coded conventions")
    misc.add_argument("--json", action="store_true", help="print the run summary as JSON")
    return p


def collect_economics(args) -> dict:
    mapping = {
        "commodity_price": args.price,
        "price_unit": args.price_unit,
        "mining_cost_per_tonne": args.mining_cost,
        "processing_cost_per_tonne": args.processing_cost,
        "ga_cost_per_tonne": args.ga_cost,
        "recovery_pct": args.recovery,
        "overall_slope_angle_deg": args.slope,
        "discount_rate_pct": args.discount_rate,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.offline:  # must be set before minedesign.config is imported
        os.environ["MINEDESIGN_OFFLINE"] = "1"

    from minedesign.errors import MineDesignError
    from minedesign.pipeline import PipelineConfig, run_pipeline

    if args.sample:
        base = ROOT / "data" / "sample"
        if not (base / "collar.csv").exists():
            print("Sample data is missing. Generate it first:\n"
                  "    python scripts/make_sample_data.py", file=sys.stderr)
            return 2
        paths = [base / f"{n}.csv" for n in ("collar", "survey", "assay", "geology")]
        if args.project == "Untitled Deposit":
            args.project = "Synthetic Gold Deposit"
    else:
        # a per-kind flag always wins; --workbook fills in whatever is left unset,
        # so one Excel file plus a single override (e.g. a corrected assay CSV) works
        resolved = {n: getattr(args, n) or args.workbook
                   for n in ("collar", "survey", "assay", "geology")}
        missing = [n for n in resolved if not resolved[n]]
        if missing:
            print(f"Missing required input file(s): {', '.join('--' + m for m in missing)}\n"
                  f"Provide --{'/--'.join(missing)} individually, or --workbook for an Excel "
                  f"file holding all four tables, or use --sample for the bundled dataset.",
                  file=sys.stderr)
            return 2
        paths = [resolved["collar"], resolved["survey"], resolved["assay"], resolved["geology"]]

    config = PipelineConfig(
        project_name=args.project,
        commodity=args.commodity,
        grade_unit=args.grade_unit,
        grade_element=args.grade_element,
        density=args.density,
        composite_length=args.composite_length,
        block_size=args.block_size,
        top_cut_percentile=args.top_cut_percentile,
        run_cross_validation=not args.no_cross_validation,
        user_economics=collect_economics(args),
        output_path=args.output,
        gempy_model=not args.no_gempy,
        gempy_target_cells=args.gempy_cells,
        gempy_max_contacts=args.gempy_contacts,
    )

    try:
        result = run_pipeline(*paths, config=config)
    except MineDesignError as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"report": str(result.report_path),
                          "duration_s": round(result.duration_s, 2),
                          "summary": result.summary,
                          "calls": result.call_log}, indent=2, default=str))
    else:
        s = result.summary
        print("\n" + "=" * 66)
        print(f"  {config.project_name}")
        print("=" * 66)
        print(f"  Drillholes            {s['n_holes']}")
        print(f"  Composites            {s['n_composites']:,} at {s['composite_length']:g} m")
        print(f"  Cutoff grade          {s['cutoff_grade']:.3f} {s['grade_unit']}")
        print(f"  Estimation            {s['estimation_method']} "
              f"({s['variogram_model']}, range {s['variogram_range']:.0f} m)")
        print(f"  Block model           {s['n_blocks_total']:,} blocks of {s['block_size_label']}")
        print(f"  In-situ resource      {s['resource_tonnes'] / 1e6:,.2f} Mt "
              f"at {s['resource_grade']:.3f} {s['grade_unit']}")
        print(f"  Pit ore / waste       {s['pit_ore_tonnes'] / 1e6:,.2f} Mt / "
              f"{s['pit_waste_tonnes'] / 1e6:,.2f} Mt")
        print(f"  Strip ratio           {s['pit_strip_ratio']:.2f}")
        print(f"  Recovered metal       {s['pit_recovered_oz'] / 1000:,.1f} koz")
        if s.get("slope_warning"):
            print(f"\n  NOTE: {s['slope_warning']}")

        print(f"\n  Report                {result.report_path}")
        print(f"  Elapsed               {result.duration_s:.1f}s")
        print("=" * 66)
        print("  This is decision support, not a JORC or NI 43-101 resource statement.")
        print("  Have a mining engineer review the cutoff, variogram and pit economics.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
