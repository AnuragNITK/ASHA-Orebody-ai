"""load_and_validate: four source tables in, four clean DataFrames plus a quality
report out.

Each of the four inputs (collar, survey, assay, geology) may be a CSV or an
Excel file (.xlsx/.xlsm/.xls). All four may also point at the same Excel
workbook, one sheet per kind -- the sheet is picked by matching its name to the
data kind, falling back to the first sheet for a single-sheet file.

Deterministic. The only escape hatch is the local fallback model, and it is used
for one file's column mapping when the synonym dictionary fails -- never for the
data itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import IngestionError
from ..logging_utils import get_logger
from . import wamex
from .column_map import NUMERIC_FIELDS, SCHEMAS, map_columns

log = get_logger(__name__)


@dataclass
class QualityFlag:
    """One data-quality issue, surfaced prominently in the final report."""

    severity: str  # "info" | "warning" | "error"
    file: str
    message: str
    count: int = 0

    def as_dict(self) -> dict:
        return {"severity": self.severity, "file": self.file,
                "message": self.message, "count": self.count}


@dataclass
class IngestResult:
    collar: pd.DataFrame
    survey: pd.DataFrame
    assay: pd.DataFrame
    geology: pd.DataFrame
    flags: list[QualityFlag] = field(default_factory=list)
    column_mappings: dict[str, dict[str, str]] = field(default_factory=dict)
    grade_column_source: str = "grade"
    grade_unit: str = ""
    grade_note: str = ""

    def summary(self) -> dict:
        return {
            "n_holes": int(self.collar["hole_id"].nunique()),
            "n_survey_records": int(len(self.survey)),
            "n_assay_intervals": int(len(self.assay)),
            "n_geology_intervals": int(len(self.geology)),
            "grade_column_source": self.grade_column_source,
            "grade_unit": self.grade_unit,
            "grade_note": self.grade_note,
            "column_mappings": self.column_mappings,
            "flags": [f.as_dict() for f in self.flags],
            "collar_extent": {
                "x": [float(self.collar["x"].min()), float(self.collar["x"].max())],
                "y": [float(self.collar["y"].min()), float(self.collar["y"].max())],
                "z": [float(self.collar["z"].min()), float(self.collar["z"].max())],
            },
        }


EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}


def _pick_sheet(sheet_names: list[str], kind: str) -> str | int:
    """Match a sheet to a data kind by name, so one workbook with a sheet per
    kind (collar/survey/assay/geology) is picked automatically. Falls back to
    the first sheet, which covers both a single-sheet file and a workbook whose
    sheets are not named after the kind.
    """
    for name in sheet_names:
        normalised = str(name).strip().lower()
        if normalised == kind or normalised.startswith(kind):
            return name
    return 0


def _read_excel(path: Path, kind: str) -> pd.DataFrame:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    try:
        book = pd.ExcelFile(path, engine=engine)
    except ImportError as exc:
        raise IngestionError(
            f"reading {kind} needs the '{engine}' package; install it with: "
            f"python -m pip install {engine}") from exc
    sheet = _pick_sheet(book.sheet_names, kind)
    if len(book.sheet_names) > 1:
        log.info("%s: workbook has %d sheets, using '%s'",
                 kind, len(book.sheet_names), sheet if isinstance(sheet, str)
                 else book.sheet_names[sheet])
    return book.parse(sheet_name=sheet)


def _read_wamex_table(p: Path, kind: str, grade_element: str | None,
                      flags: list | None, context: dict | None) -> pd.DataFrame:
    """Read a WAMEX file and, for an assay file, resolve which element is the grade.

    A multi-element assay file has no column called "grade": it has dozens of
    element columns, several of them the same element by a different analytical
    method. The synonym dictionary would happily bind the first one it
    recognises -- on this Paterson uranium data that is Cu -- so the element is
    required to be stated, and the column actually used is recorded.
    """
    table = wamex.read_wamex(p)
    df = table.frame
    if kind != "assay":
        return df

    column, unit, note = wamex.select_grade_column(table, grade_element)
    values, below, above = wamex.numeric_with_detection_limits(df[column])
    df = df.copy()
    df["grade"] = values

    if context is not None:
        context["grade_unit"] = unit
        context["grade_note"] = note
        context["grade_column"] = column
    if flags is not None:
        flags.append(QualityFlag("info", "assay", note))
        if below:
            flags.append(QualityFlag(
                "warning", "assay",
                "results below the laboratory detection limit were substituted with half "
                "the limit, the standard convention for censored assay data", below))
        if above:
            flags.append(QualityFlag(
                "warning", "assay",
                "results over the top of the calibrated range were substituted with the "
                "limit itself, so these intervals understate the true grade", above))
    log.info("assay grade column: %s (%s), %d below-detection, %d over-range",
             column, unit, below, above)
    return df


def _read_table(path: str | Path, kind: str, grade_element: str | None = None,
                flags: list | None = None, context: dict | None = None) -> pd.DataFrame:
    """Read one source table as CSV, Excel or WAMEX, by extension and content."""
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"{kind} file not found: {p}")
    if wamex.is_wamex(p):
        return _read_wamex_table(p, kind, grade_element, flags, context)
    try:
        if p.suffix.lower() in EXCEL_SUFFIXES:
            df = _read_excel(p, kind)
        else:
            df = pd.read_csv(p, sep=None, engine="python", comment="#", skipinitialspace=True)
    except IngestionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"could not parse {kind} file {p.name}: {exc}") from exc
    if df.empty:
        raise IngestionError(f"{kind} file {p.name} contains no rows")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _apply_mapping(df: pd.DataFrame, mapping: dict[str, str], kind: str) -> pd.DataFrame:
    out = df.rename(columns=mapping)
    keep = [c for c in SCHEMAS[kind] if c in out.columns]
    extra = [c for c in out.columns if c not in keep]
    out = out[keep + extra].copy()
    out["hole_id"] = out["hole_id"].astype(str).str.strip()
    for col in NUMERIC_FIELDS[kind]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _map_one_file(df: pd.DataFrame, kind: str, allow_llm_fallback: bool,
                  flags: list[QualityFlag]) -> dict[str, str]:
    mapping, unresolved = map_columns(list(df.columns), kind)
    if not unresolved:
        return mapping

    if allow_llm_fallback:
        log.warning("column mapping failed for %s (%s); escalating this file only "
                    "to the local fallback model", kind, ", ".join(unresolved))
        from ..llm.client import map_columns_with_fallback_model  # lazy: avoids a cycle

        try:
            llm_mapping = map_columns_with_fallback_model(
                kind, list(df.columns), unresolved, df.head(5).to_dict(orient="records"))
        except Exception as exc:  # noqa: BLE001
            llm_mapping = {}
            flags.append(QualityFlag("warning", kind,
                                     f"local fallback column mapper unavailable: {exc}"))
        for src, canonical in llm_mapping.items():
            if src in df.columns and canonical in SCHEMAS[kind]:
                mapping[src] = canonical
        if llm_mapping:
            flags.append(QualityFlag(
                "warning", kind,
                f"column(s) {unresolved} were resolved by the local fallback model rather than "
                "the fixed dictionary; verify the mapping before relying on the result"))
        _, unresolved = map_columns(
            [mapping.get(c, c) for c in df.columns], kind)

    if unresolved:
        raise IngestionError(
            f"{kind} file is missing required column(s) {unresolved}; "
            f"available columns are {list(df.columns)}")
    return mapping


def load_and_validate(collar_path, survey_path, assay_path, geology_path,
                      allow_llm_fallback: bool = True,
                      grade_element: str | None = None) -> IngestResult:
    """Load the four source files, map columns, coerce types and sanity-check."""
    flags: list[QualityFlag] = []
    mappings: dict[str, dict[str, str]] = {}
    frames: dict[str, pd.DataFrame] = {}
    context: dict[str, str] = {}

    for kind, path in (("collar", collar_path), ("survey", survey_path),
                       ("assay", assay_path), ("geology", geology_path)):
        raw = _read_table(path, kind, grade_element, flags, context)
        mapping = _map_one_file(raw, kind, allow_llm_fallback, flags)
        mappings[kind] = mapping
        frames[kind] = _apply_mapping(raw, mapping, kind)

    collar, survey = frames["collar"], frames["survey"]

    grade_source = next((src for src, canon in mappings["assay"].items()
                         if canon == "grade"), "grade")

    # ---- collar checks
    before = len(collar)
    collar = (collar.dropna(subset=["x", "y", "z"])
                    .drop_duplicates(subset=["hole_id"], keep="first"))
    if len(collar) < before:
        flags.append(QualityFlag("warning", "collar",
                                 "collar rows dropped for missing coordinates or a duplicate hole_id",
                                 before - len(collar)))
    if collar.empty:
        raise IngestionError("no usable collar records after validation")

    known = set(collar["hole_id"])

    # ---- survey checks
    survey = survey.dropna(subset=["depth", "azimuth", "dip"]).copy()
    orphan = int((~survey["hole_id"].isin(known)).sum())
    if orphan:
        flags.append(QualityFlag("warning", "survey",
                                 "survey records reference an unknown hole_id", orphan))
        survey = survey[survey["hole_id"].isin(known)].copy()
    survey["azimuth"] = survey["azimuth"] % 360.0
    steep = int((survey["dip"].abs() > 90).sum())
    if steep:
        flags.append(QualityFlag("error", "survey",
                                 "survey dips outside +/-90 degrees were clipped", steep))
        survey["dip"] = survey["dip"].clip(-90, 90)
    survey = survey.sort_values(["hole_id", "depth"]).drop_duplicates(["hole_id", "depth"])

    # Surpac convention: dip is negative downwards. If every dip is positive and the
    # holes are not flat, the file uses positive-down; normalise it once, loudly.
    if not survey.empty and (survey["dip"] > 0).all() and survey["dip"].abs().max() > 5:
        survey["dip"] = -survey["dip"]
        flags.append(QualityFlag("info", "survey",
                                 "all dips were positive; interpreted as positive-down and negated"))

    no_survey = sorted(known - set(survey["hole_id"]))
    if no_survey:
        flags.append(QualityFlag("info", "survey",
                                 "holes with no survey record are treated as vertical", len(no_survey)))

    # ---- interval checks, shared by assay and geology
    for kind in ("assay", "geology"):
        df = frames[kind]
        n0 = len(df)
        df = df.dropna(subset=["from", "to"]).copy()
        bad = df["to"] <= df["from"]
        if bad.any():
            flags.append(QualityFlag("error", kind,
                                     "intervals with to <= from were removed", int(bad.sum())))
            df = df[~bad].copy()
        orphan_mask = ~df["hole_id"].isin(known)
        if orphan_mask.any():
            flags.append(QualityFlag("warning", kind,
                                     "intervals reference an unknown hole_id", int(orphan_mask.sum())))
            df = df[~orphan_mask].copy()
        df = df.sort_values(["hole_id", "from"]).reset_index(drop=True)
        if len(df) < n0:
            log.info("%s: %d of %d intervals retained", kind, len(df), n0)
        frames[kind] = df

    assay, geology = frames["assay"], frames["geology"]

    # ---- grade checks
    missing_grade = int(assay["grade"].isna().sum())
    if missing_grade:
        flags.append(QualityFlag("warning", "assay",
                                 "intervals with a missing grade were excluded from statistics",
                                 missing_grade))
    negative = int((assay["grade"] < 0).sum())
    if negative:
        flags.append(QualityFlag(
            "error", "assay",
            "negative grades found and treated as missing; the below-detection convention "
            "of half the detection limit cannot be applied without a stated limit", negative))
        assay.loc[assay["grade"] < 0, "grade"] = np.nan
    if int(assay["grade"].notna().sum()) == 0:
        raise IngestionError("assay file has no usable numeric grade values")

    # ---- overlapping downhole intervals (a standard Surpac validation)
    for kind, df in (("assay", assay), ("geology", geology)):
        overlaps = 0
        for _, grp in df.groupby("hole_id", sort=False):
            if len(grp) > 1:
                overlaps += int((grp["from"].to_numpy()[1:]
                                 < grp["to"].to_numpy()[:-1] - 1e-9).sum())
        if overlaps:
            flags.append(QualityFlag("warning", kind,
                                     "overlapping downhole intervals detected", overlaps))

    if geology.empty:
        flags.append(QualityFlag("warning", "geology",
                                 "no usable geology intervals; every interval is coded UNKNOWN"))
    else:
        geology["rock_type"] = geology["rock_type"].astype(str).str.strip().str.upper()

    result = IngestResult(collar=collar.reset_index(drop=True),
                          survey=survey.reset_index(drop=True),
                          assay=assay, geology=geology, flags=flags,
                          column_mappings=mappings,
                          grade_column_source=context.get('grade_column', grade_source),
                          grade_unit=context.get('grade_unit', ''),
                          grade_note=context.get('grade_note', ''))
    log.info("ingested %d holes, %d assay intervals, %d geology intervals, %d quality flag(s)",
             result.collar["hole_id"].nunique(), len(assay), len(geology), len(flags))
    return result
