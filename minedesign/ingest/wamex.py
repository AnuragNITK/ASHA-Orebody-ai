"""Reader for WAMEX / GSWA statutory exploration data files.

Western Australian exploration data is lodged with the Geological Survey in a
fixed tab-separated layout: metadata lines keyed ``H0002`` through ``H0900``, a
column-name line keyed ``H1000``, optional unit (``H1001``), analytical method
(``H1002``) and precision (``H1004``) lines, then data rows whose first field is
a literal ``D``, and a trailing ``EOF`` marker.

Three details break a naive reader, and each is handled here rather than left to
pandas:

* **Column names repeat.** An assay file carries the same element once per
  analytical method -- the Paterson dataset this was built against has three
  ``U`` columns. Letting pandas rename the second to ``U.1`` throws away the one
  fact that distinguishes them, so the method line is used to name them instead.
* **Values are censored.** Results below or above the laboratory detection limit
  are written ``<0.5`` or ``>2000``. Left alone they turn an entire element
  column into text and every grade silently becomes NaN.
* **There is no single grade column.** A multi-element file offers dozens of
  candidates and the right one is a decision about the deposit, not something to
  guess. Picking silently is how a copper number ends up labelled as the grade
  of a uranium project, so an ambiguous file raises instead.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..errors import IngestionError
from ..logging_utils import get_logger

log = get_logger(__name__)

HEADER_PREFIX = re.compile(r"^H\d{4}\b")
COLUMN_LINE = "H1000"
UNIT_LINE = "H1001"
METHOD_LINE = "H1002"
DATA_PREFIX = "D"

# A four-acid digest or a peroxide fusion dissolves the whole sample, so the
# result is a total assay. A weak-acid leach deliberately extracts only part of
# it and reads low against the true grade; it is a pathfinder, not a grade.
TOTAL_DIGEST = {"4AMS", "4AOES", "4AHMS", "DXMS", "XRF", "FA", "FA50", "AR"}
PARTIAL_LEACH = {"WAPMS", "WACMS", "AQUA", "WA"}

CONCENTRATION_UNITS = {"ppm", "ppb", "pct", "%", "g/t", "gpt", "ppt", "oz/t", "ozt"}

# "<0.5" (below detection limit) or ">2000" (over range)
CENSORED = re.compile(r"^\s*([<>])\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*$")


@dataclass
class WamexTable:
    """One parsed WAMEX file: the data, plus the header lines that explain it."""

    frame: pd.DataFrame
    units: dict[str, str] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    path: str = ""

    def element_columns(self) -> list[str]:
        """Columns carrying a concentration unit, i.e. candidate grade columns."""
        return [c for c, u in self.units.items()
                if str(u).strip().lower() in CONCENTRATION_UNITS]


def is_wamex(path: str | Path, probe_lines: int = 200) -> bool:
    """True when the file carries a WAMEX ``H1000`` column-name line."""
    p = Path(path)
    if p.suffix.lower() not in {".txt", ".dat", ".tab", ""}:
        return False
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if line.startswith(COLUMN_LINE):
                    return True
                if i > probe_lines:
                    break
    except OSError:
        return False
    return False


def _disambiguate(names: list[str], methods: list[str], units: list[str]) -> list[str]:
    """Give repeated column names a suffix that says how they differ.

    The same element assayed by two methods is two different measurements. The
    method code is what separates them, so it becomes part of the name.
    """
    counts = Counter(names)
    out: list[str] = []
    for name, method, unit in zip(names, methods, units):
        if counts[name] <= 1:
            out.append(name)
            continue
        tag = (method or unit or "").strip() or str(out.count(name) + 1)
        candidate = f"{name}_{tag}"
        n = 2
        while candidate in out:
            candidate = f"{name}_{tag}_{n}"
            n += 1
        out.append(candidate)
    return out


def numeric_with_detection_limits(series: pd.Series) -> tuple[pd.Series, int, int]:
    """Coerce a laboratory column to numbers, substituting censored results.

    ``<0.5`` becomes half the detection limit and ``>2000`` becomes the limit
    itself -- the conventional substitution for censored assay data. Both are
    counted so the caller can report how much of a column was substituted
    instead of letting it pass unremarked.
    """
    below = above = 0
    out = []
    for raw in series.astype("object"):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)) or raw is pd.NA:
            out.append(float("nan"))
            continue
        text = str(raw).strip()
        if not text:
            out.append(float("nan"))
            continue
        hit = CENSORED.match(text)
        if hit:
            sign, magnitude = hit.group(1), float(hit.group(2))
            if sign == "<":
                below += 1
                out.append(magnitude / 2.0)
            else:
                above += 1
                out.append(magnitude)
            continue
        try:
            out.append(float(text))
        except ValueError:
            out.append(float("nan"))
    return pd.Series(out, index=series.index, dtype=float), below, above


def read_wamex(path: str | Path) -> WamexTable:
    """Parse one WAMEX file into a frame plus its unit, method and metadata lines."""
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"WAMEX file not found: {p}")

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    header_row: list[str] | None = None
    unit_row: list[str] = []
    method_row: list[str] = []
    metadata: dict[str, str] = {}
    records: list[list[str]] = []

    for line in lines:
        if not line.strip() or line.startswith("EOF"):
            continue
        fields = line.split("\t")
        key = fields[0].strip()
        if key == COLUMN_LINE:
            header_row = fields
        elif key == UNIT_LINE:
            unit_row = fields
        elif key == METHOD_LINE:
            method_row = fields
        elif HEADER_PREFIX.match(key):
            values = [f.strip() for f in fields[2:] if f.strip()]
            if len(fields) > 1 and fields[1].strip():
                metadata[fields[1].strip()] = " | ".join(values) if values else ""
        elif key == DATA_PREFIX:
            records.append(fields)

    if header_row is None:
        raise IngestionError(
            f"{p.name} has no {COLUMN_LINE} column-name line, so it is not a WAMEX file")
    if not records:
        raise IngestionError(f"{p.name} has a {COLUMN_LINE} header but no D data rows")

    width = len(header_row)
    names = [f.strip() for f in header_row[1:]]
    unit_row = (unit_row + [""] * width)[:width]
    method_row = (method_row + [""] * width)[:width]
    units_by_pos = [f.strip() for f in unit_row[1:width]]
    methods_by_pos = [f.strip() for f in method_row[1:width]]

    rows = []
    ragged = 0
    for rec in records:
        body = rec[1:width]
        if len(body) != width - 1:
            ragged += 1
            body = (body + [""] * (width - 1))[:width - 1]
        rows.append([f.strip() for f in body])
    if ragged:
        log.warning("%s: %d data row(s) did not match the header width and were padded",
                    p.name, ragged)

    columns = _disambiguate(names, methods_by_pos, units_by_pos)
    frame = pd.DataFrame(rows, columns=columns).replace("", pd.NA)

    table = WamexTable(
        frame=frame,
        units=dict(zip(columns, units_by_pos)),
        methods=dict(zip(columns, methods_by_pos)),
        metadata=metadata,
        path=str(p),
    )
    project = metadata.get("Project_name", "")
    log.info("%s: WAMEX file, %d rows x %d columns%s", p.name, len(frame), len(columns),
             f", project {project!r}" if project else "")
    return table


def describe_elements(table: WamexTable) -> list[dict]:
    """Every candidate grade column, with the method and unit that define it."""
    out = []
    for col in table.element_columns():
        method = str(table.methods.get(col, "")).upper()
        out.append({
            "column": col,
            "element": col.split("_")[0],
            "unit": table.units.get(col, ""),
            "method": table.methods.get(col, ""),
            "digest": ("total" if method in TOTAL_DIGEST
                       else "partial" if method in PARTIAL_LEACH else "unknown"),
        })
    return out


def select_grade_column(table: WamexTable, element: str | None) -> tuple[str, str, str]:
    """Resolve which assayed column is the grade.

    Returns ``(column, unit, note)``. When one element was assayed by several
    methods the total digest wins over a partial leach, because a weak-acid
    result recovers only a fraction of the contained metal and reading it as the
    grade understates the deposit. The choice is always reported, never implied.
    """
    candidates = describe_elements(table)
    if not candidates:
        raise IngestionError(
            f"{Path(table.path).name} has no column carrying a concentration unit "
            f"({', '.join(sorted(CONCENTRATION_UNITS))}), so no grade column can be identified")

    if not element:
        listing = ", ".join(sorted({c["element"] for c in candidates}))
        raise IngestionError(
            f"{Path(table.path).name} is a multi-element assay file with "
            f"{len(candidates)} assayed columns, so the grade element has to be stated "
            f"rather than guessed. Set grade_element to one of: {listing}")

    wanted = str(element).strip().lower()
    exact = [c for c in candidates if c["column"].lower() == wanted]
    by_element = exact or [c for c in candidates if c["element"].lower() == wanted]
    if not by_element:
        listing = ", ".join(sorted({c["element"] for c in candidates}))
        raise IngestionError(
            f"grade element {element!r} is not assayed in {Path(table.path).name}; "
            f"available elements are: {listing}")

    if len(by_element) == 1:
        chosen = by_element[0]
        note = (f"grade taken from column {chosen['column']!r} "
                f"({chosen['unit']}, method {chosen['method'] or 'unstated'})")
        return chosen["column"], chosen["unit"], note

    total = [c for c in by_element if c["digest"] == "total"]
    unknown = [c for c in by_element if c["digest"] == "unknown"]
    chosen = (total or unknown or by_element)[0]
    rejected = [f"{c['column']} ({c['unit']}, {c['method']}, {c['digest']} digest)"
                for c in by_element if c["column"] != chosen["column"]]
    note = (f"{element} was assayed {len(by_element)} ways; the grade is taken from "
            f"{chosen['column']!r} ({chosen['unit']}, method "
            f"{chosen['method'] or 'unstated'}, {chosen['digest']} digest) "
            f"in preference to {', '.join(rejected)}")
    return chosen["column"], chosen["unit"], note
