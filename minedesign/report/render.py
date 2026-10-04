"""Render the final self-contained HTML report.

The narrative comes from LLM Call 2; every number, table row, chart value and
geometry layer comes from a tool output and is injected here. That split is the
point: the model writes prose, the pipeline supplies facts, and no figure in the
document can be a fabrication.

Three.js, its orbit controls and Chart.js are vendored into the package and
inlined, so the report is a single file that works with no network at all.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..logging_utils import get_logger

log = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
VENDOR_DIR = Path(__file__).parent / "vendor"

MAX_TABLE_ROWS = 5000  # what a browser can filter without stalling


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _clean(obj):
    """Replace non-finite floats with None so the JSON is valid for a browser."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    return obj


def to_json(obj) -> str:
    return json.dumps(_clean(obj), default=_json_default, separators=(",", ":"))


def _read_vendor(name: str) -> str:
    path = VENDOR_DIR / name
    if not path.exists():
        log.warning("vendored asset %s is missing; the report will lack that feature", name)
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def dataframe_to_table(df: pd.DataFrame, columns: list[str], rounding: dict | None = None,
                       max_rows: int = MAX_TABLE_ROWS) -> dict:
    """Serialise a DataFrame into a compact rows payload for the HTML tables."""
    rounding = rounding or {}
    present = [c for c in columns if c in df.columns]
    view = df[present]

    truncated = False
    if len(view) > max_rows:
        view = view.head(max_rows)
        truncated = True

    rows = []
    for record in view.itertuples(index=False):
        row = []
        for col, value in zip(present, record):
            if isinstance(value, (float, np.floating)):
                value = float(value)
                row.append(None if not np.isfinite(value) else round(value, rounding.get(col, 3)))
            elif isinstance(value, (np.integer,)):
                row.append(int(value))
            elif isinstance(value, (bool, np.bool_)):
                row.append(bool(value))
            elif value is None or (isinstance(value, float) and not np.isfinite(value)):
                row.append(None)
            else:
                row.append(str(value))
        rows.append(row)

    return {
        "columns": present,
        "rows": rows,
        "n_total": int(len(df)),
        "n_shown": len(rows),
        "truncated": truncated,
    }


def build_environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["tojson_safe"] = to_json
    env.filters["fmt"] = _format_number
    return env


def _format_number(value, decimals: int = 2, default: str = "n/a") -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return f"{value:,.{decimals}f}"


def render_report(context: dict, output_path: str | Path) -> Path:
    """Render and write the single-file HTML report."""
    env = build_environment()
    template = env.get_template("report.html.j2")

    payload = {
        "scene": context.get("scene", {}),
        "charts": context.get("charts", {}),
        "tables": context.get("tables", {}),
    }

    html = template.render(
        meta=context.get("meta", {}),
        report=context.get("report", {}),
        economics=context.get("economics", {}),
        assumptions=context.get("assumptions", []),
        flags=context.get("flags", []),
        provenance=context.get("provenance", {}),
        summary=context.get("summary", {}),
        validation=context.get("validation", {}),
        stats=context.get("stats", {}),
        gempy=context.get("gempy", {"enabled": False}),
        bench_schedule=context.get("bench_schedule", {"benches": [], "n_benches": 0}),
        payload_json=to_json(payload),
        three_js=_read_vendor("three.min.js"),
        orbit_js=_read_vendor("OrbitControls.js"),
        chart_js=_read_vendor("chart.umd.js"),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    log.info("report written to %s (%.1f MB, self-contained)", out, size_mb)
    return out
