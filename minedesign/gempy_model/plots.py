"""Cross-sections through the implicit GemPy model.

The point of these drawings is comparison. Each one puts the implicit solid, the
explicit section wireframe, the contact points that constrained the interpolation
and the drillholes themselves on the same axes, so the two orebody models can be
judged against each other and against the data rather than admired separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..report.svg_canvas import INK, SvgCanvas

log = get_logger(__name__)

GEMPY_FILL = "#00e5c8"
CONTACT_TOP = "#ffd166"
CONTACT_BASE = "#ff8a3d"


def _slice_indices(extent: list[float], resolution: list[int], axis: int,
                   world_value: float) -> int | None:
    """Grid index of the slice nearest a world coordinate on one axis."""
    lo, hi = extent[axis * 2], extent[axis * 2 + 1]
    n = resolution[axis]
    if hi - lo <= 0:
        return None
    idx = int(round((world_value - lo) / (hi - lo) * n - 0.5))
    return int(np.clip(idx, 0, n - 1))


def implicit_section(result, section_id: int, composites: pd.DataFrame, frame,
                     boundaries: list[dict] | None = None,
                     width: int = 1000, height: int = 620,
                     half_thickness: float | None = None) -> str | None:
    """One drill section through the implicit model, with everything overlaid."""
    if not result.available or result.lith_block is None or result.grid_shape is None:
        return None

    section = composites[composites["section_id"] == section_id]
    if section.empty:
        return None
    section_s = float(section["section_s"].iloc[0])
    half = half_thickness if half_thickness is not None else frame.spacing / 2.0

    near = composites[np.abs(composites["s"] - section_s) <= half]
    if near.empty:
        near = section

    nx, ny, nz = result.grid_shape
    extent = result.extent
    ids = np.round(np.asarray(result.lith_block, float)).astype(int).reshape(nx, ny, nz)

    # the ore unit is whichever id the builder settled on: recover it by majority
    # inside the modelled solid, which is the same rule the builder used
    unique, counts = np.unique(ids, return_counts=True)
    ore_id = None
    if result.n_ore_cells:
        for candidate, count in zip(unique, counts):
            if int(count) == int(result.n_ore_cells):
                ore_id = int(candidate)
                break
    if ore_id is None:
        ore_id = int(unique[np.argmin(counts)])

    # sample the implicit grid on the section plane
    u_axis = np.linspace(near["u"].min() - 60.0, near["u"].max() + 60.0, 220)
    v_axis = np.linspace(near["v"].min() - 60.0, near["v"].max() + 60.0, 150)
    uu, vv = np.meshgrid(u_axis, v_axis, indexing="ij")
    world = frame.to_world(np.column_stack([
        np.full(uu.size, section_s), uu.ravel(), vv.ravel()]))

    i = np.clip(((world[:, 0] - extent[0]) / max(extent[1] - extent[0], 1e-9) * nx
                 ).astype(int), 0, nx - 1)
    j = np.clip(((world[:, 1] - extent[2]) / max(extent[3] - extent[2], 1e-9) * ny
                 ).astype(int), 0, ny - 1)
    k = np.clip(((world[:, 2] - extent[4]) / max(extent[5] - extent[4], 1e-9) * nz
                 ).astype(int), 0, nz - 1)
    inside_extent = ((world[:, 0] >= extent[0]) & (world[:, 0] <= extent[1])
                     & (world[:, 1] >= extent[2]) & (world[:, 1] <= extent[3])
                     & (world[:, 2] >= extent[4]) & (world[:, 2] <= extent[5]))
    is_ore = (ids[i, j, k] == ore_id) & inside_extent

    bounds = (u_axis.min(), v_axis.min(), u_axis.max(), v_axis.max())
    canvas = SvgCanvas(width, height, bounds)

    # implicit solid, drawn as the filled cells on this slice
    du = float(u_axis[1] - u_axis[0])
    dv = float(v_axis[1] - v_axis[0])
    sel = np.flatnonzero(is_ore)
    if sel.size:
        canvas.rects(uu.ravel()[sel], vv.ravel()[sel], du * 1.05, dv * 1.05,
                     [GEMPY_FILL] * sel.size, opacity=0.30)

    # explicit wireframe outline, for direct comparison
    if boundaries:
        for b in boundaries:
            if int(b.get("section_id", -999)) != int(section_id):
                continue
            local = frame.to_local(np.asarray(b["boundary_points"], float))
            canvas.polyline(local[:, 1], local[:, 2], "#ff5c8a", 2.0, closed=True,
                            dash="7,4")

    # drillholes, thin and neutral so the models stay readable
    seg = near.dropna(subset=["x_from", "x_to"])
    if not seg.empty:
        a = frame.to_local(seg[["x_from", "y_from", "z_from"]].to_numpy(float))
        b = frame.to_local(seg[["x_to", "y_to", "z_to"]].to_numpy(float))
        ore_flag = (seg["is_ore_sample"].to_numpy(bool)
                    if "is_ore_sample" in seg.columns
                    else np.zeros(len(seg), bool))
        colours = ["#e8e8e8" if flag else "#46536220" for flag in ore_flag]
        canvas.segments(a[:, 1], a[:, 2], b[:, 1], b[:, 2], colours, width=2.2)

    # the contact points that constrained the interpolation
    if result.contacts is not None and not result.contacts.points.empty:
        pts = result.contacts.points
        local = frame.to_local(pts[["x", "y", "z"]].to_numpy(float))
        on_section = np.abs(local[:, 0] - section_s) <= half
        crossing = pts["crossing"].to_numpy() if "crossing" in pts.columns else None
        if on_section.any():
            if crossing is not None:
                top = on_section & (crossing == "hanging_wall")
                base = on_section & (crossing == "footwall")
                canvas.circles(local[top, 1], local[top, 2], 3.2, CONTACT_TOP)
                canvas.circles(local[base, 1], local[base, 2], 3.2, CONTACT_BASE)
            else:
                canvas.circles(local[on_section, 1], local[on_section, 2], 3.2, CONTACT_TOP)

    canvas.frame("Across strike (m)", "Elevation relative to collar datum (m)")
    canvas.title_block(
        f"IMPLICIT MODEL - SECTION {section_id}",
        f"{section_s:+,.0f} m along strike | GemPy potential-field solid vs "
        f"section wireframe | slab {2 * half:.0f} m")
    canvas.scale_bar()
    canvas.legend([(GEMPY_FILL, "GemPy implicit solid"),
                   ("#ff5c8a", "section wireframe"),
                   (CONTACT_TOP, "hanging-wall contact"),
                   (CONTACT_BASE, "footwall contact"),
                   ("#e8e8e8", "ore intercept")])
    return canvas.render()


def orientation_rose(result, width: int = 460, height: int = 380) -> str | None:
    """Dip and dip-azimuth of every fitted orientation, on a polar plot.

    Worth drawing because it shows at a glance whether the normals really do
    wrap a closed body: on a lens they spread across the full circle and split
    between upward and downward, which a single open surface would not do.
    """
    if result.contacts is None or result.contacts.orientations.empty:
        return None
    ori = result.contacts.orientations

    canvas = SvgCanvas(width, height, (-1.15, -1.15, 1.15, 1.15), margin=40)
    cx, cy = float(canvas.sx(0.0)), float(canvas.sy(0.0))
    radius = float(canvas.sx(1.0)) - cx

    for ring, label in ((1.0, "90"), (2 / 3, "60"), (1 / 3, "30")):
        canvas.parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius * ring:.1f}" fill="none" '
            f'stroke="{INK["grid"]}" stroke-width="0.8"/>')
        canvas.text(cx + 3, cy - radius * ring - 3, f"{label}°", size=8,
                    screen=True, fill=INK["text_faint"] if "text_faint" in INK
                    else INK["text"])
    for angle, label in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        rad = np.radians(angle)
        x = cx + (radius + 14) * np.sin(rad)
        y = cy - (radius + 14) * np.cos(rad)
        canvas.text(x, y + 3, label, size=10, anchor="middle", screen=True,
                    fill=INK["text_bright"])
        canvas.parts.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" '
            f'x2="{cx + radius * np.sin(rad):.1f}" y2="{cy - radius * np.cos(rad):.1f}" '
            f'stroke="{INK["grid"]}" stroke-width="0.8"/>')

    for row in ori.itertuples(index=False):
        rad = np.radians(float(row.azimuth))
        r = radius * float(row.dip) / 90.0
        x = cx + r * np.sin(rad)
        y = cy - r * np.cos(rad)
        colour = GEMPY_FILL if float(row.G_z) >= 0 else CONTACT_BASE
        canvas.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{colour}" '
                            f'fill-opacity="0.9"/>')

    canvas.title_block("FITTED ORIENTATIONS",
                       "radius = dip, bearing = dip azimuth")
    canvas.legend([(GEMPY_FILL, "normal points up"), (CONTACT_BASE, "normal points down")],
                  x=14, y=height - 34)
    return canvas.render()
