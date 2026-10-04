"""Build the 3D scene payload for the report viewer.

Every spatial figure becomes a toggleable layer in one orbitable scene rather
than a separate static image. Coordinates are shifted to a local origin before
serialisation: mine grids run to six or seven figures, and float32 in the
browser cannot hold that without visible jitter.
"""
from __future__ import annotations

import colorsys

import numpy as np
import pandas as pd

from ..geology.stats import color_for_grade, rock_type_colors
from ..logging_utils import get_logger
from ..pit.lerchs_grossmann import bench_contours

log = get_logger(__name__)

MAX_BLOCKS_IN_SCENE = 60_000
MAX_TUBE_SEGMENTS = 40_000


def _level_colour(hue_deg: float) -> str:
    """A vivid, fully-saturated colour at the given hue, CAD-layer style."""
    r, g, b = colorsys.hsv_to_rgb((hue_deg % 360.0) / 360.0, 0.85, 0.95)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _round(values, nd=2):
    return [float(np.round(v, nd)) for v in np.asarray(values, float).ravel()]


class SceneBuilder:
    """Accumulates layers, all sharing one local origin."""

    def __init__(self, origin: np.ndarray):
        self.origin = np.asarray(origin, float)
        self.layers: dict[str, dict] = {}

    def local(self, xyz) -> np.ndarray:
        return np.asarray(xyz, float).reshape(-1, 3) - self.origin

    def add(self, name: str, layer: dict, *, label: str, kind: str,
            visible: bool = True, group: str = "general") -> None:
        layer.update({"label": label, "kind": kind, "visible": visible, "group": group})
        self.layers[name] = layer

    def as_dict(self) -> dict:
        return {"origin": _round(self.origin, 3), "layers": self.layers}


def drillhole_layer(builder: SceneBuilder, traces: pd.DataFrame,
                    intervals: pd.DataFrame, ramp: dict,
                    rock_colors: dict[str, str]) -> None:
    """Drillholes as coloured segments, switchable between grade and rock type."""
    segments = []
    grade_colors = []
    rock_seg_colors = []

    df = intervals.dropna(subset=["x_from", "x_to"])
    if len(df) > MAX_TUBE_SEGMENTS:
        step = int(np.ceil(len(df) / MAX_TUBE_SEGMENTS))
        df = df.iloc[::step]
        log.info("drillhole layer decimated to every %dth interval for the browser", step)

    start = builder.local(df[["x_from", "y_from", "z_from"]].to_numpy(float))
    end = builder.local(df[["x_to", "y_to", "z_to"]].to_numpy(float))
    for a, b in zip(start, end):
        segments.extend([*a, *b])

    for row in df.itertuples(index=False):
        grade_colors.append(color_for_grade(float(getattr(row, "grade", np.nan)), ramp))
        rock_seg_colors.append(rock_colors.get(str(getattr(row, "rock_type", "UNKNOWN")), "#9e9e9e"))

    # the hole traces themselves, so empty collar-to-first-sample runs still show
    trace_lines = []
    for _, grp in traces.groupby("hole_id", sort=False):
        pts = builder.local(grp[["x", "y", "z"]].to_numpy(float))
        for a, b in zip(pts[:-1], pts[1:]):
            trace_lines.extend([*a, *b])

    builder.add("drillholes", {
        "segments": _round(segments),
        "grade_colors": grade_colors,
        "rock_colors": rock_seg_colors,
        "traces": _round(trace_lines),
        "n_intervals": int(len(df)),
        "radius": 1.5,
    }, label="Drillholes", kind="tubes", group="drilling")


def sample_points_layer(builder: SceneBuilder, name: str, df: pd.DataFrame,
                        ramp: dict, label: str, visible: bool = False,
                        group: str = "samples", grade_column: str = "grade") -> None:
    """A point cloud of interval midpoints, coloured by grade."""
    df = df.dropna(subset=["x", "y", "z"])
    if df.empty:
        return
    pts = builder.local(df[["x", "y", "z"]].to_numpy(float))
    colors = [color_for_grade(float(g), ramp)
              for g in pd.to_numeric(df[grade_column], errors="coerce").to_numpy(float)]
    builder.add(name, {
        "positions": _round(pts),
        "colors": colors,
        "size": 3.0,
        "n_points": int(len(df)),
    }, label=label, kind="points", visible=visible, group=group)


def top_cut_layer(builder: SceneBuilder, capped: pd.DataFrame, ramp: dict) -> None:
    """Capped composites highlighted against the uncapped population."""
    if "capped" not in capped.columns:
        return
    flagged = capped[capped["capped"]].dropna(subset=["x", "y", "z"])
    if flagged.empty:
        return
    pts = builder.local(flagged[["x", "y", "z"]].to_numpy(float))
    builder.add("top_cut_affected", {
        "positions": _round(pts),
        "colors": ["#ff1744"] * len(flagged),
        "size": 6.0,
        "n_points": int(len(flagged)),
        "uncapped_values": _round(flagged["grade_uncapped"].to_numpy(float), 3)
        if "grade_uncapped" in flagged.columns else [],
    }, label="Composites affected by top cut", kind="points", visible=False, group="samples")


def section_layer(builder: SceneBuilder, boundaries: list[dict], frame) -> None:
    """Digitised section boundaries, as closed polylines plus per-section metadata."""
    polylines = []
    sections_meta = []
    for b in boundaries:
        pts = builder.local(np.asarray(b["boundary_points"], float))
        closed = np.vstack([pts, pts[:1]])
        flat = []
        for a, c in zip(closed[:-1], closed[1:]):
            flat.extend([*a, *c])
        polylines.append(_round(flat))
        sections_meta.append({
            "section_id": int(b.get("section_id", 0)),
            "section_s": float(b.get("section_s", 0.0)),
            "n_points": int(len(pts)),
        })

    builder.add("sections", {
        "polylines": polylines,
        "sections": sections_meta,
        "strike_azimuth_deg": float(frame.strike_azimuth_deg),
        "strike_dir": _round(frame.strike_dir, 6),
        "across_dir": _round(frame.across_dir, 6),
        "origin_world": _round(frame.origin, 3),
        "color": "#ffd54f",
    }, label="Section boundaries", kind="polylines", group="domaining")


def solid_layer(builder: SceneBuilder, solid) -> None:
    """The orebody wireframe as a dark, solid-reading mesh."""
    verts = builder.local(solid.vertices)
    builder.add("orebody_solid", {
        "vertices": _round(verts, 3),
        "faces": [int(i) for i in solid.faces.ravel()],
        "color": "#0d7377",
        "opacity": 0.85,
        "n_faces": int(len(solid.faces)),
    }, label="Orebody solid", kind="mesh", group="domaining")


def block_model_layer(builder: SceneBuilder, blocks: pd.DataFrame, ramp: dict,
                      block_size, name: str = "block_model",
                      label: str = "Block model (ore)", visible: bool = True,
                      only_ore: bool = True) -> None:
    """Instanced cubes coloured by grade."""
    df = blocks
    if only_ore and "is_ore" in df.columns:
        df = df[df["is_ore"]]
    df = df.dropna(subset=["grade"])
    if df.empty:
        return

    if len(df) > MAX_BLOCKS_IN_SCENE:
        df = df.nlargest(MAX_BLOCKS_IN_SCENE, "grade")
        log.info("block layer %s limited to the %d highest-grade blocks",
                 name, MAX_BLOCKS_IN_SCENE)

    pts = builder.local(df[["x", "y", "z"]].to_numpy(float))
    grades = df["grade"].to_numpy(float)
    builder.add(name, {
        "positions": _round(pts),
        "colors": [color_for_grade(float(g), ramp) for g in grades],
        "grades": _round(grades, 4),
        "block_size": [float(v) for v in block_size],
        "n_blocks": int(len(df)),
    }, label=label, kind="blocks", visible=visible, group="block model")


def pit_layer(builder: SceneBuilder, blocks: pd.DataFrame, in_pit: np.ndarray,
              surface: dict | None, block_size) -> None:
    """The ultimate pit: its mined blocks and the pit floor surface."""
    if in_pit is None or not in_pit.any():
        return

    pit_blocks = blocks[in_pit]
    sample = pit_blocks
    if len(sample) > MAX_BLOCKS_IN_SCENE:
        sample = sample.sample(MAX_BLOCKS_IN_SCENE, random_state=0)
    pts = builder.local(sample[["x", "y", "z"]].to_numpy(float))
    builder.add("pit_blocks", {
        "positions": _round(pts),
        "colors": ["#ff8a65"] * len(sample),
        "block_size": [float(v) for v in block_size],
        "n_blocks": int(len(pit_blocks)),
        "opacity": 0.25,
    }, label="Blocks inside the pit", kind="blocks", visible=False, group="pit")

    if surface:
        bands = bench_contours(surface, n_levels=24)
        if bands:
            polylines = []
            colors = []
            for i, band in enumerate(bands):
                # bands are ordered floor to crest (ascending elevation); the
                # hue sweep runs magenta at the floor up to red at the crest,
                # the same warm-high/cool-low convention as an elevation ramp
                hue = 300.0 * (1.0 - i / max(len(bands) - 1, 1))
                colour = _level_colour(hue)
                for loop in band["loops"]:
                    z_col = np.full(len(loop), band["z"])
                    world = np.column_stack([loop[:, 0], loop[:, 1], z_col])
                    closed = np.vstack([world, world[:1]])
                    local = builder.local(closed)
                    flat = []
                    for a, b in zip(local[:-1], local[1:]):
                        flat.extend([*a, *b])
                    polylines.append(_round(flat))
                    colors.append(colour)

            builder.add("pit_shell", {
                "polylines": polylines,
                "colors": colors,
                "n_levels": len(bands),
            }, label="Ultimate pit shell (bench contours)", kind="banded_polylines",
               group="pit")


def topography_layer(builder: SceneBuilder, topo, bbox: np.ndarray, n: int = 40) -> None:
    """A translucent topographic surface, for context under the pit."""
    xs = np.linspace(bbox[0][0], bbox[1][0], n)
    ys = np.linspace(bbox[0][1], bbox[1][1], n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    z = topo.elevation(np.column_stack([gx.ravel(), gy.ravel()]))
    z = np.asarray(z, float).reshape(n, n)

    vertices = []
    for a in range(n):
        for b in range(n):
            local = builder.local([[xs[a], ys[b], z[a, b]]])[0]
            vertices.extend(_round(local, 3))
    faces = []
    for a in range(n - 1):
        for b in range(n - 1):
            v0 = a * n + b
            v1 = (a + 1) * n + b
            v2 = (a + 1) * n + b + 1
            v3 = a * n + b + 1
            faces.extend([v0, v1, v2, v0, v2, v3])

    builder.add("topography", {
        "vertices": vertices, "faces": faces,
        "color": "#8d9e6a", "opacity": 0.25, "n_faces": len(faces) // 3,
    }, label="Topography", kind="mesh", visible=False, group="general")


def build_scene(traces, composited_raw, composited, capped, boundaries, frame,
                solid, blocks, in_pit, pit_surface_data, ramp, block_size,
                topo=None, bbox=None, gempy_result=None) -> dict:
    """Assemble every spatial layer into one scene."""
    origin = np.asarray(frame.origin, float)
    builder = SceneBuilder(origin)

    rock_colors = rock_type_colors(
        capped["rock_type"].unique() if "rock_type" in capped.columns else [])

    drillhole_layer(builder, traces, capped, ramp, rock_colors)
    sample_points_layer(builder, "raw_samples", composited_raw, ramp,
                        "Raw assay samples (pre-composite)", visible=False)
    sample_points_layer(builder, "composites", composited, ramp,
                        "Composited intervals", visible=False)
    sample_points_layer(builder, "composites_topcut", capped, ramp,
                        "Composites after top cut", visible=False)
    top_cut_layer(builder, capped, ramp)

    if boundaries:
        section_layer(builder, boundaries, frame)
    if solid is not None:
        solid_layer(builder, solid)
    if blocks is not None and not blocks.empty:
        block_model_layer(builder, blocks, ramp, block_size)
        block_model_layer(builder, blocks, ramp, block_size, name="block_model_all",
                          label="Block model (all estimated blocks)", visible=False,
                          only_ore=False)
    if in_pit is not None:
        pit_layer(builder, blocks, in_pit, pit_surface_data, block_size)
    if topo is not None and bbox is not None:
        topography_layer(builder, topo, bbox)
    gempy_layers(builder, gempy_result)

    scene = builder.as_dict()
    scene["rock_colors"] = rock_colors
    scene["grade_ramp"] = ramp
    log.info("scene built with %d layers", len(scene["layers"]))
    return scene


def gempy_layers(builder: SceneBuilder, result) -> None:
    """Add the implicit model to the scene, alongside the explicit wireframe.

    Both orebody models live in one scene on purpose: the useful question is not
    what either looks like alone, but where they disagree.
    """
    if result is None or not getattr(result, "available", False):
        return

    for mesh in result.meshes:
        verts = builder.local(mesh["vertices"])
        builder.add(f"gempy_{mesh['name']}", {
            "vertices": _round(verts, 3),
            "faces": [int(i) for i in np.asarray(mesh["faces"], int).ravel()],
            "color": "#00e5c8",
            "opacity": 0.42,
            "n_faces": int(mesh["n_faces"]),
        }, label="GemPy implicit surface", kind="mesh", group="gempy model")

    contacts = getattr(result, "contacts", None)
    if contacts is not None and not contacts.points.empty:
        pts = contacts.points
        local = builder.local(pts[["x", "y", "z"]].to_numpy(float))
        crossing = (pts["crossing"].to_numpy() if "crossing" in pts.columns
                    else np.full(len(pts), "hanging_wall"))
        colours = ["#ffd166" if c == "hanging_wall" else "#ff8a3d" for c in crossing]
        builder.add("gempy_contacts", {
            "positions": _round(local),
            "colors": colours,
            "size": 7.0,
            "n_points": int(len(pts)),
        }, label="Ore contact points", kind="points", visible=False, group="gempy model")

    if contacts is not None and not contacts.orientations.empty:
        ori = contacts.orientations
        start = builder.local(ori[["x", "y", "z"]].to_numpy(float))
        normals = ori[["G_x", "G_y", "G_z"]].to_numpy(float)
        length = 24.0
        segments = []
        for a, n in zip(start, normals):
            b = a + n * length
            segments.extend([*a, *b])
        builder.add("gempy_orientations", {
            "segments": _round(segments),
            "grade_colors": ["#ffffff"] * len(start),
            "rock_colors": ["#ffffff"] * len(start),
            "traces": [],
            "n_intervals": int(len(start)),
        }, label="Fitted orientations", kind="tubes", visible=False, group="gempy model")
