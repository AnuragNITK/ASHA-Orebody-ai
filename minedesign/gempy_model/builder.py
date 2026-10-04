"""Build an implicit orebody model with GemPy.

This is a second, independent orebody model, and deliberately not the one the
rest of the pipeline uses. The difference matters:

  section wireframe   an explicit surface. Ore points are enveloped per drill
                      section and the sections are lofted together. The shape is
                      whatever the digitised outlines say it is.
  GemPy model         an implicit surface. A scalar potential field is
                      interpolated through the contact points, honouring the
                      measured orientations, and the orebody is an isosurface of
                      that field. The shape is a consequence of the data and the
                      interpolation, with no digitising step.

No language model is involved at any point here. Where the wireframe route asks
an LLM for a cutoff and a set of boundary points, this route takes the same ore
flags and solves a co-kriging system for the potential field.

The orebody is modelled as ONE closed contact surface with outward-pointing
orientations, so the solid is the volume the surface encloses. Modelling the
hanging wall and footwall as two conformable surfaces instead forces them
parallel, and a lens-shaped body collapses into a sliver.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .contacts import ORE_SURFACE, ContactSet

log = get_logger(__name__)

ORE_UNIT_NAME = "orebody"


class GemPyUnavailable(RuntimeError):
    """GemPy is not installed, or could not be imported."""


def gempy_version() -> str | None:
    try:
        import gempy as gp
    except Exception:  # noqa: BLE001
        return None
    return getattr(gp, "__version__", "unknown")


@dataclass
class GemPyResult:
    """Everything the report needs from the implicit model."""

    available: bool
    version: str | None = None
    extent: list[float] = field(default_factory=list)
    resolution: list[int] = field(default_factory=list)
    lith_block: np.ndarray | None = None
    scalar_field: np.ndarray | None = None
    grid_shape: tuple[int, int, int] | None = None
    meshes: list[dict] = field(default_factory=list)     # {name, vertices, faces}
    ore_volume_m3: float = 0.0
    ore_tonnes: float = 0.0
    ore_grade: float | None = None
    contained_metal_oz: float | None = None
    n_ore_cells: int = 0
    cell_volume_m3: float = 0.0
    contacts: ContactSet | None = None
    compute_seconds: float = 0.0
    ore_capture_fraction: float | None = None
    comparison: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> dict:
        counts = self.contacts.counts() if self.contacts else {}
        return {
            "available": self.available,
            "version": self.version,
            "extent": [round(float(v), 1) for v in self.extent],
            "resolution": list(self.resolution),
            "n_grid_cells": int(np.prod(self.resolution)) if self.resolution else 0,
            "cell_volume_m3": round(self.cell_volume_m3, 2),
            "n_ore_cells": self.n_ore_cells,
            "ore_volume_m3": round(self.ore_volume_m3, 0),
            "ore_tonnes": round(self.ore_tonnes, 0),
            "ore_grade": None if self.ore_grade is None else round(self.ore_grade, 4),
            "contained_metal_oz": (None if self.contained_metal_oz is None
                                   else round(self.contained_metal_oz, 0)),
            "compute_seconds": round(self.compute_seconds, 2),
            "ore_capture_pct": (None if self.ore_capture_fraction is None
                                else round(self.ore_capture_fraction * 100.0, 1)),
            "n_surfaces": len(self.meshes),
            "warnings": self.warnings,
            "error": self.error,
            **counts,
        }


def _grid_extent(points: pd.DataFrame, margin_fraction: float = 0.15) -> list[float]:
    """Bounding box around the contacts, padded so surfaces are not clipped."""
    xyz = points[["x", "y", "z"]].to_numpy(float)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    pad = span * margin_fraction
    lo, hi = lo - pad, hi + pad
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]),
            float(lo[2]), float(hi[2])]


def _choose_resolution(extent: list[float], target_cells: int = 120_000,
                       max_per_axis: int = 80,
                       min_thin_axis_cells: int = 32) -> list[int]:
    """Near-isotropic voxels, but never at the cost of the thinnest axis.

    An orebody is almost always far thinner than it is wide. Sizing all three
    axes from a single isotropic cell and then clipping starves the thin one: a
    10 km x 10 km x 240 m contact cloud lands on 8 vertical layers, and a body
    quantised into 8 layers renders as a flat slab rather than a volume. The
    wide axes keep the isotropic sizing, and only the thin axis is refined, so
    the extra cells are spent where the geometry actually needs them.
    """
    span = np.array([extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4]])
    span = np.maximum(span, 1.0)

    cell = float((span.prod() / target_cells) ** (1 / 3))
    res = np.clip(np.round(span / cell).astype(int), 8, max_per_axis)

    thin = int(np.argmin(span))
    res[thin] = int(np.clip(max(res[thin], min_thin_axis_cells), 8, max_per_axis))
    return [int(v) for v in res]


def build_gempy_model(contact_set: ContactSet, project_name: str = "orebody",
                      resolution: list[int] | None = None,
                      target_cells: int = 120_000,
                      density: float = 2.7,
                      blocks: pd.DataFrame | None = None,
                      wireframe_volume: float | None = None,
                      ore_sample_xyz: np.ndarray | None = None) -> GemPyResult:
    """Interpolate the implicit orebody and report its geometry.

    Returns a result object in every case: when GemPy is missing or the contact
    data is too thin, the result carries the reason rather than raising, so the
    report can state plainly why the section is empty.
    """
    version = gempy_version()
    if version is None:
        return GemPyResult(
            available=False, contacts=contact_set,
            error="GemPy is not installed. Install it with: python -m pip install gempy")

    if not contact_set.is_usable:
        return GemPyResult(
            available=False, version=version, contacts=contact_set,
            warnings=contact_set.warnings,
            error="the drilling does not provide enough ore contacts and orientations "
                  "to interpolate an implicit surface")

    import gempy as gp
    from gempy.core.data.structural_element import StructuralElement
    from gempy.core.data.structural_frame import StructuralFrame
    from gempy.core.data.structural_group import StackRelationType, StructuralGroup

    points = contact_set.points
    orientations = contact_set.orientations
    extent = _grid_extent(points)
    resolution = resolution or _choose_resolution(extent, target_cells)

    # One closed surface, not a hanging wall and a footwall as separate elements.
    # Two elements in one series share a scalar field and are therefore forced
    # roughly parallel, which collapses a lens-shaped body into a sliver.
    surface_points = gp.data.SurfacePointsTable.from_arrays(
        x=points["x"].to_numpy(float), y=points["y"].to_numpy(float),
        z=points["z"].to_numpy(float), names=ORE_SURFACE)
    orientation_table = gp.data.OrientationsTable.from_arrays(
        x=orientations["x"].to_numpy(float), y=orientations["y"].to_numpy(float),
        z=orientations["z"].to_numpy(float),
        G_x=orientations["G_x"].to_numpy(float), G_y=orientations["G_y"].to_numpy(float),
        G_z=orientations["G_z"].to_numpy(float), names=ORE_SURFACE)
    elements = [StructuralElement(
        name=ORE_SURFACE, surface_points=surface_points,
        orientations=orientation_table, color="#00e5c8")]

    group = StructuralGroup(name="ore_series", elements=elements,
                            structural_relation=StackRelationType.ERODE)
    frame = StructuralFrame(structural_groups=[group],
                            color_gen=gp.data.ColorsGenerator())

    started = time.perf_counter()
    try:
        model = gp.create_geomodel(project_name=project_name, extent=extent,
                                   resolution=resolution, structural_frame=frame)
        gp.compute_model(model)
    except Exception as exc:  # noqa: BLE001
        log.error("GemPy computation failed: %s", exc)
        return GemPyResult(available=False, version=version, contacts=contact_set,
                           extent=extent, resolution=resolution,
                           warnings=contact_set.warnings,
                           error=f"GemPy failed to compute the model: {exc}")
    elapsed = time.perf_counter() - started

    raw = model.solutions.raw_arrays
    lith = np.asarray(raw.lith_block, float) if raw.lith_block is not None else None
    scalar = (np.asarray(raw.scalar_field_matrix, float)[0]
              if raw.scalar_field_matrix is not None else None)

    nx, ny, nz = resolution
    cell_volume = (((extent[1] - extent[0]) / nx) * ((extent[3] - extent[2]) / ny)
                   * ((extent[5] - extent[4]) / nz))

    # One interface splits the volume in two. Which side is ore is settled by
    # the data rather than assumed: the side holding more of the logged ore
    # intercepts is the orebody.
    ore_cells = 0
    ore_mask_grid = None
    capture = None
    if lith is not None and lith.size:
        ids = np.round(lith).astype(int)
        ore_id, capture = _identify_ore_unit(ids, extent, resolution, ore_sample_xyz)
        if ore_id is not None:
            ore_mask_grid = ids == ore_id
            ore_cells = int(ore_mask_grid.sum())

    if capture is not None and capture < 0.5:
        contact_set.warnings.append(
            f"the implicit solid contains only {capture:.0%} of the logged ore "
            f"intercepts; treat the volume as indicative and review the contacts")

    meshes = []
    if raw.vertices is not None and raw.edges is not None:
        names = [ORE_SURFACE]
        for i, (verts, tris) in enumerate(zip(raw.vertices, raw.edges)):
            verts = np.asarray(verts, float)
            tris = np.asarray(tris, int)
            if len(verts) == 0 or len(tris) == 0:
                continue
            # GemPy solves in an internally centred and scaled space, so the mesh
            # comes back in roughly unit coordinates. Without the inverse
            # transform the surface lands at the origin, kilometres from the
            # deposit, and silently wrecks the extents of any scene it joins.
            try:
                verts = np.asarray(model.input_transform.apply_inverse(verts.copy()), float)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not map the GemPy mesh back to world coordinates: %s", exc)
                continue
            meshes.append({"name": names[i] if i < len(names) else f"surface_{i}",
                           "vertices": verts, "faces": tris,
                           "n_vertices": int(len(verts)), "n_faces": int(len(tris))})

    result = GemPyResult(
        available=True, version=version, extent=extent, resolution=list(resolution),
        lith_block=lith, scalar_field=scalar, grid_shape=(nx, ny, nz),
        meshes=meshes, n_ore_cells=ore_cells, cell_volume_m3=cell_volume,
        ore_volume_m3=ore_cells * cell_volume,
        ore_tonnes=ore_cells * cell_volume * density,
        contacts=contact_set, compute_seconds=elapsed,
        ore_capture_fraction=capture,
        warnings=list(contact_set.warnings))

    # grade the implicit solid from the same block estimates the wireframe used,
    # so the two volumes are compared on identical grade information
    if blocks is not None and ore_mask_grid is not None:
        result.ore_grade, result.contained_metal_oz = _grade_implicit_solid(
            blocks, ore_mask_grid, extent, resolution, result.ore_tonnes)

    if wireframe_volume:
        result.comparison = _compare(result.ore_volume_m3, wireframe_volume)

    log.info("GemPy model: %d ore cells, %.0f m3 (%.2f Mt) in %.1fs, %d surface mesh(es)",
             ore_cells, result.ore_volume_m3, result.ore_tonnes / 1e6, elapsed, len(meshes))
    return result


def _grade_implicit_solid(blocks: pd.DataFrame, ore_mask_grid: np.ndarray,
                          extent: list[float], resolution: list[int],
                          tonnes: float) -> tuple[float | None, float | None]:
    """Mean grade inside the implicit solid, sampled from the block model."""
    grade = pd.to_numeric(blocks.get("grade"), errors="coerce")
    if grade is None or not np.isfinite(grade).any():
        return None, None

    nx, ny, nz = resolution
    grid = ore_mask_grid.reshape(nx, ny, nz)

    xyz = blocks[["x", "y", "z"]].to_numpy(float)
    i = ((xyz[:, 0] - extent[0]) / (extent[1] - extent[0]) * nx).astype(int)
    j = ((xyz[:, 1] - extent[2]) / (extent[3] - extent[2]) * ny).astype(int)
    k = ((xyz[:, 2] - extent[4]) / (extent[5] - extent[4]) * nz).astype(int)

    inside_extent = ((i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz))
    if not inside_extent.any():
        return None, None

    selected = np.zeros(len(blocks), bool)
    selected[inside_extent] = grid[i[inside_extent], j[inside_extent], k[inside_extent]]

    values = grade.to_numpy(float)[selected]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None

    mean_grade = float(values.mean())
    return mean_grade, float(tonnes * mean_grade / 31.1034768)


def _compare(gempy_volume: float, wireframe_volume: float) -> dict:
    """Compare the implicit and explicit orebody volumes."""
    if wireframe_volume <= 0:
        return {}
    difference = gempy_volume - wireframe_volume
    return {
        "gempy_volume_m3": round(gempy_volume, 0),
        "wireframe_volume_m3": round(wireframe_volume, 0),
        "difference_m3": round(difference, 0),
        "difference_pct": round(difference / wireframe_volume * 100.0, 1),
        "ratio": round(gempy_volume / wireframe_volume, 3),
    }


def _identify_ore_unit(ids: np.ndarray, extent: list[float], resolution: list[int],
                       ore_sample_xyz: np.ndarray | None) -> tuple[int | None, float | None]:
    """Decide which side of the interface is ore, and how well it captures the data.

    The capture fraction is the honest quality measure for an implicit model: the
    share of logged ore intercepts that fall inside the modelled solid. A surface
    can look perfectly smooth and still miss the ore, and this number exposes it.
    """
    unique = np.unique(ids)
    if unique.size == 0:
        return None, None
    if ore_sample_xyz is None or len(ore_sample_xyz) == 0:
        return int(unique[-1]), None

    nx, ny, nz = resolution
    xyz = np.asarray(ore_sample_xyz, float).reshape(-1, 3)
    i = ((xyz[:, 0] - extent[0]) / (extent[1] - extent[0]) * nx).astype(int)
    j = ((xyz[:, 1] - extent[2]) / (extent[3] - extent[2]) * ny).astype(int)
    k = ((xyz[:, 2] - extent[4]) / (extent[5] - extent[4]) * nz).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
    if not inside.any():
        return int(unique[-1]), None

    grid = ids.reshape(nx, ny, nz)
    sampled = grid[i[inside], j[inside], k[inside]]

    best_id, best_hits = None, -1
    for candidate in unique:
        hits = int((sampled == candidate).sum())
        if hits > best_hits:
            best_id, best_hits = int(candidate), hits
    return best_id, float(best_hits / int(inside.sum()))
