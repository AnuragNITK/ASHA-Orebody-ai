"""Triangulate section boundary polygons into a closed 3D orebody solid.

The classic wireframing operation: each section carries a closed boundary
polygon, consecutive sections are lofted together with a triangle strip, and the
two end sections are capped. Vertices are resampled to a common count in polar
coordinates about each polygon centroid so corresponding vertices line up and
the loft cannot twist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..logging_utils import get_logger
from .sections import SectionFrame

log = get_logger(__name__)


@dataclass
class Solid:
    """A triangulated closed surface."""

    vertices: np.ndarray  # (V, 3) float
    faces: np.ndarray     # (F, 3) int
    section_polygons: list[dict]  # per-section resampled polygon, for the 3D scene

    @property
    def bbox(self) -> np.ndarray:
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    def volume(self) -> float:
        """Signed volume by the divergence theorem, over closed triangle soup."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0)

    def surface_area(self) -> float:
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum() / 2.0)

    def as_scene_dict(self) -> dict:
        return {"vertices": [float(v) for v in self.vertices.ravel()],
                "faces": [int(i) for i in self.faces.ravel()]}


def _order_polygon(points_2d: np.ndarray) -> np.ndarray:
    """Order a point cloud into a simple polygon by angle about its centroid."""
    centroid = points_2d.mean(axis=0)
    angles = np.arctan2(points_2d[:, 1] - centroid[1], points_2d[:, 0] - centroid[0])
    return points_2d[np.argsort(angles)]


def _resample_polar(points_2d: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to ``n`` vertices at equally spaced angles.

    Star-shaped about the centroid is assumed, which holds for the ore envelopes
    this pipeline generates and keeps the loft between sections untwisted.
    """
    ordered = _order_polygon(points_2d)
    centroid = ordered.mean(axis=0)
    rel = ordered - centroid
    angles = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    radii = np.linalg.norm(rel, axis=1)

    order = np.argsort(angles)
    angles, radii = angles[order], radii[order]
    # wrap one period so interpolation is continuous across -pi/pi
    angles = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi])
    radii = np.concatenate([radii, radii, radii])

    target = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = np.interp(target, angles, radii)
    return centroid + np.column_stack([r * np.cos(target), r * np.sin(target)])


def triangulate_solid(section_boundaries: list[dict], frame: SectionFrame,
                      n_ring: int = 24, close_ends: bool = True) -> Solid:
    """Build a closed mesh from per-section boundary polygons (world coordinates)."""
    rings: list[np.ndarray] = []
    ring_s: list[float] = []
    polygons: list[dict] = []

    usable = [b for b in section_boundaries if len(b.get("boundary_points", [])) >= 3]
    usable.sort(key=lambda b: b.get("section_s", b.get("section_id", 0)))
    if len(usable) < 2:
        raise ValueError(
            f"need at least 2 sections with 3+ boundary points to triangulate a solid, got {len(usable)}")

    for b in usable:
        pts = np.asarray(b["boundary_points"], float)
        local = frame.to_local(pts)
        s = float(np.median(local[:, 0]))
        ring2d = _resample_polar(local[:, 1:3], n_ring)
        ring3d = frame.to_world(np.column_stack([np.full(n_ring, s), ring2d]))
        rings.append(ring3d)
        ring_s.append(s)
        polygons.append({
            "section_id": int(b.get("section_id", len(polygons))),
            "section_s": s,
            "points": [[float(p[0]), float(p[1]), float(p[2])] for p in ring3d],
        })

    vertices = [np.vstack(rings)]
    faces: list[list[int]] = []

    # loft consecutive rings with a quad strip split into two triangles
    for k in range(len(rings) - 1):
        a0 = k * n_ring
        b0 = (k + 1) * n_ring
        for i in range(n_ring):
            j = (i + 1) % n_ring
            faces.append([a0 + i, b0 + i, b0 + j])
            faces.append([a0 + i, b0 + j, a0 + j])

    verts = vertices[0]
    if close_ends:
        # fan-cap each end from a centroid vertex, wound outwards
        first_centre = rings[0].mean(axis=0)
        last_centre = rings[-1].mean(axis=0)
        c_first = len(verts)
        c_last = len(verts) + 1
        verts = np.vstack([verts, first_centre, last_centre])
        last0 = (len(rings) - 1) * n_ring
        # Each cap fan must traverse its ring edge opposite to the loft, or the
        # two faces meeting on that edge disagree and the winding check fails.
        for i in range(n_ring):
            j = (i + 1) % n_ring
            faces.append([c_first, i, j])                 # start cap
            faces.append([c_last, last0 + j, last0 + i])  # end cap

    solid = Solid(vertices=verts, faces=np.asarray(faces, dtype=np.int64),
                  section_polygons=polygons)
    solid = orient_outwards(solid)
    log.info("triangulated solid: %d vertices, %d faces, volume %.0f m3",
             len(solid.vertices), len(solid.faces), solid.volume())
    return solid


def orient_outwards(solid: Solid) -> Solid:
    """Flip every face if the signed volume came out negative (inward winding)."""
    v0 = solid.vertices[solid.faces[:, 0]]
    v1 = solid.vertices[solid.faces[:, 1]]
    v2 = solid.vertices[solid.faces[:, 2]]
    signed = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
    if signed < 0:
        solid.faces = solid.faces[:, [0, 2, 1]]
    return solid


def face_normals(solid: Solid) -> np.ndarray:
    v0 = solid.vertices[solid.faces[:, 0]]
    v1 = solid.vertices[solid.faces[:, 1]]
    v2 = solid.vertices[solid.faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return n / lengths
