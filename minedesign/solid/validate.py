"""Solid validation: the Surpac wireframe validation checks, in numpy.

Reported checks, matching the standard validation panel:
    closed              every edge is shared by exactly two faces
    connected           the surface is a single connected component
    duplicate           duplicate vertices or duplicate faces
    invalid edges       an edge used by three or more faces (non-manifold)
    intersecting        self-intersecting triangles
    reversed            inconsistent face winding, i.e. flipped normals
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..logging_utils import get_logger
from .triangulate import Solid

log = get_logger(__name__)


@dataclass
class ValidationReport:
    checks: dict[str, dict] = field(default_factory=dict)
    volume: float = 0.0
    surface_area: float = 0.0
    n_vertices: int = 0
    n_faces: int = 0

    @property
    def is_valid(self) -> bool:
        return all(c["passed"] for c in self.checks.values())

    @property
    def failures(self) -> list[str]:
        return [f"{k}: {v['detail']}" for k, v in self.checks.items() if not v["passed"]]

    def as_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "checks": self.checks,
            "volume_m3": self.volume,
            "surface_area_m2": self.surface_area,
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "failures": self.failures,
        }


def _edge_table(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(edges, axis=1)
    unique, counts = np.unique(undirected, axis=0, return_counts=True)
    return unique, counts


def _connected_components(n_vertices: int, faces: np.ndarray) -> int:
    """Union-find over face-shared vertices."""
    parent = np.arange(n_vertices)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in faces:
        r0 = find(int(tri[0]))
        for v in tri[1:]:
            r = find(int(v))
            if r != r0:
                parent[r] = r0

    used = np.unique(faces)
    return len({find(int(v)) for v in used})


def _triangles_intersect(t1: np.ndarray, t2: np.ndarray, eps: float = 1e-9) -> bool:
    """Separating-axis test for two triangles in 3D.

    The 11 classic axes (two face normals plus nine edge-edge cross products) are
    not sufficient on their own: when the triangles are coplanar every one of
    those axes is parallel to the shared normal, so two disjoint coplanar
    triangles are wrongly reported as touching. The in-plane perpendiculars
    normal-cross-edge close that gap, which matters here because a lofted solid
    is full of coplanar strips.
    """
    e1 = [t1[1] - t1[0], t1[2] - t1[1], t1[0] - t1[2]]
    e2 = [t2[1] - t2[0], t2[2] - t2[1], t2[0] - t2[2]]

    n1 = np.cross(e1[0], e1[1])
    n2 = np.cross(e2[0], e2[1])
    if np.linalg.norm(n1) < eps or np.linalg.norm(n2) < eps:
        return False  # degenerate triangle: reported by the duplicates check instead

    axes = [n1, n2]
    for a in e1:
        for b in e2:
            axes.append(np.cross(a, b))
    for e in e1:
        axes.append(np.cross(n1, e))
    for e in e2:
        axes.append(np.cross(n2, e))

    for axis in axes:
        if np.linalg.norm(axis) < eps:
            continue
        p1 = t1 @ axis
        p2 = t2 @ axis
        if p1.min() > p2.max() + eps or p2.min() > p1.max() + eps:
            return False
    return True


def _self_intersection_count(solid: Solid, max_pairs: int = 400_000) -> int:
    """Count self-intersecting triangle pairs, bounded by an AABB pre-filter."""
    v, f = solid.vertices, solid.faces
    tris = v[f]
    lo, hi = tris.min(axis=1), tris.max(axis=1)

    # bucket triangles on a coarse grid so only nearby pairs are tested
    extent = solid.bbox
    size = np.maximum(extent[1] - extent[0], 1e-9)
    n_cells = max(2, int(round(len(f) ** (1 / 3))))
    cell = size / n_cells

    buckets: dict[tuple, list[int]] = {}
    for i in range(len(f)):
        c0 = np.floor((lo[i] - extent[0]) / cell).astype(int)
        c1 = np.floor((hi[i] - extent[0]) / cell).astype(int)
        for a in range(c0[0], c1[0] + 1):
            for b in range(c0[1], c1[1] + 1):
                for c in range(c0[2], c1[2] + 1):
                    buckets.setdefault((a, b, c), []).append(i)

    tested: set[tuple[int, int]] = set()
    hits = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        for ii in range(len(members)):
            for jj in range(ii + 1, len(members)):
                i, j = members[ii], members[jj]
                key = (i, j)
                if key in tested:
                    continue
                tested.add(key)
                if len(tested) > max_pairs:
                    log.warning("self-intersection check truncated at %d pairs", max_pairs)
                    return hits
                if np.intersect1d(f[i], f[j]).size:  # adjacent faces legitimately touch
                    continue
                if lo[i][0] > hi[j][0] or lo[j][0] > hi[i][0]:
                    continue
                if lo[i][1] > hi[j][1] or lo[j][1] > hi[i][1]:
                    continue
                if lo[i][2] > hi[j][2] or lo[j][2] > hi[i][2]:
                    continue
                if _triangles_intersect(tris[i], tris[j]):
                    hits += 1
    return hits


def validate_solid(solid: Solid, check_intersections: bool = True) -> ValidationReport:
    """Run every wireframe validation check and return a structured report."""
    v, f = solid.vertices, solid.faces
    report = ValidationReport(n_vertices=int(len(v)), n_faces=int(len(f)))

    unique_edges, counts = _edge_table(f)

    n_open = int((counts == 1).sum())
    report.checks["closed"] = {
        "passed": n_open == 0,
        "count": n_open,
        "detail": "watertight, every edge shared by two faces" if n_open == 0
        else f"{n_open} boundary edge(s): the surface is open",
    }

    n_bad_edges = int((counts > 2).sum())
    report.checks["invalid_edges"] = {
        "passed": n_bad_edges == 0,
        "count": n_bad_edges,
        "detail": "no non-manifold edges" if n_bad_edges == 0
        else f"{n_bad_edges} edge(s) shared by three or more faces",
    }

    n_components = _connected_components(len(v), f)
    report.checks["connected"] = {
        "passed": n_components <= 1,
        "count": n_components,
        "detail": "single connected surface" if n_components <= 1
        else f"{n_components} disconnected components",
    }

    rounded = np.round(v, 6)
    n_dup_v = int(len(rounded) - len(np.unique(rounded, axis=0)))
    n_dup_f = int(len(f) - len(np.unique(np.sort(f, axis=1), axis=0)))
    degenerate = int((np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]],
                                              v[f[:, 2]] - v[f[:, 0]]), axis=1) < 1e-12).sum())
    n_dup = n_dup_v + n_dup_f + degenerate
    report.checks["duplicates"] = {
        "passed": n_dup == 0,
        "count": n_dup,
        "detail": "no duplicate vertices, faces or zero-area triangles" if n_dup == 0
        else (f"{n_dup_v} duplicate vertex/vertices, {n_dup_f} duplicate face(s), "
              f"{degenerate} zero-area triangle(s)"),
    }

    # reversed: on a consistently wound closed surface each shared edge is
    # traversed once in each direction
    directed = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    keys = {}
    reversed_edges = 0
    for a, b in directed:
        keys[(int(a), int(b))] = keys.get((int(a), int(b)), 0) + 1
    for (a, b), n in keys.items():
        if n > 1 or (b, a) not in keys:
            reversed_edges += 1
    report.checks["reversed"] = {
        "passed": reversed_edges == 0,
        "count": reversed_edges,
        "detail": "consistent outward winding" if reversed_edges == 0
        else f"{reversed_edges} edge(s) with inconsistent face winding",
    }

    if check_intersections:
        n_int = _self_intersection_count(solid)
    else:
        n_int = 0
    report.checks["intersecting"] = {
        "passed": n_int == 0,
        "count": n_int,
        "detail": "no self-intersecting triangles" if n_int == 0
        else f"{n_int} self-intersecting triangle pair(s)",
    }

    report.volume = solid.volume()
    report.surface_area = solid.surface_area()

    if report.is_valid:
        log.info("solid validation passed: volume %.0f m3", report.volume)
    else:
        log.warning("solid validation FAILED: %s", "; ".join(report.failures))
    return report
