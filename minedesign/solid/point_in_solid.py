"""Point-in-solid classification by ray casting.

A ray is cast in +X from every test point and the crossings of the triangulated
surface are counted: odd means inside. Triangles are bucketed on a 2D (y, z)
grid so each point only tests the triangles whose projection can possibly be
hit, which keeps a full block model well inside interactive time.
"""
from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from .triangulate import Solid

log = get_logger(__name__)


class _YZIndex:
    """Bucket triangles by their (y, z) bounding box."""

    def __init__(self, tris: np.ndarray, n_cells: int = 48):
        lo = tris.min(axis=1)[:, 1:3]
        hi = tris.max(axis=1)[:, 1:3]
        self.origin = lo.min(axis=0)
        extent = np.maximum(hi.max(axis=0) - self.origin, 1e-9)
        self.cell = extent / n_cells
        self.n_cells = n_cells

        self.buckets: dict[tuple[int, int], list[int]] = {}
        c0 = np.clip(np.floor((lo - self.origin) / self.cell).astype(int), 0, n_cells - 1)
        c1 = np.clip(np.floor((hi - self.origin) / self.cell).astype(int), 0, n_cells - 1)
        for i in range(len(tris)):
            for a in range(c0[i, 0], c1[i, 0] + 1):
                for b in range(c0[i, 1], c1[i, 1] + 1):
                    self.buckets.setdefault((a, b), []).append(i)

    def query(self, yz: np.ndarray) -> list[int]:
        cell = np.clip(np.floor((yz - self.origin) / self.cell).astype(int),
                       0, self.n_cells - 1)
        return self.buckets.get((int(cell[0]), int(cell[1])), [])


def _ray_parity(point: np.ndarray, direction: np.ndarray, tris: np.ndarray,
                eps: float = 1e-9) -> tuple[int, bool]:
    """Crossings of one ray against every triangle.

    Returns ``(count, degenerate)``. ``degenerate`` is True when a hit landed on a
    triangle edge or vertex, where the crossing may be counted twice or not at
    all and the parity cannot be trusted.
    """
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = v1 - v0, v2 - v0

    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    live = np.abs(det) > eps
    if not live.any():
        return 0, False

    inv_det = 1.0 / det[live]
    tvec = point - v0[live]
    u = np.einsum("ij,ij->i", tvec, pvec[live]) * inv_det
    qvec = np.cross(tvec, e1[live])
    v = (qvec @ direction) * inv_det
    t = np.einsum("ij,ij->i", e2[live], qvec) * inv_det

    hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > eps)

    # a hit within tol of an edge or vertex is where parity breaks down
    tol = 1e-7
    near_edge = ((np.abs(u) < tol) | (np.abs(v) < tol) | (np.abs(u + v - 1.0) < tol)
                 | (np.abs(t) < tol))
    degenerate = bool((hit & near_edge).any())
    return int(hit.sum()), degenerate


def point_in_solid(points: np.ndarray, solid: Solid, eps: float = 1e-9) -> np.ndarray:
    """Boolean mask: is each point inside the closed solid?

    The fast path casts a +X ray and uses the (y, z) bucket index. When that ray
    lands exactly on a shared edge or vertex -- which happens constantly on a
    lofted solid, where a point on the axis hits the cap fan's apex -- the parity
    is unreliable, so the point is re-tested with three tilted rays against the
    full triangle set and decided by majority.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    inside = np.zeros(len(points), dtype=bool)
    if len(points) == 0:
        return inside

    tris = solid.vertices[solid.faces]
    bbox = solid.bbox
    # only points inside the bounding box can be inside the solid
    candidate = np.all((points >= bbox[0] - eps) & (points <= bbox[1] + eps), axis=1)
    if not candidate.any():
        return inside

    index = _YZIndex(tris)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = v1 - v0, v2 - v0

    direction = np.array([1.0, 0.0, 0.0])
    # Moller-Trumbore terms that do not depend on the ray origin
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    parallel = np.abs(det) < eps
    inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))

    # tilted, mutually independent directions for the robust fallback
    fallback_dirs = [
        np.array([1.0, 0.0733, 0.0331]),
        np.array([0.0417, 1.0, 0.0619]),
        np.array([0.0281, 0.0537, 1.0]),
    ]
    fallback_dirs = [d / np.linalg.norm(d) for d in fallback_dirs]

    n_fallback = 0
    tol = 1e-7

    for pi in np.flatnonzero(candidate):
        p = points[pi]
        tri_ids = index.query(p[1:3])
        degenerate = False
        crossings = 0

        if tri_ids:
            t_ids = np.asarray(tri_ids)
            t_ids = t_ids[~parallel[t_ids]]
            if t_ids.size:
                tvec = p - v0[t_ids]
                u = np.einsum("ij,ij->i", tvec, pvec[t_ids]) * inv_det[t_ids]
                qvec = np.cross(tvec, e1[t_ids])
                vv = qvec[:, 0] * inv_det[t_ids]  # direction is +X
                t = np.einsum("ij,ij->i", e2[t_ids], qvec) * inv_det[t_ids]

                hit = (u >= 0.0) & (vv >= 0.0) & (u + vv <= 1.0) & (t > eps)
                near_edge = ((np.abs(u) < tol) | (np.abs(vv) < tol)
                             | (np.abs(u + vv - 1.0) < tol) | (np.abs(t) < tol))
                crossings = int(hit.sum())
                degenerate = bool((hit & near_edge).any())

        if degenerate:
            n_fallback += 1
            votes = 0
            for d in fallback_dirs:
                count, bad = _ray_parity(p, d, tris, eps)
                if not bad:
                    votes += count % 2
                else:  # even the tilted ray was degenerate: abstain
                    votes += 0
            inside[pi] = votes >= 2
        else:
            inside[pi] = bool(crossings % 2 == 1)

    if n_fallback:
        log.info("point-in-solid: %d point(s) needed the robust multi-ray test", n_fallback)
    log.info("point-in-solid: %d of %d points inside", int(inside.sum()), len(points))
    return inside


def distance_to_surface(points: np.ndarray, solid: Solid) -> np.ndarray:
    """Approximate distance from each point to the nearest mesh vertex.

    Used for reporting only (how far a block sits from the wireframe), never for
    the ore/waste decision itself.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    from scipy.spatial import cKDTree

    tree = cKDTree(solid.vertices)
    return tree.query(points)[0]
