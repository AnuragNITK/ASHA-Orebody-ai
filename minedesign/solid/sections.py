"""Drill sections: orientation, assignment, and a deterministic boundary proposal.

Cross-sections are cut perpendicular to the strike of the drill grid, which is
recovered from the collar layout by principal component analysis. The section
frame is:
    s  along strike        (section spacing axis, one section per bin)
    u  across strike       (horizontal in-section axis)
    v  elevation           (vertical in-section axis, = z)

`propose_section_boundaries` is the deterministic fallback used when the LLM is
unavailable, and also the reference envelope the LLM proposal is checked against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class SectionFrame:
    """Orthonormal frame of the drill grid, plus section spacing."""

    origin: np.ndarray      # (3,) centroid of collars
    strike_dir: np.ndarray  # (3,) unit, horizontal, along strike (the s axis)
    across_dir: np.ndarray  # (3,) unit, horizontal, across strike (the u axis)
    spacing: float
    strike_azimuth_deg: float

    def to_local(self, xyz: np.ndarray) -> np.ndarray:
        """World xyz -> (s, u, v); all three are relative to the frame origin."""
        d = np.asarray(xyz, float).reshape(-1, 3) - self.origin
        return np.column_stack([d @ self.strike_dir, d @ self.across_dir, d[:, 2]])

    def to_world(self, suv: np.ndarray) -> np.ndarray:
        """(s, u, v) -> world xyz."""
        suv = np.asarray(suv, float).reshape(-1, 3)
        pts = (self.origin
               + suv[:, 0:1] * self.strike_dir
               + suv[:, 1:2] * self.across_dir)
        pts[:, 2] = self.origin[2] + suv[:, 2]
        return pts

    def as_dict(self) -> dict:
        return {
            "origin": [float(v) for v in self.origin],
            "strike_azimuth_deg": float(self.strike_azimuth_deg),
            "section_spacing": float(self.spacing),
        }


def _nearest_neighbour_spacing(points: np.ndarray) -> float:
    """Median nearest-neighbour distance: the working definition of drill spacing."""
    if len(points) < 2:
        return 25.0
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def build_section_frame(collar: pd.DataFrame, n_sections: int | None = None) -> SectionFrame:
    """Recover strike from the collar layout and choose a section spacing."""
    xy = collar[["x", "y"]].to_numpy(float)
    origin = np.array([xy[:, 0].mean(), xy[:, 1].mean(), collar["z"].to_numpy(float).mean()])

    centred = xy - xy.mean(axis=0)
    if len(xy) >= 3:
        cov = np.cov(centred.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, int(np.argmax(eigvals))]
    else:
        principal = np.array([1.0, 0.0])

    strike = np.array([principal[0], principal[1], 0.0])
    strike /= np.linalg.norm(strike)
    across = np.array([-strike[1], strike[0], 0.0])

    azimuth = float((np.degrees(np.arctan2(strike[0], strike[1]))) % 360.0)

    s_coord = centred @ principal
    span = float(s_coord.max() - s_coord.min())
    nn = _nearest_neighbour_spacing(xy)
    if n_sections is None:
        spacing = max(nn, span / 12.0) if span > 0 else max(nn, 25.0)
    else:
        spacing = max(span / max(n_sections, 1), 1e-6)

    log.info("section frame: strike azimuth %.1f deg, spacing %.1f m, drill spacing ~%.1f m",
             azimuth, spacing, nn)
    return SectionFrame(origin=origin, strike_dir=strike, across_dir=across,
                        spacing=spacing, strike_azimuth_deg=azimuth)


def assign_sections(intervals: pd.DataFrame, frame: SectionFrame) -> pd.DataFrame:
    """Add s/u/v local coordinates and an integer section_id to an interval table."""
    out = intervals.copy()
    local = frame.to_local(out[["x", "y", "z"]].to_numpy(float))
    out["s"], out["u"], out["v"] = local[:, 0], local[:, 1], local[:, 2]
    out["section_id"] = np.floor(out["s"] / frame.spacing).astype(int)
    out["section_s"] = (out["section_id"] + 0.5) * frame.spacing
    return out


def drill_spacing(collar: pd.DataFrame) -> float:
    return _nearest_neighbour_spacing(collar[["x", "y"]].to_numpy(float))


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull, returned counter-clockwise."""
    pts = np.unique(np.round(points, 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(p):
        stack: list[np.ndarray] = []
        for q in p:
            while len(stack) >= 2:
                a, b = stack[-2], stack[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    stack.pop()
                else:
                    break
            stack.append(q)
        return stack

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def propose_section_boundaries(ore_intervals: pd.DataFrame, frame: SectionFrame,
                               dilation: float | None = None,
                               min_points: int = 3) -> list[dict]:
    """Deterministic ore-envelope proposal, one polygon per section.

    The envelope is the convex hull of the ore-flagged sample points in the
    section plane, dilated outwards by half the local drill spacing so the solid
    extends the conventional half-spacing beyond the outermost ore intercept.
    """
    if ore_intervals.empty:
        return []

    if dilation is None:
        dilation = max(frame.spacing * 0.25, 5.0)

    boundaries: list[dict] = []
    for section_id, grp in ore_intervals.groupby("section_id", sort=True):
        pts = grp[["u", "v"]].to_numpy(float)
        if len(pts) < min_points:
            if len(pts) == 0:
                continue
            # a single intercept still defines ore: use a drill-spacing-sized box
            c = pts.mean(axis=0)
            r = dilation
            hull = np.array([[c[0] - r, c[1] - r], [c[0] + r, c[1] - r],
                             [c[0] + r, c[1] + r], [c[0] - r, c[1] + r]])
        else:
            hull = _convex_hull_2d(pts)
            if len(hull) < 3:
                c = pts.mean(axis=0)
                r = max(dilation, float(np.abs(pts - c).max()))
                hull = np.array([[c[0] - r, c[1] - r], [c[0] + r, c[1] - r],
                                 [c[0] + r, c[1] + r], [c[0] - r, c[1] + r]])
            else:
                centroid = hull.mean(axis=0)
                radial = hull - centroid
                norms = np.linalg.norm(radial, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                hull = hull + radial / norms * dilation

        s = float(grp["section_s"].iloc[0])
        world = frame.to_world(np.column_stack([np.full(len(hull), s), hull[:, 0], hull[:, 1]]))
        boundaries.append({
            "section_id": int(section_id),
            "section_s": s,
            "n_ore_points": int(len(grp)),
            "boundary_points": [[float(p[0]), float(p[1]), float(p[2])] for p in world],
        })

    log.info("proposed %d section boundary polygon(s) from ore intercepts", len(boundaries))
    return boundaries


def section_summary(intervals: pd.DataFrame, frame: SectionFrame,
                    max_points_per_section: int = 60) -> list[dict]:
    """Compact per-section payload for LLM Call 1 (points, not whole tables)."""
    summaries = []
    for section_id, grp in intervals.groupby("section_id", sort=True):
        g = grp
        if len(g) > max_points_per_section:  # keep the highest-grade evidence
            g = g.nlargest(max_points_per_section, "grade")
        summaries.append({
            "section_id": int(section_id),
            "section_s": float(grp["section_s"].iloc[0]),
            "n_intervals": int(len(grp)),
            "grade_mean": float(np.nanmean(grp["grade"])),
            "grade_max": float(np.nanmax(grp["grade"])),
            "rock_types": sorted({str(r) for r in grp.get("rock_type", pd.Series(dtype=str))}),
            "points": [
                {"u": round(float(r.u), 1), "v": round(float(r.v), 1),
                 "grade": round(float(r.grade), 4) if np.isfinite(r.grade) else None,
                 "rock_type": str(getattr(r, "rock_type", "UNKNOWN"))}
                for r in g.itertuples(index=False)
            ],
        })
    return summaries
