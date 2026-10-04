"""Derive the orebody contact surface and its orientations from drillholes.

GemPy models a body implicitly: it interpolates a scalar potential field through
observed *interface points* and constrains the field's gradient with
*orientations*. Both come out of the drilling, and neither is a judgement call --
a contact is where a hole crosses between waste and ore, and an orientation is
the plane that best fits the contacts around it.

The orebody is modelled as ONE closed surface, not as a hanging wall and a
footwall separately. That choice is the whole thing:

    Two conformable surfaces in one GemPy series are isosurfaces of the same
    scalar field, so they are forced roughly parallel. An orebody is not two
    parallel sheets -- it is a closed body whose top is a dome and whose base is
    a bowl. Modelled as a pair, the field has to compromise between contradictory
    orientations and the two surfaces collapse onto each other, leaving a sliver
    that contains almost none of the ore.

So every contact, top and bottom alike, belongs to a single closed surface, and
each orientation is flipped to point *out* of the ore. Whether a contact is a
hanging-wall or footwall crossing is still recorded, because it is useful to
report, but it does not split the surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

ORE_SURFACE = "ore_contact"
TOP = "hanging_wall"    # a hole passes from waste into ore
BASE = "footwall"       # a hole passes from ore back into waste


@dataclass
class ContactSet:
    """Interface points and outward orientations, ready for GemPy."""

    points: pd.DataFrame                      # x, y, z, crossing, hole_id
    orientations: pd.DataFrame                # x, y, z, G_x, G_y, G_z, dip, azimuth
    warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict:
        crossing = self.points["crossing"] if "crossing" in self.points.columns else pd.Series(dtype=str)
        return {
            "n_contact_points": int(len(self.points)),
            "n_hanging_wall_contacts": int((crossing == TOP).sum()),
            "n_footwall_contacts": int((crossing == BASE).sum()),
            "n_orientations": int(len(self.orientations)),
            "n_holes_with_contacts": int(self.points["hole_id"].nunique())
            if not self.points.empty else 0,
        }

    @property
    def is_usable(self) -> bool:
        """GemPy needs points to define the surface and at least one orientation."""
        return len(self.points) >= 4 and len(self.orientations) >= 1


def extract_contacts(composites: pd.DataFrame, ore_mask: np.ndarray) -> pd.DataFrame:
    """Find every waste-to-ore and ore-to-waste crossing down each hole.

    The contact is placed midway between the last waste sample and the first ore
    sample, the conventional treatment when the true contact is only bracketed
    by the sample spacing.
    """
    df = composites.copy()
    df["is_ore"] = np.asarray(ore_mask, bool)
    df = df.dropna(subset=["x", "y", "z"]).sort_values(["hole_id", "from"])

    records: list[dict] = []
    for hole_id, grp in df.groupby("hole_id", sort=False):
        ore = grp["is_ore"].to_numpy()
        if not ore.any():
            continue
        xyz = grp[["x", "y", "z"]].to_numpy(float)

        for i in range(len(grp)):
            previous = ore[i - 1] if i > 0 else False
            if ore[i] and not previous:
                point = xyz[i] if i == 0 else (xyz[i] + xyz[i - 1]) / 2.0
                records.append({"x": point[0], "y": point[1], "z": point[2],
                                "crossing": TOP, "hole_id": hole_id})
            if ore[i] and (i == len(grp) - 1 or not ore[i + 1]):
                point = xyz[i] if i == len(grp) - 1 else (xyz[i] + xyz[i + 1]) / 2.0
                records.append({"x": point[0], "y": point[1], "z": point[2],
                                "crossing": BASE, "hole_id": hole_id})

    out = pd.DataFrame.from_records(records)
    if out.empty:
        log.warning("no ore contacts found in the drilling")
        return out

    log.info("extracted %d contacts (%d hanging wall, %d footwall) from %d holes",
             len(out), int((out["crossing"] == TOP).sum()),
             int((out["crossing"] == BASE).sum()), out["hole_id"].nunique())
    return out


def _fit_plane_normal(points: np.ndarray) -> np.ndarray | None:
    """Least-squares plane through a point cloud; returns the unit normal."""
    if len(points) < 3:
        return None
    centred = points - points.mean(axis=0)
    try:
        _, singular, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular[-1] > singular[0] * 0.95:   # an isotropic blob defines no plane
        return None
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    return None if norm < 1e-9 else normal / norm


def normal_to_dip_azimuth(normal: np.ndarray) -> tuple[float, float]:
    """Convert a unit normal into geological dip and dip azimuth, in degrees.

    Dip azimuth is the compass bearing of steepest descent in the plane. For a
    plane written as z = (c - nx*x - ny*y) / nz with an upward normal, the
    downhill gradient is (nx, ny) -- so the bearing is the horizontal part of the
    normal itself, not its reverse. The normal is flipped upward first so a
    contact on the underside of a body still reports a dip in 0-90 degrees.
    """
    normal = np.asarray(normal, float)
    upward = normal if normal[2] >= 0 else -normal
    nx, ny, nz = upward
    dip = float(np.degrees(np.arccos(np.clip(nz, -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(nx, ny)) % 360.0)
    return dip, azimuth


def estimate_orientations(contacts: pd.DataFrame, ore_points: np.ndarray,
                          n_orientations: int = 12,
                          neighbours: int = 14) -> pd.DataFrame:
    """Outward normals on the contact surface, fitted to the contacts themselves.

    Each orientation is a plane fitted to the nearest contact points, then
    flipped to point away from the local centre of the ore. On a closed body the
    normals must reverse between the top and the bottom, and anchoring the sign
    to the ore mass is what makes that happen automatically.
    """
    empty = pd.DataFrame(columns=["x", "y", "z", "G_x", "G_y", "G_z",
                                  "dip", "azimuth", "kind"])
    if contacts.empty:
        return empty

    from scipy.spatial import cKDTree

    pts = contacts[["x", "y", "z"]].to_numpy(float)
    if len(pts) < 3:
        return empty

    contact_tree = cKDTree(pts)
    ore_points = np.asarray(ore_points, float).reshape(-1, 3)
    ore_tree = cKDTree(ore_points) if len(ore_points) else None
    global_centre = ore_points.mean(axis=0) if len(ore_points) else pts.mean(axis=0)

    picks = np.unique(np.linspace(0, len(pts) - 1, min(n_orientations, len(pts))).astype(int))
    records: list[dict] = []
    for index in picks:
        p = pts[index]
        _, idx = contact_tree.query(p, k=min(neighbours, len(pts)))
        normal = _fit_plane_normal(pts[np.asarray(idx)])
        if normal is None:
            continue

        # point the normal away from the ore: locally if we can, globally if not
        if ore_tree is not None and len(ore_points) >= 8:
            _, near_ore = ore_tree.query(p, k=min(30, len(ore_points)))
            centre = ore_points[np.asarray(near_ore)].mean(axis=0)
        else:
            centre = global_centre
        outward = p - centre
        if np.dot(normal, outward) < 0:
            normal = -normal

        dip, azimuth = normal_to_dip_azimuth(normal)
        records.append({"x": p[0], "y": p[1], "z": p[2],
                        "G_x": normal[0], "G_y": normal[1], "G_z": normal[2],
                        "dip": dip, "azimuth": azimuth,
                        "kind": f"plane through {neighbours} nearest contacts, "
                                f"oriented away from the ore"})

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        upward = int((out["G_z"] > 0).sum())
        log.info("fitted %d outward orientations (%d point up, %d down), mean dip %.1f",
                 len(out), upward, len(out) - upward, out["dip"].mean())
    return out if not out.empty else empty


def build_contact_set(composites: pd.DataFrame, ore_mask: np.ndarray,
                      min_points: int = 8, thin: int | None = None,
                      n_orientations: int = 12) -> ContactSet:
    """Extract the contact surface and its outward orientations."""
    warnings: list[str] = []
    points = extract_contacts(composites, ore_mask)

    if points.empty:
        return ContactSet(points, pd.DataFrame(), ["no ore intercepts were found"])

    # A well-spread, moderate point set interpolates more stably than thousands
    # of near-coincident contacts, and solves far faster.
    if thin and len(points) > thin:
        keep = np.linspace(0, len(points) - 1, thin).astype(int)
        points = points.iloc[keep].reset_index(drop=True)
        warnings.append(f"contact points were thinned to {thin} for interpolation stability")

    if len(points) < min_points:
        warnings.append(f"only {len(points)} ore contacts were found, below the "
                        f"{min_points} that make an interpolated surface meaningful")

    ore_points = composites.loc[np.asarray(ore_mask, bool), ["x", "y", "z"]].dropna()
    orientations = estimate_orientations(points, ore_points.to_numpy(float),
                                         n_orientations=n_orientations)
    if orientations.empty:
        warnings.append("no orientation could be fitted to the contacts")

    contact_set = ContactSet(points=points, orientations=orientations, warnings=warnings)
    if not contact_set.is_usable:
        warnings.append("the contact set does not meet GemPy's minimum of four points "
                        "and one orientation")
    return contact_set
