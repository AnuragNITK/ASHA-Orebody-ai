"""Generate a synthetic but geologically plausible drilling campaign.

Creates the four Surpac-style source files for a dipping tabular gold deposit:
a mineralised lens inside an altered halo inside barren host rock, drilled on a
regular grid with angled holes. Grades are lognormal with a spatially smooth
trend, which is what makes the downstream variography meaningful rather than
noise.

Usage:  python scripts/make_sample_data.py [--out data/sample] [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --- deposit geometry (local mine grid, metres) ----------------------------
ORIGIN_X, ORIGIN_Y, ORIGIN_Z = 10_000.0, 20_000.0, 1_250.0
STRIKE_AZIMUTH = 45.0     # lens strikes NE
LENS_DIP = 65.0           # dips 65 degrees to the SE
LENS_LENGTH = 460.0       # along strike
LENS_THICKNESS = 70.0     # true thickness
LENS_DEPTH_TOP = 30.0     # below surface
LENS_DEPTH_BOTTOM = 300.0


def _rotation(azimuth_deg: float) -> np.ndarray:
    a = np.radians(azimuth_deg)
    return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def _to_lens_frame(xyz: np.ndarray) -> np.ndarray:
    """World -> lens frame: (along strike, across strike, depth below collar plane)."""
    rel = xyz - np.array([ORIGIN_X, ORIGIN_Y, ORIGIN_Z])
    rot = _rotation(STRIKE_AZIMUTH)
    horiz = rel[:, :2] @ rot.T
    depth = -rel[:, 2]
    dip = np.radians(LENS_DIP)
    # shear the across-strike axis so the lens plane becomes vertical in this frame
    across = horiz[:, 1] - depth / np.tan(dip)
    return np.column_stack([horiz[:, 0], across, depth])


def _mineralisation(xyz: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (grade g/t, rock code) for sample midpoints."""
    lf = _to_lens_frame(xyz)
    along, across, depth = lf[:, 0], lf[:, 1], lf[:, 2]

    # normalised distance from the lens core, 0 at the centre
    r_along = along / (LENS_LENGTH / 2.0)
    r_across = across / (LENS_THICKNESS / 2.0)
    mid_depth = (LENS_DEPTH_TOP + LENS_DEPTH_BOTTOM) / 2.0
    r_depth = (depth - mid_depth) / ((LENS_DEPTH_BOTTOM - LENS_DEPTH_TOP) / 2.0)
    r = np.sqrt(r_along ** 2 + r_across ** 2 + (0.8 * r_depth) ** 2)

    # smooth grade trend: high-grade core decaying outwards
    trend = 7.5 * np.exp(-1.35 * r ** 2)

    # correlated noise via a smooth random field, plus a nugget
    field = np.zeros_like(trend)
    for wavelength, amp in ((160.0, 0.55), (70.0, 0.35), (30.0, 0.2)):
        phase = rng.uniform(0, 2 * np.pi, 3)
        field += amp * (np.sin(2 * np.pi * along / wavelength + phase[0])
                        * np.cos(2 * np.pi * across / (wavelength * 0.4) + phase[1])
                        * np.sin(2 * np.pi * depth / wavelength + phase[2]))
    multiplier = np.exp(0.55 * field + rng.normal(0, 0.35, size=len(xyz)))

    grade = trend * multiplier + rng.lognormal(-3.2, 0.9, size=len(xyz))

    # a handful of nuggety outliers, so the top cut has something to bite on
    spikes = rng.random(len(xyz)) < 0.004
    grade[spikes] *= rng.uniform(3, 8, size=int(spikes.sum()))

    rock = np.full(len(xyz), "HOST", dtype=object)
    rock[r < 1.6] = "ALTERED"
    rock[r < 1.0] = "MINZONE"
    rock[depth < 12.0] = "OXIDE"
    return np.round(grade, 3), rock


def build(seed: int = 42, hole_spacing: float = 50.0,
          section_spacing: float = 50.0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rot = _rotation(STRIKE_AZIMUTH)

    collars, surveys, assays, geology = [], [], [], []
    hole_no = 0

    along_positions = np.arange(-LENS_LENGTH / 2 - 40, LENS_LENGTH / 2 + 41, section_spacing)
    across_positions = np.arange(-100, 151, hole_spacing)

    for along in along_positions:
        for across in across_positions:
            hole_no += 1
            hole_id = f"DDH{hole_no:03d}"

            jitter = rng.normal(0, 3.0, 2)
            local = np.array([along + jitter[0], across + jitter[1]])
            world_xy = local @ rot + np.array([ORIGIN_X, ORIGIN_Y])
            # gently undulating topography
            z = (ORIGIN_Z + 8.0 * np.sin(along / 130.0) + 5.0 * np.cos(across / 90.0)
                 + rng.normal(0, 1.0))

            # holes are drilled to intersect the lens perpendicular to dip
            azimuth = (STRIKE_AZIMUTH + 90.0 + rng.normal(0, 4.0)) % 360.0
            dip = -(90.0 - LENS_DIP) - 20.0 + rng.normal(0, 3.0)
            depth_needed = 120.0 + abs(across) * 1.4 + rng.uniform(0, 60)
            max_depth = float(np.round(min(depth_needed, 420.0), 1))

            collars.append({"HoleID": hole_id, "East": round(world_xy[0], 2),
                            "North": round(world_xy[1], 2), "RL": round(z, 2),
                            "MaxDepth": max_depth})

            # survey stations every 30 m with realistic downhole deviation
            stations = np.arange(0.0, max_depth + 1e-6, 30.0)
            if stations[-1] < max_depth - 1e-6:
                stations = np.append(stations, max_depth)
            az_drift = azimuth
            dip_drift = dip
            for k, station in enumerate(stations):
                if k > 0:
                    az_drift += rng.normal(0, 0.8)
                    dip_drift += rng.normal(0.35, 0.4)  # holes flatten with depth
                surveys.append({"HoleID": hole_id, "Depth": round(float(station), 2),
                                "Azimuth": round(az_drift % 360.0, 2),
                                "Dip": round(float(np.clip(dip_drift, -90, 0)), 2)})

            # assay intervals: mostly 1 m, some 1.5 m and 0.5 m runs
            edges = [0.0]
            while edges[-1] < max_depth - 1e-6:
                length = rng.choice([1.0, 1.0, 1.0, 1.0, 1.5, 0.5], p=[.55, .1, .1, .1, .1, .05])
                edges.append(min(edges[-1] + float(length), max_depth))
            edges = np.array(edges)

            f, t = edges[:-1], edges[1:]
            mid = (f + t) / 2.0

            # desurvey the midpoints with a simple tangential step for sampling
            az_r = np.radians(azimuth)
            dip_r = np.radians(dip + 0.35 * mid / 30.0)
            dx = np.cos(dip_r) * np.sin(az_r) * mid
            dy = np.cos(dip_r) * np.cos(az_r) * mid
            dz = np.sin(dip_r) * mid
            xyz = np.column_stack([world_xy[0] + dx, world_xy[1] + dy, z + dz])

            grade, rock = _mineralisation(xyz, rng)

            for i in range(len(f)):
                assays.append({"HoleID": hole_id, "From": round(float(f[i]), 2),
                               "To": round(float(t[i]), 2), "Au_gpt": float(grade[i])})

            # collapse consecutive identical rock codes into geology intervals
            start = 0
            for i in range(1, len(rock) + 1):
                if i == len(rock) or rock[i] != rock[start]:
                    geology.append({"HoleID": hole_id, "From": round(float(f[start]), 2),
                                    "To": round(float(t[i - 1]), 2),
                                    "Lithology": str(rock[start])})
                    start = i

    return {
        "collar": pd.DataFrame(collars),
        "survey": pd.DataFrame(surveys),
        "assay": pd.DataFrame(assays),
        "geology": pd.DataFrame(geology),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hole-spacing", type=float, default=50.0)
    ap.add_argument("--section-spacing", type=float, default=50.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = build(args.seed, args.hole_spacing, args.section_spacing)
    for name, df in frames.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{path}  {len(df):>7,} rows")

    assay = frames["assay"]
    print(f"\nholes            {frames['collar'].shape[0]}")
    print(f"grade mean/max   {assay['Au_gpt'].mean():.3f} / {assay['Au_gpt'].max():.3f} g/t")
    print(f"rock codes       {sorted(frames['geology']['Lithology'].unique())}")


if __name__ == "__main__":
    main()
