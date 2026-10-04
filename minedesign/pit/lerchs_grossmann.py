"""Ultimate pit limit optimization.

The ultimate pit is the maximum-weight closure of the block precedence graph.
Lerchs and Grossmann solved that with their graph algorithm in 1965; Picard
showed in 1976 that the same closure is obtained exactly from a minimum cut on
an augmented network, which is what this module does. The result is the true
optimum, identical to what the Lerchs-Grossmann algorithm returns, computed with
a fast C min-cut instead of a slow Python graph traversal.

Slope control uses the standard single-level arc template: every block points at
the blocks one bench above it inside a radius of dz / tan(slope). Composed over
benches, those arcs generate exactly the inverted cone of the required angle.

Economics and slope angles arrive from LLM Call 2; nothing here re-decides them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from ..errors import PitOptimizationError
from ..logging_utils import get_logger

log = get_logger(__name__)

GRAMS_PER_OZ = 31.1034768
_INT32_HEADROOM = 5e8  # keeps the scaled network well inside int32


@dataclass
class EconomicParameters:
    """Every economic input to the pit, each traceable to user input or a default."""

    commodity: str = "gold"
    commodity_price: float = 2000.0        # per price_unit
    price_unit: str = "USD/oz"
    mining_cost_per_tonne: float = 3.0
    processing_cost_per_tonne: float = 25.0
    ga_cost_per_tonne: float = 4.0
    recovery_pct: float = 90.0
    overall_slope_angle_deg: float = 45.0
    slope_angles_by_sector: dict = field(default_factory=dict)
    discount_rate_pct: float = 8.0
    grade_unit: str = "g/t"

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    @property
    def revenue_per_tonne_per_grade_unit(self) -> float:
        """Revenue from one tonne at a grade of 1.0, after recovery.

        For a precious metal quoted per ounce with grades in g/t this converts
        grams to ounces; for a base metal quoted per tonne with grades in percent
        it converts percent to tonnes.
        """
        recovery = self.recovery_pct / 100.0
        unit = self.price_unit.lower()
        if "oz" in unit:
            return self.commodity_price * recovery / GRAMS_PER_OZ
        if "lb" in unit:  # percent grade, price per pound
            return self.commodity_price * recovery * 22.0462
        return self.commodity_price * recovery / 100.0  # percent grade, price per tonne

    def internal_cutoff_grade(self) -> float:
        """Grade at which revenue exactly covers processing, the marginal cutoff."""
        revenue_rate = self.revenue_per_tonne_per_grade_unit
        if revenue_rate <= 0:
            return float("inf")
        return (self.processing_cost_per_tonne + self.ga_cost_per_tonne) / revenue_rate


@dataclass
class PitResult:
    in_pit: np.ndarray
    economics: EconomicParameters
    summary: dict
    block_values: np.ndarray
    pit_surface: dict | None = None
    nested_shells: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"economics": self.economics.as_dict(), "summary": self.summary,
                "nested_shells": self.nested_shells}


def compute_block_values(blocks: pd.DataFrame, econ: EconomicParameters,
                         revenue_factor: float = 1.0,
                         grade_column: str = "grade") -> np.ndarray:
    """Economic value of each block, taking the better of processing or dumping."""
    tonnes = blocks["tonnes"].to_numpy(float)
    grade = blocks[grade_column].to_numpy(float)
    grade = np.where(np.isfinite(grade), grade, 0.0)

    revenue = tonnes * grade * econ.revenue_per_tonne_per_grade_unit * revenue_factor
    processing = tonnes * (econ.processing_cost_per_tonne + econ.ga_cost_per_tonne)
    mining = tonnes * econ.mining_cost_per_tonne

    value_as_ore = revenue - processing - mining
    value_as_waste = -mining
    values = np.maximum(value_as_ore, value_as_waste)

    if "is_air" in blocks.columns:  # blocks above topography cost nothing to remove
        values = np.where(blocks["is_air"].to_numpy(bool), 0.0, values)
    return values


def _grid_indices(blocks: pd.DataFrame, block_size: tuple[float, float, float]):
    """Map block centroids onto integer grid indices."""
    bx, by, bz = block_size
    x = blocks["x"].to_numpy(float)
    y = blocks["y"].to_numpy(float)
    z = blocks["z"].to_numpy(float)
    i = np.round((x - x.min()) / bx).astype(np.int64)
    j = np.round((y - y.min()) / by).astype(np.int64)
    k = np.round((z - z.min()) / bz).astype(np.int64)
    return i, j, k


def _precedence_offsets(block_size: tuple[float, float, float],
                        slope_angle_deg: float,
                        max_benches: int = 6) -> tuple[np.ndarray, float]:
    """Arc template generating the inverted cone of the requested slope angle.

    A single-bench template can only ever produce slopes of atan(bz / (m * bx))
    for integer m, so a requested 40 degrees silently becomes 45 -- a steeper
    wall, less waste, and an optimistic pit. Spanning several benches gives a
    much finer set of achievable angles, because a horizontal run of 3 blocks
    over 2 benches is a slope no single-bench template can express.

    Returns the offsets and the slope angle actually achieved, which the caller
    reports rather than quietly presenting the requested angle as delivered.
    """
    bx, by, bz = block_size
    angle = np.radians(np.clip(slope_angle_deg, 5.0, 85.0))
    run_per_bench = bz / np.tan(angle)

    # enough benches for the horizontal run to resolve to a few blocks
    n_benches = int(np.clip(np.ceil(2.0 * bx / max(run_per_bench, 1e-9)), 1, max_benches))

    eps = 1e-6  # relative, so an exact 2:1 ratio is not lost to floating point
    offsets: list[tuple[int, int, int]] = []
    for dk in range(1, n_benches + 1):
        radius = run_per_bench * dk
        inner = run_per_bench * (dk - 1)
        ni = int(np.ceil(radius / bx))
        nj = int(np.ceil(radius / by))
        for di in range(-ni, ni + 1):
            for dj in range(-nj, nj + 1):
                if di == 0 and dj == 0 and dk > 1:
                    continue  # already implied by stacking (0, 0, 1)
                dist = np.hypot(di * bx, dj * by)
                # only the new ring at this bench; inner rings are implied
                # transitively by composing shorter arcs
                if dist <= radius * (1 + eps) and dist > inner * (1 + eps):
                    offsets.append((di, dj, dk))

    if not offsets:  # a near-vertical wall still needs the block directly above
        offsets = [(0, 0, 1)]

    # The flattest arc in the template governs the wall the optimizer can cut.
    lateral = [(di, dj, dk) for di, dj, dk in offsets if di or dj]
    achieved = min(
        (np.degrees(np.arctan2(dk * bz, np.hypot(di * bx, dj * by)))
         for di, dj, dk in lateral), default=89.0)
    return np.asarray(offsets, dtype=np.int64), float(achieved)


def _solve_max_closure(values: np.ndarray, arcs_from: np.ndarray,
                       arcs_to: np.ndarray) -> np.ndarray:
    """Maximum-weight closure by minimum cut. Returns the in-pit boolean mask."""
    n = len(values)
    positive = values > 0
    total_positive = float(values[positive].sum())
    if total_positive <= 0:
        log.warning("no block has a positive value at these economics; the pit is empty")
        return np.zeros(n, dtype=bool)

    scale = _INT32_HEADROOM / max(total_positive, 1e-9)
    scale = min(scale, 1e6)  # no point resolving below a millionth of a currency unit

    cap_source = np.where(positive, np.maximum(np.round(values * scale), 1), 0).astype(np.int64)
    cap_sink = np.where(~positive, np.maximum(np.round(-values * scale), 0), 0).astype(np.int64)
    infinite = int(cap_source.sum()) + 1

    source, sink = n, n + 1
    rows = np.concatenate([
        np.full(int(positive.sum()), source, dtype=np.int64),
        np.flatnonzero(~positive).astype(np.int64),
        arcs_from,
    ])
    cols = np.concatenate([
        np.flatnonzero(positive).astype(np.int64),
        np.full(int((~positive).sum()), sink, dtype=np.int64),
        arcs_to,
    ])
    data = np.concatenate([
        cap_source[positive],
        cap_sink[~positive],
        np.full(len(arcs_from), infinite, dtype=np.int64),
    ])

    keep = data > 0
    rows, cols, data = rows[keep], cols[keep], data[keep]

    if data.max() > np.iinfo(np.int32).max:
        raise PitOptimizationError(
            "scaled block values overflow the 32-bit min-cut solver; "
            "reduce the block count or the economic magnitudes")

    graph = csr_matrix((data.astype(np.int32), (rows, cols)), shape=(n + 2, n + 2))
    result = maximum_flow(graph, source, sink)

    # blocks still reachable from the source in the residual network form the pit
    residual = result.flow
    cap = graph.tocsr()
    residual_cap = cap - residual
    residual_cap.eliminate_zeros()

    visited = np.zeros(n + 2, dtype=bool)
    visited[source] = True
    stack = [source]
    indptr, indices, values_r = residual_cap.indptr, residual_cap.indices, residual_cap.data
    while stack:
        node = stack.pop()
        for p in range(indptr[node], indptr[node + 1]):
            nxt = indices[p]
            if values_r[p] > 0 and not visited[nxt]:
                visited[nxt] = True
                stack.append(nxt)

    return visited[:n]


def run_lerchs_grossmann(block_model: pd.DataFrame, economic_values: EconomicParameters | dict,
                         slope_angles: float | dict | None = None,
                         block_size: tuple[float, float, float] | None = None,
                         revenue_factor: float = 1.0,
                         grade_column: str = "grade") -> PitResult:
    """Compute the ultimate pit limit, its tonnages and its strip ratio."""
    econ = (economic_values if isinstance(economic_values, EconomicParameters)
            else EconomicParameters(**economic_values))
    if slope_angles is not None:
        if isinstance(slope_angles, dict):
            econ.slope_angles_by_sector = slope_angles
            if "overall" in slope_angles:
                econ.overall_slope_angle_deg = float(slope_angles["overall"])
        else:
            econ.overall_slope_angle_deg = float(slope_angles)

    if block_size is None:
        block_size = block_model.attrs.get("block_size")
    if block_size is None:
        raise PitOptimizationError("block size is required to build the precedence graph")
    block_size = tuple(float(v) for v in block_size)

    if "tonnes" not in block_model.columns:
        raise PitOptimizationError("block model must carry a tonnes column")

    values = compute_block_values(block_model, econ, revenue_factor, grade_column)

    i, j, k = _grid_indices(block_model, block_size)
    ni, nj, nk = int(i.max()) + 1, int(j.max()) + 1, int(k.max()) + 1
    lookup = np.full((ni, nj, nk), -1, dtype=np.int64)
    lookup[i, j, k] = np.arange(len(block_model))

    offsets, achieved_slope = _precedence_offsets(block_size, econ.overall_slope_angle_deg)
    arcs_from, arcs_to = [], []
    base = np.arange(len(block_model))
    for di, dj, dk in offsets:
        ii, jj, kk = i + di, j + dj, k + dk
        valid = (ii >= 0) & (ii < ni) & (jj >= 0) & (jj < nj) & (kk >= 0) & (kk < nk)
        if not valid.any():
            continue
        target = lookup[ii[valid], jj[valid], kk[valid]]
        exists = target >= 0
        arcs_from.append(base[valid][exists])
        arcs_to.append(target[exists])

    if arcs_from:
        arcs_from = np.concatenate(arcs_from)
        arcs_to = np.concatenate(arcs_to)
    else:
        arcs_from = np.zeros(0, dtype=np.int64)
        arcs_to = np.zeros(0, dtype=np.int64)

    log.info("pit graph: %d blocks, %d precedence arcs, slope %.1f degrees requested "
             "(%.1f achievable on this block size)",
             len(block_model), len(arcs_from), econ.overall_slope_angle_deg, achieved_slope)
    if abs(achieved_slope - econ.overall_slope_angle_deg) > 1.5:
        log.warning("the %.1f degree slope cannot be represented exactly on a %.1f m "
                    "block; the pit is cut at %.1f degrees",
                    econ.overall_slope_angle_deg, block_size[0], achieved_slope)

    in_pit = _solve_max_closure(values, arcs_from, arcs_to)
    summary = summarise_pit(block_model, in_pit, values, econ, block_size, grade_column)
    summary["slope_angle_requested_deg"] = float(econ.overall_slope_angle_deg)
    summary["slope_angle_achieved_deg"] = float(achieved_slope)
    summary["n_precedence_arcs"] = int(len(arcs_from))
    surface = pit_surface(block_model, in_pit, block_size)

    log.info("ultimate pit: %.2f Mt ore, %.2f Mt waste, strip ratio %.2f, value %.1f M",
             summary["ore_tonnes"] / 1e6, summary["waste_tonnes"] / 1e6,
             summary["strip_ratio"], summary["pit_value"] / 1e6)

    return PitResult(in_pit=in_pit, economics=econ, summary=summary,
                     block_values=values, pit_surface=surface)


def summarise_pit(blocks: pd.DataFrame, in_pit: np.ndarray, values: np.ndarray,
                  econ: EconomicParameters, block_size: tuple[float, float, float],
                  grade_column: str = "grade") -> dict:
    """Tonnage, grade, strip ratio and value inside the shell."""
    bx, by, bz = block_size
    block_volume = bx * by * bz

    pit = blocks[in_pit]
    tonnes = pit["tonnes"].to_numpy(float) if len(pit) else np.zeros(0)
    grade = pit[grade_column].to_numpy(float) if len(pit) else np.zeros(0)
    grade = np.where(np.isfinite(grade), grade, 0.0)

    is_air = pit["is_air"].to_numpy(bool) if "is_air" in pit.columns else np.zeros(len(pit), bool)
    cutoff = econ.internal_cutoff_grade()
    is_ore = (grade >= cutoff) & ~is_air
    is_waste = ~is_ore & ~is_air

    ore_tonnes = float(tonnes[is_ore].sum())
    waste_tonnes = float(tonnes[is_waste].sum())
    metal_g = float((tonnes[is_ore] * grade[is_ore]).sum())
    mean_grade = metal_g / ore_tonnes if ore_tonnes > 0 else 0.0

    revenue = ore_tonnes * mean_grade * econ.revenue_per_tonne_per_grade_unit
    mining_cost = (ore_tonnes + waste_tonnes) * econ.mining_cost_per_tonne
    processing_cost = ore_tonnes * (econ.processing_cost_per_tonne + econ.ga_cost_per_tonne)

    z = blocks["z"].to_numpy(float)
    pit_z = z[in_pit] if in_pit.any() else np.zeros(1)

    return {
        "n_blocks_in_pit": int(in_pit.sum()),
        "n_blocks_total": int(len(blocks)),
        "pit_volume_m3": float(in_pit.sum() * block_volume),
        "ore_tonnes": ore_tonnes,
        "waste_tonnes": waste_tonnes,
        "total_tonnes": ore_tonnes + waste_tonnes,
        "strip_ratio": float(waste_tonnes / ore_tonnes) if ore_tonnes > 0 else float("inf"),
        "mean_ore_grade": float(mean_grade),
        "internal_cutoff_grade": float(cutoff),
        "contained_metal_g": metal_g,
        "contained_metal_oz": float(metal_g / GRAMS_PER_OZ),
        "recovered_metal_oz": float(metal_g / GRAMS_PER_OZ * econ.recovery_pct / 100.0),
        "gross_revenue": float(revenue),
        "mining_cost": float(mining_cost),
        "processing_cost": float(processing_cost),
        "pit_value": float(values[in_pit].sum()),
        "undiscounted_cashflow": float(revenue - mining_cost - processing_cost),
        "pit_depth_m": float(pit_z.max() - pit_z.min()) if in_pit.any() else 0.0,
        "pit_crest_rl": float(pit_z.max()) if in_pit.any() else 0.0,
        "pit_floor_rl": float(pit_z.min()) if in_pit.any() else 0.0,
        "block_size": [bx, by, bz],
    }


def pit_surface(blocks: pd.DataFrame, in_pit: np.ndarray,
                block_size: tuple[float, float, float]) -> dict | None:
    """Pit floor and real-rock crest elevation per column, for the 3D scene.

    Two grids, not one. Precedence pulls "air" blocks (grid cells above
    topography, worth nothing) into the closure whenever real material below
    them is worth mining -- there is no cost to including them, so the
    optimizer always does. A column's *floor* is unaffected by that (air is
    always the shallowest thing in a column, never the deepest), but a
    column's true top is wherever real rock actually ends, not wherever the
    padded block grid happens to end above it. Reporting only a floor grid
    and letting a caller infer the crest from it -- or from the raw in-pit
    block range -- silently counts that overhead air as material at every
    bench between the real ground surface and the inflated crest.
    """
    if not in_pit.any():
        return None
    bx, by, bz = block_size
    pit = blocks[in_pit]
    x = pit["x"].to_numpy(float)
    y = pit["y"].to_numpy(float)
    z = pit["z"].to_numpy(float)
    is_air = (pit["is_air"].to_numpy(bool) if "is_air" in pit.columns
             else np.zeros(len(pit), dtype=bool))

    # A contiguous index range, not just the columns that happen to contain pit
    # blocks: indexing by unique occupied columns closes the gaps, and the mesh
    # then stretches quads across them into spurious spikes at the pit rim.
    xi_all = np.round(x / bx).astype(int)
    yi_all = np.round(y / by).astype(int)
    xs = np.arange(xi_all.min(), xi_all.max() + 1)
    ys = np.arange(yi_all.min(), yi_all.max() + 1)
    xi = xi_all - xs[0]
    yi = yi_all - ys[0]

    # seeded with +inf, not NaN: np.minimum against NaN would propagate the NaN
    floor = np.full((len(xs), len(ys)), np.inf)
    np.minimum.at(floor, (xi, yi), z)
    floor[~np.isfinite(floor)] = np.nan

    # crest from real rock only (air excluded): seeded with -inf so a column
    # with no rock at all -- air pulled in above nothing, which should not
    # happen but is handled rather than silently mis-seeded -- reports as NaN
    rock = ~is_air
    crest = np.full((len(xs), len(ys)), -np.inf)
    if rock.any():
        np.maximum.at(crest, (xi[rock], yi[rock]), z[rock])
    crest[~np.isfinite(crest)] = np.nan

    return {
        "x": [float(v * bx) for v in xs],
        "y": [float(v * by) for v in ys],
        "floor_z": [[None if not np.isfinite(v) else float(v - bz / 2.0) for v in row]
                    for row in floor],
        "crest_z": [[None if not np.isfinite(v) else float(v + bz / 2.0) for v in row]
                    for row in crest],
        "block_size": [bx, by, bz],
    }


def _floor_and_crest_grids(pit_surface_data: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract the floor and real-rock-crest grids, tolerating an older/
    synthetic ``pit_surface_data`` that has no ``crest_z`` (treated as
    unbounded above, i.e. no air to exclude). Returns ``(floor, crest,
    crest_known)`` -- callers need the flag because "unbounded" and "no data"
    both leave every finite crest cell at +inf, and the two mean different
    things when deciding how high the crest of the whole pit sits."""
    floor = np.array([[np.nan if v is None else v for v in row]
                      for row in pit_surface_data["floor_z"]], dtype=float)
    crest_rows = pit_surface_data.get("crest_z")
    if crest_rows is None:
        crest = np.full_like(floor, np.inf)
        crest[~np.isfinite(floor)] = np.nan
        return floor, crest, False
    crest = np.array([[np.nan if v is None else v for v in row] for row in crest_rows],
                     dtype=float)
    return floor, crest, True


def bench_contours(pit_surface_data: dict, n_levels: int = 24) -> list[dict]:
    """Slice the pit floor surface into elevation-banded outline rings.

    Because Lerchs-Grossmann's precedence constraint makes the in-pit set
    upward-closed in every column (a block is only ever mined if the block
    above it is too), a column belongs to the pit at elevation E exactly when
    E lies between its floor RL and its real-rock crest RL. That turns the
    floor and crest grids already computed by ``pit_surface`` into a plan-view
    mask at *any* elevation, not just the block model's own bench spacing --
    which is what lets the ring spacing here be chosen independently of block
    size, the same way "quick planes" are cut through a solid at a chosen
    interval rather than at its native mesh resolution. The crest bound
    matters as much as the floor one: without it, a column whose real rock
    ends well below the pit's overall crest still shows up in every ring up
    to that crest, because nothing stops it once the floor test alone is
    satisfied.

    Returns one entry per level: ``{"z": elevation, "loops": [(N,2) world xy, ...]}``.
    """
    if pit_surface_data is None:
        return []

    floor, crest, crest_known = _floor_and_crest_grids(pit_surface_data)
    if not np.isfinite(floor).any():
        return []

    xs = np.asarray(pit_surface_data["x"], float)
    ys = np.asarray(pit_surface_data["y"], float)
    bx, by, _ = pit_surface_data["block_size"]

    z_min = float(np.nanmin(floor))
    if crest_known:
        finite_crest = crest[~np.isnan(crest) & np.isfinite(crest)]
        z_max = float(finite_crest.max()) if finite_crest.size else z_min
    else:
        z_max = float(np.nanmax(floor))  # legacy behaviour: no crest data supplied
    if z_max - z_min < 1e-6:
        return []

    from ..report.contours import loop_area, smooth_loop, trace_contours

    levels = np.linspace(z_min, z_max, max(int(n_levels), 2))
    # a column counts if it has a floor and is not known to have zero rock;
    # +inf crest (no crest data supplied) deliberately passes this test
    finite = np.isfinite(floor) & ~np.isnan(crest)

    bands = []
    for level in levels:
        mask = finite & (floor <= level + 1e-6) & (level - 1e-6 <= crest)
        loops = trace_contours(mask)
        if not loops:
            continue
        world_loops = []
        for loop in loops:
            if loop_area(loop) < 2:  # a sliver a couple of cells across is noise
                continue
            smoothed = smooth_loop(loop)
            world = np.column_stack([
                xs[0] + smoothed[:, 0] * bx,
                ys[0] + smoothed[:, 1] * by,
            ])
            world_loops.append(world)
        if world_loops:
            bands.append({"z": float(level), "loops": world_loops})

    return bands


def bench_schedule(pit_surface_data: dict, bench_height: float, density: float,
                   n_sectors: int = 8) -> dict:
    """The real per-bench schedule: RL, footprint, volume and tonnes, bench by bench.

    A column contributes to bench N only when its rock fully spans that
    bench's height -- floor at or below the bench's floor RL, and real-rock
    crest at or above the bench's own crest RL. The crest bound is not
    optional: without it, a column whose material actually ends at, say,
    bench 4 keeps counting as present in every bench above that too, because
    nothing stops it once the floor half of the test is satisfied. That crest
    is read from ``pit_surface_data["crest_z"]``, which -- unlike the block
    model's raw z-range -- already excludes the "air" blocks precedence pulls
    into the closure purely to unlock real material beneath them. Air is not
    rock and does not belong in a tonnes-per-bench figure.

    Bench 1 is the crest (mined first); bench N is the floor. Each bench's
    footprint is an exact cell count, not a smoothed estimate, so the volumes
    sum to the same total the optimizer reports for real (non-air) material.

    Also reports, per bench, the minimum straight-line footprint half-width
    around ``n_sectors`` compass sectors -- the geometric quantity a batter
    and berm design is actually constrained by (a bench pinched to a few
    metres in one direction cannot carry a full berm there even if the
    average width is generous), without asserting a batter angle, berm width
    or factor of safety that would need geotechnical data this pipeline was
    never given.
    """
    if pit_surface_data is None or bench_height <= 0:
        return {"benches": [], "n_benches": 0}

    floor, crest, crest_known = _floor_and_crest_grids(pit_surface_data)
    finite = np.isfinite(floor)
    if not finite.any():
        return {"benches": [], "n_benches": 0}

    bx, by, _ = pit_surface_data["block_size"]
    cell_area = bx * by
    xs = np.asarray(pit_surface_data["x"], float)
    ys = np.asarray(pit_surface_data["y"], float)

    z_min = float(np.nanmin(floor))
    if crest_known:
        finite_crest = crest[~np.isnan(crest) & np.isfinite(crest)]
        z_max = float(finite_crest.max()) if finite_crest.size else z_min
    else:
        z_max = float(np.nanmax(floor))  # legacy fallback: no crest data supplied
    n_benches = max(int(np.ceil((z_max - z_min) / bench_height - 1e-6)), 1)

    # bench 1 = crest (top), bench n_benches = floor (bottom)
    floor_rls = [z_max - (i + 1) * bench_height for i in range(n_benches)]
    floor_rls[-1] = z_min  # the last bench ends exactly at the true floor

    ii, jj = np.meshgrid(np.arange(floor.shape[0]), np.arange(floor.shape[1]), indexing="ij")
    cx = xs[0] + ii * bx
    cy = ys[0] + jj * by

    benches = []
    cumulative_volume = 0.0
    for bench_no, floor_rl in enumerate(floor_rls, start=1):
        crest_rl = floor_rl + bench_height
        mask = finite & (floor <= floor_rl + 1e-6) & (crest >= crest_rl - 1e-6)
        n_cells = int(mask.sum())
        footprint_area = n_cells * cell_area
        volume = footprint_area * bench_height
        cumulative_volume += volume

        half_widths = []
        if n_cells > 0:
            centre_x = float(cx[mask].mean())
            centre_y = float(cy[mask].mean())
            dx = cx[mask] - centre_x
            dy = cy[mask] - centre_y
            for s in range(n_sectors):
                angle = 2 * np.pi * s / n_sectors
                proj = dx * np.cos(angle) + dy * np.sin(angle)
                half_widths.append(float(proj.max()) if proj.size else 0.0)

        benches.append({
            "bench": bench_no,
            "floor_rl": float(floor_rl),
            "crest_rl": float(crest_rl),
            "footprint_area_m2": float(footprint_area),
            "volume_m3": float(volume),
            "tonnes": float(volume * density),
            "cumulative_volume_m3": float(cumulative_volume),
            "cumulative_tonnes": float(cumulative_volume * density),
            "min_half_width_m": float(min(half_widths)) if half_widths else 0.0,
        })

    narrowest = min((b["min_half_width_m"] for b in benches), default=0.0)

    return {
        "benches": benches,
        "n_benches": n_benches,
        "bench_height_m": float(bench_height),
        "crest_rl": z_max,
        "floor_rl": z_min,
        "depth_m": float(z_max - z_min),
        "total_volume_m3": cumulative_volume,
        "total_tonnes": cumulative_volume * density,
        "narrowest_half_width_m": float(narrowest),
    }


def nested_pit_shells(block_model: pd.DataFrame, econ: EconomicParameters,
                      revenue_factors: np.ndarray | None = None,
                      block_size: tuple[float, float, float] | None = None,
                      grade_column: str = "grade") -> list[dict]:
    """Run the optimizer across a sweep of revenue factors.

    Nested shells are how a pit's sensitivity to price is actually read: the
    inner shells are the tonnes that pay for themselves even at a low price.
    """
    if revenue_factors is None:
        revenue_factors = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

    shells = []
    for factor in revenue_factors:
        try:
            result = run_lerchs_grossmann(block_model, econ, block_size=block_size,
                                          revenue_factor=float(factor),
                                          grade_column=grade_column)
        except PitOptimizationError as exc:
            log.warning("nested shell at revenue factor %.2f failed: %s", factor, exc)
            continue
        s = result.summary
        shells.append({
            "revenue_factor": float(factor),
            "effective_price": float(econ.commodity_price * factor),
            "n_blocks": s["n_blocks_in_pit"],
            "ore_tonnes": s["ore_tonnes"],
            "waste_tonnes": s["waste_tonnes"],
            "strip_ratio": s["strip_ratio"],
            "mean_ore_grade": s["mean_ore_grade"],
            "contained_metal_oz": s["contained_metal_oz"],
            "pit_value": s["pit_value"],
        })
    return shells
