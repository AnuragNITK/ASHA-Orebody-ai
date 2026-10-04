"""Pit optimization, compositing, statistics and estimation tests.

The pit tests are the important ones: an ultimate pit that is not a valid
closure, or not the true optimum, is worse than no pit at all, so both
properties are checked against brute force on models small enough to enumerate.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from minedesign.blockmodel.kriging import inverse_distance, ordinary_kriging
from minedesign.blockmodel.variogram import experimental_variogram, fit_all_models, variogram_model
from minedesign.geology.compositing import apply_top_cut, composite, recommend_composite_length
from minedesign.geology.stats import choose_color_breakpoints, compute_grade_stats
from minedesign.pit.lerchs_grossmann import (
    EconomicParameters,
    _precedence_offsets,
    bench_contours,
    bench_schedule,
    pit_surface,
    run_lerchs_grossmann,
)
from minedesign.report.contours import loop_area, smooth_loop, trace_contours

BLOCK = (10.0, 10.0, 10.0)


def make_model(nx, ny, nz, grades=None, seed=0):
    rows = [{"x": i * 10.0, "y": j * 10.0, "z": k * 10.0}
            for i in range(nx) for j in range(ny) for k in range(nz)]
    df = pd.DataFrame(rows)
    df["tonnes"] = 2700.0
    df["grade"] = (np.random.default_rng(seed).uniform(0, 3, len(df))
                   if grades is None else grades)
    df.attrs["block_size"] = BLOCK
    return df


def closure_violations(model, in_pit, angle):
    offsets, _ = _precedence_offsets(BLOCK, angle)
    xi = np.round(model.x / 10).astype(int)
    yi = np.round(model.y / 10).astype(int)
    zi = np.round(model.z / 10).astype(int)
    pos = {(a, b, c): m for m, (a, b, c) in enumerate(zip(xi, yi, zi))}
    inside = set(np.flatnonzero(in_pit))
    violations = 0
    for m in inside:
        for di, dj, dk in offsets:
            target = pos.get((xi[m] + di, yi[m] + dj, zi[m] + dk))
            if target is not None and target not in inside:
                violations += 1
    return violations


class TestPitOptimizer:
    def test_matches_brute_force_optimum(self):
        model = make_model(4, 1, 3, seed=3)
        result = run_lerchs_grossmann(model, EconomicParameters(overall_slope_angle_deg=45))
        values = result.block_values

        index = {(int(model.x[m] / 10), int(model.z[m] / 10)): m for m in range(len(model))}

        def is_closed(subset):
            for m in subset:
                a, c = int(model.x[m] / 10), int(model.z[m] / 10)
                if c < 2:
                    for da in (-1, 0, 1):
                        t = index.get((a + da, c + 1))
                        if t is not None and t not in subset:
                            return False
            return True

        best = 0.0
        for r in range(len(model) + 1):
            for combo in itertools.combinations(range(len(model)), r):
                if is_closed(set(combo)):
                    best = max(best, float(values[list(combo)].sum()) if combo else 0.0)
        assert float(values[result.in_pit].sum()) == pytest.approx(best)

    @pytest.mark.parametrize("angle", [45.0, 63.43])
    def test_pit_is_a_valid_closure(self, angle):
        model = make_model(9, 9, 4, seed=11)
        result = run_lerchs_grossmann(model, EconomicParameters(overall_slope_angle_deg=angle))
        assert closure_violations(model, result.in_pit, angle) == 0

    def test_worthless_deposit_gives_an_empty_pit(self):
        model = make_model(5, 5, 3, grades=0.0)
        result = run_lerchs_grossmann(model, EconomicParameters())
        assert not result.in_pit.any()
        assert result.summary["ore_tonnes"] == 0.0

    def test_steeper_slope_removes_less_waste(self):
        grades = np.zeros(21 * 1 * 6)
        model = make_model(21, 1, 6, grades=grades)
        model.loc[(model.x == 100) & (model.z == 0), "grade"] = 500.0
        shallow = run_lerchs_grossmann(model, EconomicParameters(overall_slope_angle_deg=45))
        steep = run_lerchs_grossmann(model, EconomicParameters(overall_slope_angle_deg=63.43))
        assert steep.in_pit.sum() < shallow.in_pit.sum()

    def test_achieved_slope_is_reported(self):
        model = make_model(6, 6, 3, seed=5)
        result = run_lerchs_grossmann(model, EconomicParameters(overall_slope_angle_deg=40))
        assert "slope_angle_achieved_deg" in result.summary
        assert result.summary["slope_angle_requested_deg"] == 40.0

    @pytest.mark.parametrize("angle,expected", [(45.0, 45.0), (63.43, 63.43)])
    def test_exactly_representable_angles_are_exact(self, angle, expected):
        _, achieved = _precedence_offsets(BLOCK, angle)
        assert achieved == pytest.approx(expected, abs=0.1)

    def test_higher_price_never_shrinks_the_pit(self):
        model = make_model(9, 9, 4, seed=21)
        sizes = []
        for price in (800.0, 1600.0, 3200.0):
            result = run_lerchs_grossmann(model, EconomicParameters(commodity_price=price))
            sizes.append(int(result.in_pit.sum()))
        assert sizes == sorted(sizes)


class TestContourTracing:
    def test_disk_area_matches_analytic(self):
        import math
        n = 120
        y, x = np.ogrid[-n // 2:n // 2, -n // 2:n // 2]
        loop = trace_contours((x ** 2 + y ** 2) <= 30 ** 2)[0]
        assert loop_area(loop) == pytest.approx(math.pi * 900, rel=0.02)

    def test_smoothing_keeps_the_area_close(self):
        n = 100
        y, x = np.ogrid[-n // 2:n // 2, -n // 2:n // 2]
        loop = trace_contours((x ** 2 + y ** 2) <= 25 ** 2)[0]
        assert loop_area(smooth_loop(loop)) == pytest.approx(loop_area(loop), rel=0.05)

    def test_empty_mask_gives_no_loops(self):
        assert trace_contours(np.zeros((10, 10), bool)) == []


class TestBenchContours:
    """The pit shell's elevation-banded rings, built from the floor grid.

    A column belongs to the pit at elevation E exactly when its floor RL is at
    or below E -- valid because Lerchs-Grossmann's precedence constraint makes
    the in-pit set upward-closed in every column. These tests check that
    property translates into rings that shrink monotonically towards the floor
    and that the count and geometry match what was asked for.
    """

    def _cone_surface(self, n=50, bx=10.0, by=10.0, slope=1.2):
        xs = np.arange(n) * bx
        ys = np.arange(n) * by
        cx, cy = xs.mean(), ys.mean()
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        r = np.hypot(gx - cx, gy - cy)
        floor = -200.0 + r * slope
        floor[r > 220] = np.nan
        return {
            "x": xs.tolist(), "y": ys.tolist(),
            "floor_z": [[None if np.isnan(v) else float(v) for v in row] for row in floor],
            "block_size": [bx, by, 10.0],
        }

    def test_produces_the_requested_number_of_levels(self):
        bands = bench_contours(self._cone_surface(), n_levels=15)
        assert len(bands) == 15

    def test_rings_widen_from_floor_to_crest(self):
        bands = bench_contours(self._cone_surface(), n_levels=12)
        areas = []
        for band in bands:
            total = 0.0
            for loop in band["loops"]:
                x, y = loop[:, 0], loop[:, 1]
                total += abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0
            areas.append(total)
        # bands are ordered floor (lowest z) to crest (highest z)
        assert all(b["z"] for b in bands)  # sanity: every band carries an elevation
        assert areas == sorted(areas)

    def test_ring_area_matches_the_analytic_cone_cross_section(self):
        bands = bench_contours(self._cone_surface(n=80, slope=1.0), n_levels=10)
        mid = bands[len(bands) // 2]
        z = mid["z"]
        expected_radius = (z + 200.0) / 1.0  # inverse of floor = -200 + r*slope
        expected_area = np.pi * expected_radius ** 2
        actual = sum(
            abs(np.dot(loop[:, 0], np.roll(loop[:, 1], -1))
               - np.dot(loop[:, 1], np.roll(loop[:, 0], -1))) / 2.0
            for loop in mid["loops"])
        assert actual == pytest.approx(expected_area, rel=0.1)

    def test_no_pit_gives_no_bands(self):
        assert bench_contours(None) == []
        flat = self._cone_surface()
        for row in flat["floor_z"]:
            for i in range(len(row)):
                row[i] = None
        assert bench_contours(flat) == []

    def test_bench_contours_from_a_real_optimized_pit(self):
        """End to end: a genuine Lerchs-Grossmann result feeds real rings."""
        rng = np.random.default_rng(5)
        n = 14
        rows = [{"x": i * 10.0, "y": j * 10.0, "z": k * 10.0}
               for i in range(n) for j in range(n) for k in range(5)]
        model = pd.DataFrame(rows)
        cx, cy = (n - 1) * 10 / 2, (n - 1) * 10 / 2
        r = np.hypot(model.x - cx, model.y - cy)
        model["grade"] = np.where(r < 40, rng.uniform(1, 3, len(model)), 0.02)
        model["tonnes"] = 2700.0

        result = run_lerchs_grossmann(model, EconomicParameters(), block_size=(10.0, 10.0, 10.0))
        assert result.in_pit.any(), "the synthetic deposit should produce a non-empty pit"

        surface = pit_surface(model, result.in_pit, (10.0, 10.0, 10.0))
        bands = bench_contours(surface, n_levels=8)
        assert len(bands) > 0
        assert all(len(b["loops"]) > 0 for b in bands)


class TestBenchSchedule:
    """The real, per-bench schedule: RL, footprint, volume, tonnes.

    Every number here has to trace back to the block model exactly -- this is
    the table a report reads as "number of benches" and "bench height", so an
    approximation that merely looks plausible is worse than an obvious error.
    """

    def _synthetic_pit(self, n=20, radius=60.0, seed=5, levels=8, block=10.0,
                       air_above=None):
        rng = np.random.default_rng(seed)
        rows = [{"x": i * block, "y": j * block, "z": k * block}
               for i in range(n) for j in range(n) for k in range(levels)]
        model = pd.DataFrame(rows)
        cx, cy = (n - 1) * block / 2, (n - 1) * block / 2
        r = np.hypot(model.x - cx, model.y - cy)
        model["grade"] = np.where(r < radius, rng.uniform(1, 3, len(model)), 0.02)
        model["tonnes"] = 2700.0
        if air_above is not None:
            model["is_air"] = model["z"] > air_above
        result = run_lerchs_grossmann(model, EconomicParameters(),
                                      block_size=(block, block, block))
        assert result.in_pit.any()
        return model, result, block

    def test_total_volume_matches_the_optimizer_exactly(self):
        """No air marked: the optimizer's pit_volume_m3 counts every in-pit
        block, and so should a schedule built from the same in-pit set."""
        model, result, block = self._synthetic_pit()
        surface = pit_surface(model, result.in_pit, (block, block, block))
        schedule = bench_schedule(surface, bench_height=block, density=2.7)
        assert schedule["total_volume_m3"] == pytest.approx(
            result.summary["pit_volume_m3"], rel=1e-9)

    def test_air_blocks_are_excluded_from_volume_and_tonnage(self):
        """Regression test for the real bug this function existed to fix.

        Precedence pulls "air" blocks -- grid cells above topography, worth
        nothing -- into the closure whenever real material beneath them is
        worth mining, at no cost, so the optimizer always includes them. An
        earlier version of this function counted that air as material,
        overstating volume by the full padded-overburden amount (37% on the
        deposit this was first caught on) with no error raised. The schedule
        must match the *real-rock* volume, not the raw in-pit block count.
        """
        model, result, block = self._synthetic_pit(air_above=65.0)
        assert (result.in_pit & model["is_air"].to_numpy()).sum() > 0, (
            "the test setup must actually produce some in-pit air blocks")
        surface = pit_surface(model, result.in_pit, (block, block, block))
        schedule = bench_schedule(surface, bench_height=block, density=2.7)

        real_rock_volume = float((result.in_pit & ~model["is_air"].to_numpy()).sum()
                                 * block ** 3)
        assert schedule["total_volume_m3"] == pytest.approx(real_rock_volume, rel=1e-9)
        # and it must be a real reduction, not a no-op on this fixture
        assert schedule["total_volume_m3"] < result.summary["pit_volume_m3"]

    def test_bench_count_matches_true_z_level_count(self):
        model, result, block = self._synthetic_pit(levels=8)
        surface = pit_surface(model, result.in_pit, (block, block, block))
        schedule = bench_schedule(surface, bench_height=block, density=2.7)
        n_true_levels = model.loc[result.in_pit, "z"].nunique()
        assert schedule["n_benches"] == n_true_levels

    def test_bench_count_excludes_air_only_levels(self):
        model, result, block = self._synthetic_pit(air_above=65.0)
        surface = pit_surface(model, result.in_pit, (block, block, block))
        schedule = bench_schedule(surface, bench_height=block, density=2.7)
        n_real_rock_levels = model.loc[result.in_pit & ~model["is_air"], "z"].nunique()
        assert schedule["n_benches"] == n_real_rock_levels

    def test_per_bench_footprint_matches_ground_truth_block_counts(self):
        model, result, block = self._synthetic_pit()
        surface = pit_surface(model, result.in_pit, (block, block, block))
        schedule = bench_schedule(surface, bench_height=block, density=2.7)

        pit = model[result.in_pit]
        true_counts = pit.groupby("z").size().sort_index(ascending=False).to_numpy()
        schedule_counts = np.array([b["footprint_area_m2"] / (block * block)
                                    for b in schedule["benches"]])
        assert np.array_equal(schedule_counts, true_counts)

    def test_benches_widen_from_floor_to_crest(self):
        model, result, block = self._synthetic_pit()
        surface = pit_surface(model, result.in_pit, (block, block, block))
        schedule = bench_schedule(surface, bench_height=block, density=2.7)
        areas = [b["footprint_area_m2"] for b in schedule["benches"]]  # bench 1 = crest
        assert areas == sorted(areas, reverse=True)

    def test_cumulative_tonnes_is_monotonic_and_ends_at_the_total(self):
        model, result, block = self._synthetic_pit()
        surface = pit_surface(model, result.in_pit, (block, block, block))
        schedule = bench_schedule(surface, bench_height=block, density=2.7)
        cumulative = [b["cumulative_tonnes"] for b in schedule["benches"]]
        assert cumulative == sorted(cumulative)
        assert cumulative[-1] == pytest.approx(schedule["total_tonnes"])

    def test_narrowest_half_width_is_the_true_minimum(self):
        model, result, block = self._synthetic_pit()
        surface = pit_surface(model, result.in_pit, (block, block, block))
        schedule = bench_schedule(surface, bench_height=block, density=2.7)
        widths = [b["min_half_width_m"] for b in schedule["benches"]]
        assert schedule["narrowest_half_width_m"] == pytest.approx(min(widths))

    def test_empty_pit_returns_an_empty_schedule(self):
        schedule = bench_schedule(None, bench_height=10.0, density=2.7)
        assert schedule["benches"] == [] and schedule["n_benches"] == 0

    def test_tonnes_follows_density(self):
        model, result, block = self._synthetic_pit()
        surface = pit_surface(model, result.in_pit, (block, block, block))
        a = bench_schedule(surface, bench_height=block, density=2.0)
        b = bench_schedule(surface, bench_height=block, density=4.0)
        assert b["total_tonnes"] == pytest.approx(a["total_tonnes"] * 2, rel=1e-9)


class TestEconomics:
    def test_internal_cutoff_is_the_break_even_grade(self):
        econ = EconomicParameters(commodity_price=2000.0, recovery_pct=100.0,
                                  processing_cost_per_tonne=30.0, ga_cost_per_tonne=0.0)
        cutoff = econ.internal_cutoff_grade()
        # revenue from one tonne at exactly the cutoff must equal processing cost
        revenue = cutoff * econ.revenue_per_tonne_per_grade_unit
        assert revenue == pytest.approx(30.0)

    def test_recovery_scales_revenue(self):
        full = EconomicParameters(recovery_pct=100.0).revenue_per_tonne_per_grade_unit
        half = EconomicParameters(recovery_pct=50.0).revenue_per_tonne_per_grade_unit
        assert half == pytest.approx(full / 2)


class TestCompositing:
    def test_modal_sample_length_is_recommended(self):
        assay = pd.DataFrame({"hole_id": ["A"] * 10,
                              "from": np.arange(10.0), "to": np.arange(1.0, 11.0),
                              "grade": 1.0})
        assert recommend_composite_length(assay)["length"] == pytest.approx(1.0)

    def test_composite_grade_is_length_weighted(self):
        assay = pd.DataFrame({
            "hole_id": ["A", "A"], "from": [0.0, 1.0], "to": [1.0, 2.0], "grade": [1.0, 3.0]})
        out = composite(assay, 2.0)
        assert len(out) == 1
        assert out["grade"].iloc[0] == pytest.approx(2.0)

    def test_uneven_supports_are_weighted_by_length(self):
        assay = pd.DataFrame({
            "hole_id": ["A", "A"], "from": [0.0, 1.0], "to": [1.0, 4.0], "grade": [4.0, 0.0]})
        out = composite(assay, 4.0)
        assert out["grade"].iloc[0] == pytest.approx(1.0)  # (4*1 + 0*3) / 4

    def test_composite_length_is_preserved(self):
        assay = pd.DataFrame({"hole_id": ["A"] * 12, "from": np.arange(12.0),
                              "to": np.arange(1.0, 13.0), "grade": 2.0})
        out = composite(assay, 3.0)
        assert out["length"].sum() == pytest.approx(12.0)

    def test_top_cut_caps_and_reports_metal_removed(self):
        df = pd.DataFrame({"hole_id": ["A"] * 5, "from": np.arange(5.0),
                           "to": np.arange(1.0, 6.0), "length": 1.0,
                           "grade": [1.0, 1.0, 1.0, 1.0, 100.0]})
        capped, summary = apply_top_cut(df, 10.0)
        assert capped["grade"].max() == 10.0
        assert summary["n_capped"] == 1
        assert capped["grade_uncapped"].max() == 100.0
        assert summary["metal_removed_pct"] == pytest.approx(90 / 104 * 100, rel=1e-6)


class TestStatistics:
    def test_known_distribution(self):
        df = pd.DataFrame({"grade": [1.0, 2.0, 3.0, 4.0, 5.0]})
        s = compute_grade_stats(df)
        assert s["mean"] == pytest.approx(3.0)
        assert s["median"] == pytest.approx(3.0)
        assert s["std_dev"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
        assert s["geometric_mean"] == pytest.approx(float(np.exp(np.mean(np.log([1, 2, 3, 4, 5])))))

    def test_skewness_sign_follows_the_tail(self):
        right = compute_grade_stats(pd.DataFrame({"grade": [1, 1, 1, 1, 1, 1, 20.0]}))
        assert right["skewness"] > 0

    def test_colour_breaks_are_monotonic(self):
        rng = np.random.default_rng(0)
        s = compute_grade_stats(pd.DataFrame({"grade": rng.lognormal(0, 1, 1000)}))
        ramp = choose_color_breakpoints(s)
        assert ramp["breaks"] == sorted(ramp["breaks"])
        assert len(ramp["colors"]) == len(ramp["breaks"]) - 1


class TestVariography:
    def test_model_reaches_the_sill_at_the_range(self):
        for model in ("spherical", "exponential", "gaussian"):
            at_range = variogram_model(np.array([100.0]), model, 0.0, 1.0, 100.0)[0]
            assert at_range == pytest.approx(1.0, rel=0.06)

    def test_nugget_applies_only_beyond_zero(self):
        assert variogram_model(np.array([0.0]), "spherical", 0.3, 1.0, 50.0)[0] == 0.0
        assert variogram_model(np.array([1.0]), "spherical", 0.3, 1.0, 50.0)[0] > 0.3

    def test_fit_recovers_a_synthetic_variogram(self):
        rng = np.random.default_rng(4)
        pts = rng.uniform(0, 300, size=(400, 3))
        # a smooth field has a real spatial structure to recover
        values = (np.sin(pts[:, 0] / 60) + np.cos(pts[:, 1] / 60)
                  + rng.normal(0, 0.05, len(pts)))
        exp = experimental_variogram(pts, values)
        fit = fit_all_models(exp)["best_fit"]
        assert fit["range"] > 0
        assert fit["r_squared"] > 0.5
        assert fit["nugget"] >= 0


class TestEstimation:
    def test_kriging_reproduces_a_constant_field(self):
        rng = np.random.default_rng(1)
        samples = rng.uniform(0, 100, size=(60, 3))
        values = np.full(60, 5.0)
        est, var, used = ordinary_kriging(np.array([[50.0, 50.0, 50.0]]), samples, values,
                                          "spherical", 0.0, 1.0, 60.0)
        assert est[0] == pytest.approx(5.0, abs=1e-6)

    def test_kriging_weights_sum_to_one_via_unbiasedness(self):
        """A constant field is only reproduced when the weights sum to one."""
        samples = np.array([[0.0, 0, 0], [20.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]])
        values = np.array([7.0, 7.0, 7.0, 7.0])
        est, _, _ = ordinary_kriging(np.array([[5.0, 5.0, 5.0]]), samples, values,
                                     "spherical", 0.1, 1.0, 50.0)
        assert est[0] == pytest.approx(7.0, abs=1e-6)

    def test_idw_favours_the_nearest_sample(self):
        samples = np.array([[0.0, 0, 0], [100.0, 0, 0]])
        values = np.array([10.0, 0.0])
        est, _, _ = inverse_distance(np.array([[1.0, 0.0, 0.0]]), samples, values,
                                     search_radius=500.0, min_samples=1)
        assert est[0] > 9.0

    def test_idw_returns_the_exact_value_at_a_sample(self):
        samples = np.array([[0.0, 0, 0], [100.0, 0, 0]])
        values = np.array([10.0, 0.0])
        est, _, _ = inverse_distance(np.array([[0.0, 0.0, 0.0]]), samples, values,
                                     search_radius=500.0, min_samples=1)
        assert est[0] == pytest.approx(10.0)

    def test_blocks_without_enough_neighbours_stay_unestimated(self):
        samples = np.array([[0.0, 0, 0], [10.0, 0, 0]])
        values = np.array([1.0, 2.0])
        est, _, _ = ordinary_kriging(np.array([[9000.0, 9000.0, 9000.0]]), samples, values,
                                     "spherical", 0.0, 1.0, 50.0, min_samples=4)
        assert not np.isfinite(est[0])
