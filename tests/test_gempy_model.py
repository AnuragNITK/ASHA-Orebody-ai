"""Tests for the GemPy implicit orebody model.

The contact and orientation logic is plain numpy and always runs. The tests that
need the solver are skipped when GemPy is not installed, so the suite stays green
on a machine without the optional dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minedesign.gempy_model.builder import (
    _choose_resolution,
    _grid_extent,
    _identify_ore_unit,
    build_gempy_model,
    gempy_version,
)
from minedesign.gempy_model.contacts import (
    BASE,
    TOP,
    build_contact_set,
    estimate_orientations,
    extract_contacts,
    normal_to_dip_azimuth,
)

HAS_GEMPY = gempy_version() is not None
needs_gempy = pytest.mark.skipif(not HAS_GEMPY, reason="GemPy is not installed")


def vertical_hole(hole_id: str, x: float, y: float, ore_from: float, ore_to: float,
                  length: int = 40) -> pd.DataFrame:
    """One vertical hole on 1 m composites, with a single ore intercept."""
    depths = np.arange(length, dtype=float)
    return pd.DataFrame({
        "hole_id": hole_id, "from": depths, "to": depths + 1.0,
        "x": x, "y": y, "z": 100.0 - depths - 0.5,
        "grade": np.where((depths >= ore_from) & (depths < ore_to), 3.0, 0.05),
    })


def lens_dataset(n: int = 6, spacing: float = 40.0) -> tuple[pd.DataFrame, np.ndarray]:
    """A grid of holes through a gently plunging lens."""
    frames = []
    for i in range(n):
        for j in range(n):
            depth = 8.0 + i * 1.5 + j * 1.0
            frames.append(vertical_hole(f"H{i}{j}", i * spacing, j * spacing,
                                        depth, depth + 10.0))
    composites = pd.concat(frames, ignore_index=True)
    return composites, (composites["grade"] > 1.0).to_numpy()


class TestContactExtraction:
    def test_one_intercept_gives_a_top_and_a_base(self):
        comp = vertical_hole("A", 0.0, 0.0, 10.0, 20.0)
        contacts = extract_contacts(comp, (comp["grade"] > 1.0).to_numpy())
        assert len(contacts) == 2
        assert set(contacts["crossing"]) == {TOP, BASE}

    def test_contact_sits_between_the_bracketing_samples(self):
        comp = vertical_hole("A", 0.0, 0.0, 10.0, 20.0)
        contacts = extract_contacts(comp, (comp["grade"] > 1.0).to_numpy())
        top = contacts[contacts["crossing"] == TOP].iloc[0]
        # last waste sample centre is z=90.5, first ore centre z=89.5
        assert top["z"] == pytest.approx(90.0, abs=1e-6)

    def test_two_separate_intercepts_give_four_contacts(self):
        comp = vertical_hole("A", 0.0, 0.0, 5.0, 10.0)
        grade = comp["grade"].to_numpy().copy()
        grade[20:25] = 3.0                      # a second lens lower down
        comp["grade"] = grade
        contacts = extract_contacts(comp, grade > 1.0)
        assert len(contacts) == 4
        assert (contacts["crossing"] == TOP).sum() == 2
        assert (contacts["crossing"] == BASE).sum() == 2

    def test_a_barren_hole_contributes_nothing(self):
        comp = vertical_hole("A", 0.0, 0.0, 0.0, 0.0)
        assert extract_contacts(comp, np.zeros(len(comp), bool)).empty

    def test_ore_running_to_the_end_of_hole_still_closes(self):
        comp = vertical_hole("A", 0.0, 0.0, 30.0, 60.0)   # ore to end of hole
        contacts = extract_contacts(comp, (comp["grade"] > 1.0).to_numpy())
        assert (contacts["crossing"] == BASE).sum() == 1


class TestOrientations:
    def test_horizontal_plane_reads_as_zero_dip(self):
        dip, _ = normal_to_dip_azimuth(np.array([0.0, 0.0, 1.0]))
        assert dip == pytest.approx(0.0, abs=1e-6)

    def test_vertical_plane_reads_as_ninety_dip(self):
        dip, _ = normal_to_dip_azimuth(np.array([1.0, 0.0, 0.0]))
        assert dip == pytest.approx(90.0, abs=1e-6)

    def test_dip_azimuth_points_down_the_dip(self):
        # a plane dipping east has a normal leaning east
        _, azimuth = normal_to_dip_azimuth(np.array([0.7071, 0.0, 0.7071]))
        assert azimuth == pytest.approx(90.0, abs=1.0)

    def test_normals_reverse_around_a_closed_body(self):
        """The whole model depends on this: a closed body needs both signs."""
        composites, ore = lens_dataset()
        contacts = extract_contacts(composites, ore)
        ore_xyz = composites.loc[ore, ["x", "y", "z"]].to_numpy(float)
        orientations = estimate_orientations(contacts, ore_xyz, n_orientations=14)
        assert not orientations.empty
        assert (orientations["G_z"] > 0).any(), "no normal points up"
        assert (orientations["G_z"] < 0).any(), "no normal points down"

    def test_orientations_point_away_from_the_ore(self):
        composites, ore = lens_dataset()
        contacts = extract_contacts(composites, ore)
        ore_xyz = composites.loc[ore, ["x", "y", "z"]].to_numpy(float)
        orientations = estimate_orientations(contacts, ore_xyz, n_orientations=14)
        centre = ore_xyz.mean(axis=0)
        outward = 0
        for row in orientations.itertuples(index=False):
            direction = np.array([row.x, row.y, row.z]) - centre
            if np.dot([row.G_x, row.G_y, row.G_z], direction) > 0:
                outward += 1
        assert outward >= len(orientations) * 0.7


class TestContactSet:
    def test_usable_when_the_drilling_supports_it(self):
        composites, ore = lens_dataset()
        assert build_contact_set(composites, ore).is_usable

    def test_not_usable_without_ore(self):
        composites, _ = lens_dataset()
        contact_set = build_contact_set(composites, np.zeros(len(composites), bool))
        assert not contact_set.is_usable
        assert contact_set.warnings

    def test_thinning_caps_the_point_count(self):
        composites, ore = lens_dataset(n=8)
        contact_set = build_contact_set(composites, ore, thin=20)
        assert len(contact_set.points) <= 20
        assert any("thinned" in w for w in contact_set.warnings)


class TestGridHelpers:
    def test_extent_encloses_every_contact(self):
        points = pd.DataFrame({"x": [0.0, 100.0], "y": [0.0, 50.0], "z": [-20.0, 10.0]})
        extent = _grid_extent(points)
        assert extent[0] < 0.0 and extent[1] > 100.0
        assert extent[4] < -20.0 and extent[5] > 10.0

    def test_resolution_respects_the_cell_budget(self):
        res = _choose_resolution([0, 400, 0, 400, 0, 200], target_cells=60_000)
        assert np.prod(res) <= 60_000 * 3
        assert all(8 <= v <= 80 for v in res)

    def test_ore_unit_is_the_side_holding_the_intercepts(self):
        # a 4x4x4 grid split in half: unit 2 on top, unit 1 below
        ids = np.ones((4, 4, 4), int)
        ids[:, :, 2:] = 2
        extent = [0.0, 4.0, 0.0, 4.0, 0.0, 4.0]
        ore_xyz = np.array([[2.0, 2.0, 3.5], [1.0, 1.0, 3.2], [3.0, 3.0, 2.6]])
        unit, capture = _identify_ore_unit(ids.ravel(), extent, [4, 4, 4], ore_xyz)
        assert unit == 2
        assert capture == pytest.approx(1.0)

    def test_capture_is_none_without_samples(self):
        ids = np.ones(64, int)
        unit, capture = _identify_ore_unit(ids, [0, 4, 0, 4, 0, 4], [4, 4, 4], None)
        assert unit == 1 and capture is None


class TestBuilderWithoutGemPy:
    def test_unusable_contacts_report_a_reason_rather_than_raising(self):
        composites, _ = lens_dataset()
        contact_set = build_contact_set(composites, np.zeros(len(composites), bool))
        result = build_gempy_model(contact_set)
        assert result.available is False
        assert result.error
        assert result.summary()["available"] is False


@needs_gempy
class TestGemPySolve:
    @pytest.fixture(scope="class")
    def solved(self):
        composites, ore = lens_dataset(n=6, spacing=40.0)
        contact_set = build_contact_set(composites, ore, thin=120)
        ore_xyz = composites.loc[ore, ["x", "y", "z"]].to_numpy(float)
        return build_gempy_model(contact_set, target_cells=25_000,
                                 ore_sample_xyz=ore_xyz), ore_xyz

    def test_model_solves_and_reports_a_volume(self, solved):
        result, _ = solved
        assert result.available, result.error
        assert result.ore_volume_m3 > 0
        assert result.n_ore_cells > 0

    def test_solid_captures_most_of_the_ore(self, solved):
        result, _ = solved
        assert result.ore_capture_fraction is not None
        assert result.ore_capture_fraction > 0.5

    def test_mesh_comes_back_in_world_coordinates(self, solved):
        """GemPy solves in a normalised space; an untransformed mesh sits at 0."""
        result, ore_xyz = solved
        assert result.meshes
        verts = np.asarray(result.meshes[0]["vertices"], float)
        assert verts[:, 0].max() > ore_xyz[:, 0].min()
        assert verts[:, 0].min() < ore_xyz[:, 0].max()
        # the mesh must overlap the drilling, not sit near the origin
        assert abs(verts[:, 2].mean() - ore_xyz[:, 2].mean()) < 100.0

    def test_tonnage_follows_density(self, solved):
        result, _ = solved
        assert result.ore_tonnes == pytest.approx(result.ore_volume_m3 * 2.7, rel=1e-6)

    def test_comparison_is_reported_against_a_wireframe(self, solved):
        result, _ = solved
        compared = build_gempy_model(
            result.contacts, target_cells=25_000,
            wireframe_volume=result.ore_volume_m3)
        assert compared.comparison
        assert compared.comparison["difference_pct"] == pytest.approx(0.0, abs=25.0)
