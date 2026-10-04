"""Geometry tests: desurvey, triangulation, validation, point-in-solid.

These check against analytically known answers rather than against whatever the
code happened to produce, so a regression in the maths cannot pass silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minedesign.geology.desurvey import desurvey_hole, desurvey_minimum_curvature
from minedesign.solid.point_in_solid import point_in_solid
from minedesign.solid.sections import SectionFrame
from minedesign.solid.triangulate import triangulate_solid
from minedesign.solid.validate import _triangles_intersect, validate_solid


@pytest.fixture
def unit_frame() -> SectionFrame:
    return SectionFrame(origin=np.zeros(3), strike_dir=np.array([1.0, 0.0, 0.0]),
                        across_dir=np.array([0.0, 1.0, 0.0]), spacing=50.0,
                        strike_azimuth_deg=90.0)


def cylinder(frame: SectionFrame, radius=50.0, length=200.0, n=16, n_sections=5):
    boundaries = []
    for k, s in enumerate(np.linspace(0, length, n_sections)):
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        local = np.column_stack([np.full(n, s), radius * np.cos(theta), radius * np.sin(theta)])
        boundaries.append({
            "section_id": k, "section_s": float(s),
            "boundary_points": [[float(v) for v in p] for p in frame.to_world(local)],
        })
    return boundaries


class TestDesurvey:
    def test_vertical_hole_goes_straight_down(self):
        trace = desurvey_hole((100.0, 200.0, 1000.0), [0, 50, 100], [0, 0, 0], [-90, -90, -90])
        assert np.allclose(trace["x"], 100.0)
        assert np.allclose(trace["y"], 200.0)
        assert np.allclose(trace["z"], [1000.0, 950.0, 900.0])

    def test_horizontal_hole_due_east(self):
        trace = desurvey_hole((0.0, 0.0, 0.0), [0, 100], [90, 90], [0, 0])
        assert trace["x"].iloc[-1] == pytest.approx(100.0)
        assert trace["y"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
        assert trace["z"].iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_horizontal_hole_due_north(self):
        trace = desurvey_hole((0.0, 0.0, 0.0), [0, 100], [0, 0], [0, 0])
        assert trace["y"].iloc[-1] == pytest.approx(100.0)
        assert trace["x"].iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_45_degree_hole_matches_trigonometry(self):
        trace = desurvey_hole((0.0, 0.0, 0.0), [0, 100], [90, 90], [-45, -45])
        expected = 100.0 / np.sqrt(2)
        assert trace["x"].iloc[-1] == pytest.approx(expected)
        assert trace["z"].iloc[-1] == pytest.approx(-expected)

    def test_curved_hole_is_shorter_than_tangential(self):
        """Minimum curvature must fall inside the straight-line tangential step."""
        trace = desurvey_hole((0.0, 0.0, 0.0), [0, 100], [90, 90], [-90, 0])
        travelled = np.linalg.norm(trace[["x", "y", "z"]].iloc[-1].to_numpy())
        assert travelled < 100.0  # an arc chord is shorter than the measured depth
        assert travelled > 60.0

    def test_collar_extension_when_first_station_is_deep(self):
        trace = desurvey_hole((5.0, 6.0, 7.0), [30], [0], [-90])
        assert trace["depth"].iloc[0] == 0.0
        assert trace[["x", "y", "z"]].iloc[0].tolist() == [5.0, 6.0, 7.0]

    def test_hole_without_survey_is_vertical(self):
        collar = pd.DataFrame([{"hole_id": "A", "x": 0.0, "y": 0.0, "z": 100.0,
                                "max_depth": 50.0}])
        survey = pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"])
        trace = desurvey_minimum_curvature(collar, survey)
        assert trace["z"].min() == pytest.approx(50.0)


class TestTriangulation:
    def test_cylinder_volume_matches_analytic(self, unit_frame):
        solid = triangulate_solid(cylinder(unit_frame), unit_frame, n_ring=64)
        exact = np.pi * 50.0 ** 2 * 200.0
        assert solid.volume() == pytest.approx(exact, rel=0.01)

    def test_cylinder_passes_every_validation_check(self, unit_frame):
        solid = triangulate_solid(cylinder(unit_frame), unit_frame, n_ring=32)
        report = validate_solid(solid)
        assert report.is_valid, report.failures

    def test_needs_at_least_two_sections(self, unit_frame):
        one = cylinder(unit_frame, n_sections=5)[:1]
        with pytest.raises(ValueError, match="at least 2 sections"):
            triangulate_solid(one, unit_frame)

    def test_surface_area_matches_analytic(self, unit_frame):
        solid = triangulate_solid(cylinder(unit_frame), unit_frame, n_ring=64)
        exact = 2 * np.pi * 50.0 * 200.0 + 2 * np.pi * 50.0 ** 2
        assert solid.surface_area() == pytest.approx(exact, rel=0.02)


class TestSelfIntersection:
    def test_crossing_triangles_are_detected(self):
        a = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)
        b = np.array([[3, 3, -5], [3, 3, 5], [8, 8, 5]], float)
        assert _triangles_intersect(a, b)

    def test_coplanar_disjoint_triangles_are_not_intersecting(self):
        """The classic 11-axis SAT fails this case; the in-plane axes fix it."""
        a = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)
        b = np.array([[100, 0, 0], [110, 0, 0], [100, 10, 0]], float)
        assert not _triangles_intersect(a, b)

    def test_separated_triangles_are_not_intersecting(self):
        a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        b = np.array([[0, 0, 50], [1, 0, 50], [0, 1, 50]], float)
        assert not _triangles_intersect(a, b)


class TestPointInSolid:
    def test_centre_is_inside_and_far_point_is_outside(self, unit_frame):
        solid = triangulate_solid(cylinder(unit_frame), unit_frame, n_ring=32)
        pts = np.array([[100.0, 0.0, 0.0], [100.0, 500.0, 0.0], [-500.0, 0.0, 0.0]])
        mask = point_in_solid(pts, solid)
        assert mask[0] and not mask[1] and not mask[2]

    def test_monte_carlo_volume_matches_mesh_volume(self, unit_frame):
        solid = triangulate_solid(cylinder(unit_frame), unit_frame, n_ring=48)
        bbox = solid.bbox
        rng = np.random.default_rng(7)
        pts = rng.uniform(bbox[0], bbox[1], size=(20000, 3))
        mask = point_in_solid(pts, solid)
        mc_volume = mask.mean() * np.prod(bbox[1] - bbox[0])
        assert mc_volume == pytest.approx(solid.volume(), rel=0.03)

    def test_empty_input_returns_empty_mask(self, unit_frame):
        solid = triangulate_solid(cylinder(unit_frame), unit_frame, n_ring=16)
        assert point_in_solid(np.zeros((0, 3)), solid).size == 0
