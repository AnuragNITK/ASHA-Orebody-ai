"""Ingestion, LLM-response handling and full-pipeline integration tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from minedesign.errors import IngestionError, LLMError
from minedesign.ingest.column_map import map_columns
from minedesign.ingest.loader import _pick_sheet, load_and_validate
from minedesign.llm.call1_orebody import _validate_and_repair, deterministic_decision
from minedesign.llm.call2_pit import _validate_economics
from minedesign.llm.client import extract_json

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"
HAS_SAMPLE = (SAMPLE / "collar.csv").exists()
HAS_OPENPYXL = True
try:
    import openpyxl  # noqa: F401
except ImportError:
    HAS_OPENPYXL = False
needs_openpyxl = pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl is not installed")


class TestColumnMapping:
    @pytest.mark.parametrize("columns,kind", [
        (["BHID", "East", "North", "RL", "EOH"], "collar"),
        (["hole_id", "x", "y", "z"], "collar"),
        (["HoleID", "Depth", "Azi", "Dip"], "survey"),
        (["DHID", "at", "bearing", "inclination"], "survey"),
        (["hole", "From", "To", "Au_gpt"], "assay"),
        (["BHID", "depth_from", "depth_to", "Cu_pct"], "assay"),
        (["hole", "from", "to", "Lithology"], "geology"),
        (["BHID", "From", "To", "RockCode"], "geology"),
    ])
    def test_common_conventions_resolve(self, columns, kind):
        _, unresolved = map_columns(columns, kind)
        assert unresolved == []

    def test_unmappable_columns_are_reported(self):
        _, unresolved = map_columns(["alpha", "beta", "gamma"], "collar")
        assert "hole_id" in unresolved

    def test_optional_columns_are_not_required(self):
        _, unresolved = map_columns(["BHID", "East", "North", "RL"], "collar")
        assert unresolved == []  # max_depth is optional


class TestLoader:
    def _write(self, tmp_path, name, frame):
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return str(path)

    def _minimal(self, tmp_path, **overrides):
        frames = {
            "collar.csv": pd.DataFrame({"HoleID": ["A"], "East": [0.0], "North": [0.0],
                                        "RL": [100.0], "MaxDepth": [10.0]}),
            "survey.csv": pd.DataFrame({"HoleID": ["A"], "Depth": [0.0],
                                        "Azimuth": [0.0], "Dip": [-90.0]}),
            "assay.csv": pd.DataFrame({"HoleID": ["A"], "From": [0.0], "To": [1.0],
                                       "Au_gpt": [1.5]}),
            "geology.csv": pd.DataFrame({"HoleID": ["A"], "From": [0.0], "To": [1.0],
                                         "Lithology": ["ORE"]}),
        }
        frames.update(overrides)
        return [self._write(tmp_path, n, f) for n, f in frames.items()]

    def test_minimal_dataset_loads(self, tmp_path):
        result = load_and_validate(*self._minimal(tmp_path), allow_llm_fallback=False)
        assert len(result.collar) == 1
        assert result.assay["grade"].iloc[0] == 1.5

    def test_missing_file_is_reported_clearly(self, tmp_path):
        paths = self._minimal(tmp_path)
        paths[0] = str(tmp_path / "nope.csv")
        with pytest.raises(IngestionError, match="not found"):
            load_and_validate(*paths, allow_llm_fallback=False)

    def test_inverted_intervals_are_flagged_and_removed(self, tmp_path):
        bad = pd.DataFrame({"HoleID": ["A", "A"], "From": [0.0, 5.0], "To": [1.0, 2.0],
                            "Au_gpt": [1.0, 2.0]})
        paths = self._minimal(tmp_path, **{"assay.csv": bad})
        result = load_and_validate(*paths, allow_llm_fallback=False)
        assert len(result.assay) == 1
        assert any("to <= from" in f.message for f in result.flags)

    def test_negative_grades_are_flagged(self, tmp_path):
        bad = pd.DataFrame({"HoleID": ["A", "A"], "From": [0.0, 1.0], "To": [1.0, 2.0],
                            "Au_gpt": [-1.0, 2.0]})
        paths = self._minimal(tmp_path, **{"assay.csv": bad})
        result = load_and_validate(*paths, allow_llm_fallback=False)
        assert any("negative grades" in f.message for f in result.flags)
        assert not np.isfinite(result.assay["grade"].iloc[0])

    def test_orphan_intervals_are_dropped(self, tmp_path):
        orphan = pd.DataFrame({"HoleID": ["A", "GHOST"], "From": [0.0, 0.0],
                               "To": [1.0, 1.0], "Au_gpt": [1.0, 9.0]})
        paths = self._minimal(tmp_path, **{"assay.csv": orphan})
        result = load_and_validate(*paths, allow_llm_fallback=False)
        assert set(result.assay["hole_id"]) == {"A"}
        assert any("unknown hole_id" in f.message for f in result.flags)

    def test_positive_dips_are_normalised_to_down(self, tmp_path):
        survey = pd.DataFrame({"HoleID": ["A"], "Depth": [0.0], "Azimuth": [0.0],
                               "Dip": [60.0]})
        paths = self._minimal(tmp_path, **{"survey.csv": survey})
        result = load_and_validate(*paths, allow_llm_fallback=False)
        assert result.survey["dip"].iloc[0] == -60.0
        assert any("positive-down" in f.message for f in result.flags)

    def test_dataset_with_no_usable_grades_is_rejected(self, tmp_path):
        bad = pd.DataFrame({"HoleID": ["A"], "From": [0.0], "To": [1.0], "Au_gpt": ["x"]})
        paths = self._minimal(tmp_path, **{"assay.csv": bad})
        with pytest.raises(IngestionError, match="no usable numeric grade"):
            load_and_validate(*paths, allow_llm_fallback=False)


@needs_openpyxl
class TestExcelIngestion:
    """Every input table may be Excel instead of CSV, singly or as one workbook."""

    def _frames(self):
        return {
            "collar": pd.DataFrame({"HoleID": ["A"], "East": [0.0], "North": [0.0],
                                    "RL": [100.0], "MaxDepth": [10.0]}),
            "survey": pd.DataFrame({"HoleID": ["A"], "Depth": [0.0],
                                    "Azimuth": [0.0], "Dip": [-90.0]}),
            "assay": pd.DataFrame({"HoleID": ["A"], "From": [0.0], "To": [1.0],
                                   "Au_gpt": [1.5]}),
            "geology": pd.DataFrame({"HoleID": ["A"], "From": [0.0], "To": [1.0],
                                     "Lithology": ["ORE"]}),
        }

    def _workbook(self, tmp_path, sheet_names=None):
        frames = self._frames()
        sheet_names = sheet_names or {k: k for k in frames}
        path = tmp_path / "deposit.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for kind, df in frames.items():
                df.to_excel(writer, sheet_name=sheet_names[kind], index=False)
        return path

    def test_sheet_matched_by_name(self):
        sheets = ["Collar Data", "Survey", "Assay Results", "geology"]
        assert _pick_sheet(sheets, "collar") == "Collar Data"
        assert _pick_sheet(sheets, "survey") == "Survey"
        assert _pick_sheet(sheets, "assay") == "Assay Results"
        assert _pick_sheet(sheets, "geology") == "geology"

    def test_no_name_match_falls_back_to_first_sheet(self):
        assert _pick_sheet(["Sheet1", "Sheet2"], "collar") == 0

    def test_one_workbook_loads_all_four_tables(self, tmp_path):
        path = self._workbook(tmp_path)
        result = load_and_validate(str(path), str(path), str(path), str(path),
                                   allow_llm_fallback=False)
        assert len(result.collar) == 1
        assert result.assay["grade"].iloc[0] == 1.5
        assert result.geology["rock_type"].iloc[0] == "ORE"

    def test_workbook_with_unnamed_sheets_still_loads_by_position(self, tmp_path):
        """A sheet not named after its kind still loads, via the first-sheet fallback."""
        frames = self._frames()
        paths = []
        for kind, df in frames.items():
            p = tmp_path / f"{kind}.xlsx"
            df.to_excel(p, sheet_name="Data", index=False)
            paths.append(str(p))
        result = load_and_validate(*paths, allow_llm_fallback=False)
        assert len(result.collar) == 1
        assert result.assay["grade"].iloc[0] == 1.5

    def test_mixed_csv_and_excel_sources(self, tmp_path):
        frames = self._frames()
        collar_csv = tmp_path / "collar.csv"
        frames["collar"].to_csv(collar_csv, index=False)
        survey_xlsx = tmp_path / "survey.xlsx"
        frames["survey"].to_excel(survey_xlsx, sheet_name="Data", index=False)
        assay_csv = tmp_path / "assay.csv"
        frames["assay"].to_csv(assay_csv, index=False)
        geology_xlsx = tmp_path / "geology.xlsx"
        frames["geology"].to_excel(geology_xlsx, sheet_name="Data", index=False)

        result = load_and_validate(str(collar_csv), str(survey_xlsx),
                                   str(assay_csv), str(geology_xlsx),
                                   allow_llm_fallback=False)
        assert len(result.collar) == 1
        assert result.assay["grade"].iloc[0] == 1.5

    def test_excel_and_csv_give_identical_data(self, tmp_path):
        frames = self._frames()
        csv_paths = []
        for kind, df in frames.items():
            p = tmp_path / f"{kind}.csv"
            df.to_csv(p, index=False)
            csv_paths.append(str(p))
        xlsx_path = self._workbook(tmp_path)

        from_csv = load_and_validate(*csv_paths, allow_llm_fallback=False)
        from_xlsx = load_and_validate(str(xlsx_path), str(xlsx_path),
                                      str(xlsx_path), str(xlsx_path),
                                      allow_llm_fallback=False)
        # values must match; dtype need not -- pandas sniffs a whole-number CSV
        # cell as int64 and an Excel float cell as float64, both numerically 0.0,
        # and every numeric column is coerced with to_numpy(float) downstream
        assert np.array_equal(from_csv.assay["grade"].to_numpy(float),
                              from_xlsx.assay["grade"].to_numpy(float))
        assert np.array_equal(from_csv.collar["x"].to_numpy(float),
                              from_xlsx.collar["x"].to_numpy(float))

    def test_missing_excel_file_reports_clearly(self, tmp_path):
        with pytest.raises(IngestionError, match="not found"):
            load_and_validate(str(tmp_path / "nope.xlsx"), str(tmp_path / "nope.xlsx"),
                              str(tmp_path / "nope.xlsx"), str(tmp_path / "nope.xlsx"),
                              allow_llm_fallback=False)

    def test_empty_workbook_sheet_is_rejected(self, tmp_path):
        path = tmp_path / "empty.xlsx"
        pd.DataFrame({"HoleID": [], "East": [], "North": [], "RL": []}).to_excel(
            path, sheet_name="collar", index=False)
        with pytest.raises(IngestionError, match="no rows"):
            load_and_validate(str(path), str(path), str(path), str(path),
                              allow_llm_fallback=False)


class TestJSONExtraction:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}

    def test_json_with_surrounding_prose(self):
        assert extract_json('Here you go:\n{"a": 3}\nHope that helps.') == {"a": 3}

    def test_braces_inside_strings_do_not_confuse_the_parser(self):
        text = 'prefix {"note": "a } brace", "v": 4} suffix'
        assert extract_json(text) == {"note": "a } brace", "v": 4}

    def test_empty_response_raises(self):
        with pytest.raises(LLMError):
            extract_json("   ")

    def test_unparseable_response_raises(self):
        with pytest.raises(LLMError):
            extract_json("no json at all here")


@pytest.fixture
def stats_fixture():
    rng = np.random.default_rng(0)
    return compute_stats(rng.lognormal(-1, 1, 2000))


def compute_stats(values):
    from minedesign.geology.stats import compute_grade_stats
    return compute_grade_stats(pd.DataFrame({"grade": values}))


class TestCall1Validation:
    def _fallback(self, stats):
        rock_stats = {"ORE": {"mean": 2.0, "n": 10}, "WASTE": {"mean": 0.01, "n": 90}}
        variography = {"omnidirectional": {"best_fit": {
            "model": "spherical", "range": 80.0, "sill": 1.0, "nugget": 0.2,
            "nugget_effect_ratio": 0.2}}}
        density = {"median_nearest_neighbour_m": 20.0, "n_samples": 500}
        return deterministic_decision(stats, rock_stats, variography, density), rock_stats

    def test_deterministic_decision_is_complete(self, stats_fixture):
        decision, _ = self._fallback(stats_fixture)
        for key in ("cutoff_grade", "cutoff_rationale", "ore_rock_types",
                    "variogram_model", "variogram_params", "estimation_method"):
            assert key in decision
        assert decision["estimation_method"] in ("ordinary_kriging", "idw")

    def test_sparse_data_falls_back_to_idw(self, stats_fixture):
        variography = {"omnidirectional": {"best_fit": {
            "model": "spherical", "range": 10.0, "sill": 1.0, "nugget": 0.9,
            "nugget_effect_ratio": 0.9}}}
        decision = deterministic_decision(
            stats_fixture, {"ORE": {"mean": 1.0}}, variography,
            {"median_nearest_neighbour_m": 100.0, "n_samples": 20})
        assert decision["estimation_method"] == "idw"

    def test_out_of_range_cutoff_is_replaced(self, stats_fixture):
        fallback, rock_stats = self._fallback(stats_fixture)
        bad = dict(fallback, cutoff_grade=1e9)
        repaired, notes = _validate_and_repair(bad, stats_fixture, rock_stats, fallback)
        assert repaired["cutoff_grade"] == fallback["cutoff_grade"]
        assert any("outside the observed" in n for n in notes)

    def test_invented_rock_codes_are_rejected(self, stats_fixture):
        fallback, rock_stats = self._fallback(stats_fixture)
        bad = dict(fallback, ore_rock_types=["UNOBTAINIUM"])
        repaired, notes = _validate_and_repair(bad, stats_fixture, rock_stats, fallback)
        assert repaired["ore_rock_types"] == fallback["ore_rock_types"]

    def test_unknown_variogram_model_is_replaced(self, stats_fixture):
        fallback, rock_stats = self._fallback(stats_fixture)
        bad = dict(fallback, variogram_model="magic")
        repaired, notes = _validate_and_repair(bad, stats_fixture, rock_stats, fallback)
        assert repaired["variogram_model"] in ("spherical", "exponential", "gaussian")

    def test_negative_variogram_parameters_are_replaced(self, stats_fixture):
        fallback, rock_stats = self._fallback(stats_fixture)
        bad = dict(fallback, variogram_params={"range": -5, "sill": -1, "nugget": -2})
        repaired, _ = _validate_and_repair(bad, stats_fixture, rock_stats, fallback)
        assert all(v >= 0 for v in repaired["variogram_params"].values())
        assert repaired["variogram_params"]["range"] > 0


class TestCall2Validation:
    def test_absurd_values_are_replaced_with_defaults(self):
        econ, notes = _validate_economics(
            {"commodity_price": -50, "recovery_pct": 500,
             "overall_slope_angle_deg": 89}, None)
        assert econ["commodity_price"] > 0
        assert 0 < econ["recovery_pct"] <= 100
        assert 10 <= econ["overall_slope_angle_deg"] <= 85
        assert len(notes) == 3

    def test_user_values_override_the_model(self):
        econ, notes = _validate_economics(
            {"commodity_price": 1500.0}, {"commodity_price": 2500.0})
        assert econ["commodity_price"] == 2500.0
        assert any("user input" in n for n in notes)

    def test_sector_slopes_are_range_checked(self):
        econ, _ = _validate_economics(
            {"slope_angles_by_sector": {"north": 50.0, "south": 900.0}}, None)
        assert econ["slope_angles_by_sector"] == {"north": 50.0}


@pytest.mark.skipif(not HAS_SAMPLE, reason="sample data not generated")
class TestPipelineIntegration:
    def test_offline_run_produces_a_coherent_report(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINEDESIGN_OFFLINE", "1")
        from minedesign import config as config_module
        monkeypatch.setattr(config_module.SETTINGS, "offline", True)

        from minedesign.pipeline import PipelineConfig, run_pipeline

        out = tmp_path / "report.html"
        result = run_pipeline(
            *[SAMPLE / f"{n}.csv" for n in ("collar", "survey", "assay", "geology")],
            config=PipelineConfig(project_name="Test Deposit", block_size=25.0,
                                  run_cross_validation=False, output_path=out))

        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "Test Deposit" in html
        assert "THREE" in html          # three.js is inlined
        assert "Chart" in html          # chart.js is inlined
        assert 'id="payload"' in html

        s = result.summary
        # the pit can only mine what the resource contains
        assert s["pit_ore_tonnes"] <= s["resource_tonnes"] * 1.001
        assert s["pit_strip_ratio"] >= 0
        assert s["cutoff_grade"] > 0
        assert s["n_blocks_ore"] <= s["n_blocks_total"]
        assert result.summary["pit_contained_oz"] >= 0

    def test_offline_run_makes_no_paid_calls(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINEDESIGN_OFFLINE", "1")
        from minedesign import config as config_module
        monkeypatch.setattr(config_module.SETTINGS, "offline", True)

        from minedesign.pipeline import PipelineConfig, run_pipeline

        result = run_pipeline(
            *[SAMPLE / f"{n}.csv" for n in ("collar", "survey", "assay", "geology")],
            config=PipelineConfig(project_name="No Spend", block_size=30.0,
                                  run_cross_validation=False,
                                  output_path=tmp_path / "r.html"))
        assert not any(c["provider"] == "hf" and c["ok"] for c in result.call_log)
