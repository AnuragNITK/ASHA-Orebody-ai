"""Tests for the WAMEX / GSWA statutory exploration file reader.

The format's traps are the point of these tests: repeated column names, censored
laboratory values, and a multi-element assay file that has no column called
"grade". Each one fails silently if it is not handled, which is worse than an
error because the pipeline goes on to model the wrong number.
"""
from __future__ import annotations

import pandas as pd
import pytest

from minedesign.errors import IngestionError
from minedesign.ingest.column_map import map_columns
from minedesign.ingest.wamex import (
    describe_elements,
    is_wamex,
    numeric_with_detection_limits,
    read_wamex,
    select_grade_column,
)

ASSAY = "\n".join([
    "H0002\tVersion\t4",
    "H0102\tProject_name\tPaterson",
    "H1000\tHole_id\tDepth From\tDepth To\tU\tU\tCu",
    "H1001\t\tmetres\tmetres\tppm\tppb\tppm",
    "H1002\t\t\t\t4AMS\tWACMS\t4AOES",
    "D\tPT001\t0\t1\t1.5\t180\t27",
    "D\tPT001\t1\t2\t<0.5\t210\t31",
    "D\tPT001\t2\t3\t>2000\t\t14",
    "EOF",
])

COLLAR = "\n".join([
    "H0002\tVersion\t4",
    "H1000\tHole_ID\tMGA Easting\tMGA Northing\tElevation\tPre_Collar_Depth\tTotal Hole Depth",
    "H1001\t\tmetres\tmetres\tmetres\tmetres\tmetres",
    "D\tPT001\t402366.3\t7525095\t388.92\t\t207.7",
    "EOF",
])


@pytest.fixture
def assay_file(tmp_path):
    p = tmp_path / "PT_ASS.txt"
    p.write_text(ASSAY, encoding="utf-8")
    return p


@pytest.fixture
def collar_file(tmp_path):
    p = tmp_path / "PT_COLL.txt"
    p.write_text(COLLAR, encoding="utf-8")
    return p


class TestDetectionAndParsing:
    def test_a_wamex_file_is_recognised(self, assay_file):
        assert is_wamex(assay_file)

    def test_a_plain_csv_is_not(self, tmp_path):
        p = tmp_path / "plain.txt"
        p.write_text("hole_id,from,to,grade\nA,0,1,2\n", encoding="utf-8")
        assert not is_wamex(p)

    def test_header_metadata_and_data_rows_are_separated(self, assay_file):
        table = read_wamex(assay_file)
        assert len(table.frame) == 3, "the H and EOF lines must not become data"
        assert table.metadata["Project_name"] == "Paterson"

    def test_eof_marker_is_not_a_row(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text(ASSAY.replace("EOF", "EOF\t\t\t\t\t"), encoding="utf-8")
        assert len(read_wamex(p).frame) == 3

    def test_a_file_without_a_column_line_is_refused(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("H0002\tVersion\t4\nD\tPT001\t0\t1\n", encoding="utf-8")
        with pytest.raises(IngestionError, match="H1000"):
            read_wamex(p)


class TestRepeatedColumnNames:
    """The same element by two methods is two measurements, not one column."""

    def test_repeats_are_named_by_method_not_silently_renumbered(self, assay_file):
        cols = list(read_wamex(assay_file).frame.columns)
        assert "U_4AMS" in cols and "U_WACMS" in cols
        assert "U.1" not in cols, "pandas-style renaming discards the method"

    def test_a_unique_name_is_left_alone(self, assay_file):
        assert "Cu" in read_wamex(assay_file).frame.columns

    def test_units_and_methods_are_kept_per_column(self, assay_file):
        table = read_wamex(assay_file)
        assert table.units["U_4AMS"] == "ppm"
        assert table.units["U_WACMS"] == "ppb"
        assert table.methods["U_4AMS"] == "4AMS"


class TestDetectionLimits:
    def test_below_detection_becomes_half_the_limit(self):
        values, below, above = numeric_with_detection_limits(pd.Series(["<0.5"]))
        assert values.iloc[0] == pytest.approx(0.25)
        assert below == 1 and above == 0

    def test_over_range_becomes_the_limit(self):
        values, below, above = numeric_with_detection_limits(pd.Series([">2000"]))
        assert values.iloc[0] == pytest.approx(2000.0)
        assert above == 1 and below == 0

    def test_plain_numbers_and_blanks_pass_through(self):
        values, _, _ = numeric_with_detection_limits(pd.Series(["1.5", "", None]))
        assert values.iloc[0] == pytest.approx(1.5)
        assert values.isna().sum() == 2

    def test_a_censored_column_stays_numeric(self, assay_file):
        """Without substitution the whole column would be text and every grade NaN."""
        table = read_wamex(assay_file)
        values, below, above = numeric_with_detection_limits(table.frame["U_4AMS"])
        assert values.notna().all()
        assert (below, above) == (1, 1)


class TestGradeSelection:
    def test_a_multi_element_file_refuses_to_guess(self, assay_file):
        table = read_wamex(assay_file)
        with pytest.raises(IngestionError, match="has to be stated"):
            select_grade_column(table, None)

    def test_the_error_lists_the_available_elements(self, assay_file):
        table = read_wamex(assay_file)
        with pytest.raises(IngestionError) as exc:
            select_grade_column(table, None)
        assert "U" in str(exc.value) and "Cu" in str(exc.value)

    def test_total_digest_wins_over_a_partial_leach(self, assay_file):
        """A weak-acid leach recovers a fraction of the metal; it is not the grade."""
        column, unit, note = select_grade_column(read_wamex(assay_file), "U")
        assert column == "U_4AMS"
        assert unit == "ppm"
        assert "partial" in note, "the rejected column must be named, not dropped quietly"

    def test_an_unassayed_element_is_refused(self, assay_file):
        with pytest.raises(IngestionError, match="not assayed"):
            select_grade_column(read_wamex(assay_file), "Au")

    def test_an_exact_column_name_can_be_demanded(self, assay_file):
        column, unit, _ = select_grade_column(read_wamex(assay_file), "U_WACMS")
        assert column == "U_WACMS" and unit == "ppb"

    def test_admin_columns_are_not_grade_candidates(self, assay_file):
        elements = {c["column"] for c in describe_elements(read_wamex(assay_file))}
        assert "Depth From" not in elements and "Hole_id" not in elements


class TestColumnMapping:
    def test_total_hole_depth_beats_pre_collar_depth(self, collar_file):
        """Both contain "depth"; the end of hole is the one that matters."""
        mapping, unresolved = map_columns(list(read_wamex(collar_file).frame.columns), "collar")
        assert not unresolved
        assert mapping["Total Hole Depth"] == "max_depth"
        assert mapping.get("Pre_Collar_Depth") != "max_depth"

    def test_mga_coordinates_map_to_x_and_y(self, collar_file):
        mapping, _ = map_columns(list(read_wamex(collar_file).frame.columns), "collar")
        assert mapping["MGA Easting"] == "x"
        assert mapping["MGA Northing"] == "y"
        assert mapping["Elevation"] == "z"

    def test_true_azimuth_is_preferred_over_magnetic(self):
        columns = ["Hole_ID", "Surveyed Depth", "Dip", "Azimuth_TRUE",
                   "Mag_Azimuth", "AZIMUTHGRIDNAME", "Mag_Dip"]
        mapping, unresolved = map_columns(columns, "survey")
        assert not unresolved
        assert mapping["Azimuth_TRUE"] == "azimuth"
        assert mapping["Surveyed Depth"] == "depth"
        assert mapping["Dip"] == "dip"


class TestLoaderIntegration:
    def test_the_loader_reads_wamex_and_binds_the_stated_element(
            self, assay_file, collar_file, tmp_path):
        from minedesign.ingest.loader import load_and_validate

        survey = tmp_path / "PT_SURV.txt"
        survey.write_text("\n".join([
            "H1000\tHole_ID\tSurveyed Depth\tDip\tAzimuth_TRUE",
            "H1001\t\tmetres\tdegrees\tdegrees",
            "D\tPT001\t0\t-60\t180",
            "D\tPT001\t3\t-60\t180",
            "EOF",
        ]), encoding="utf-8")
        geology = tmp_path / "PT_GEO.txt"
        geology.write_text("\n".join([
            "H1000\tHole_id\tDepth From\tDepth To\tLithology",
            "H1001\t\tmetres\tmetres\t",
            "D\tPT001\t0\t3\tSCH",
            "EOF",
        ]), encoding="utf-8")

        result = load_and_validate(collar_file, survey, assay_file, geology,
                                   allow_llm_fallback=False, grade_element="U")
        assert result.grade_column_source == "U_4AMS"
        assert result.grade_unit == "ppm"
        # 1.5, <0.5 -> 0.25, >2000 -> 2000
        assert sorted(result.assay["grade"].round(2)) == [0.25, 1.5, 2000.0]

    def test_the_loader_refuses_a_multi_element_file_with_no_element_named(
            self, assay_file, collar_file, tmp_path):
        """Left to the synonym dictionary this binds Cu on a uranium project."""
        from minedesign.ingest.loader import load_and_validate

        survey = tmp_path / "s.txt"
        survey.write_text("H1000\tHole_ID\tSurveyed Depth\tDip\tAzimuth_TRUE\n"
                          "D\tPT001\t0\t-60\t180\nEOF", encoding="utf-8")
        geology = tmp_path / "g.txt"
        geology.write_text("H1000\tHole_id\tDepth From\tDepth To\tLithology\n"
                           "D\tPT001\t0\t3\tSCH\nEOF", encoding="utf-8")
        with pytest.raises(IngestionError, match="has to be stated"):
            load_and_validate(collar_file, survey, assay_file, geology,
                              allow_llm_fallback=False)
