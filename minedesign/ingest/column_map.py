"""Fixed column-name mapping for the four Surpac-style source files.

Deterministic step: a synonym dictionary plus normalisation covers the naming
conventions used by Surpac, Datamine, Micromine, Vulcan and most hand-kept CSVs.
Only a column that cannot be resolved here escalates to the local fallback model,
and only for the file that failed.
"""
from __future__ import annotations

import re

# canonical name -> accepted synonyms (all compared after normalisation)
COLLAR_SYNONYMS: dict[str, list[str]] = {
    "hole_id": ["holeid", "bhid", "hole", "dhid", "drillhole", "holenumber", "hole_no", "id", "collar_id"],
    "x": ["x", "east", "easting", "xcollar", "x_collar", "utme", "utm_e", "coord_x"],
    "y": ["y", "north", "northing", "ycollar", "y_collar", "utmn", "utm_n", "coord_y"],
    "z": ["z", "elev", "elevation", "rl", "zcollar", "z_collar", "collar_z", "coord_z"],
    # "totalholedepth" is listed explicitly: a WAMEX collar file also carries
    # "Pre_Collar_Depth", and the loose "depth" substring match would otherwise
    # take the pre-collar rather than the end of hole.
    "max_depth": ["totalholedepth", "holedepth", "maxdepth", "eoh", "endofhole",
                  "totaldepth", "depth", "hole_length", "length", "td"],
}

SURVEY_SYNONYMS: dict[str, list[str]] = {
    "hole_id": ["holeid", "bhid", "hole", "dhid", "drillhole", "id"],
    "depth": ["surveyeddepth", "depth", "at", "md", "measureddepth", "distance",
              "from", "downhole", "dh_depth"],
    # "azimuthtrue" before the bare "azimuth" substring: a WAMEX survey file also
    # carries "Mag_Azimuth" and "AZIMUTHGRIDNAME", and true north is the one wanted.
    "azimuth": ["azimuthtrue", "truazimuth", "azimuth", "azi", "az", "bearing",
                "brg", "dir", "direction"],
    "dip": ["dip", "inclination", "incl", "plunge"],
}

ASSAY_SYNONYMS: dict[str, list[str]] = {
    "hole_id": ["holeid", "bhid", "hole", "dhid", "drillhole", "id"],
    "from": ["from", "fromdepth", "depthfrom", "from_m", "top", "start"],
    "to": ["to", "todepth", "depthto", "to_m", "bottom", "end"],
    "grade": ["grade", "value", "assay", "au", "aug_t", "au_ppm", "au_gpt", "cu", "cu_pct",
              "cupct", "grade_value", "fe", "zn", "ag", "ni", "pb", "element", "result"],
}

GEOLOGY_SYNONYMS: dict[str, list[str]] = {
    "hole_id": ["holeid", "bhid", "hole", "dhid", "drillhole", "id"],
    "from": ["from", "fromdepth", "depthfrom", "top", "start"],
    "to": ["to", "todepth", "depthto", "bottom", "end"],
    "rock_type": ["rocktype", "lith", "lithology", "rock", "geol", "geology", "unit",
                  "rockcode", "litho", "domain", "zone", "formation"],
}

# columns the pipeline can run without
OPTIONAL: dict[str, set[str]] = {
    "collar": {"max_depth"},
    "survey": set(),
    "assay": set(),
    "geology": set(),
}

SCHEMAS: dict[str, dict[str, list[str]]] = {
    "collar": COLLAR_SYNONYMS,
    "survey": SURVEY_SYNONYMS,
    "assay": ASSAY_SYNONYMS,
    "geology": GEOLOGY_SYNONYMS,
}

NUMERIC_FIELDS: dict[str, set[str]] = {
    "collar": {"x", "y", "z", "max_depth"},
    "survey": {"depth", "azimuth", "dip"},
    "assay": {"from", "to", "grade"},
    "geology": {"from", "to"},
}


def normalise(name: str) -> str:
    """Lowercase and strip everything that is not alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def map_columns(columns: list[str], file_kind: str) -> tuple[dict[str, str], list[str]]:
    """Map source columns onto canonical names.

    Returns ``(mapping, unresolved)`` where ``mapping`` is source->canonical and
    ``unresolved`` lists required canonical fields that could not be matched.
    """
    synonyms = SCHEMAS[file_kind]
    norm_to_source: dict[str, str] = {}
    for col in columns:
        norm_to_source.setdefault(normalise(col), col)

    mapping: dict[str, str] = {}
    taken: set[str] = set()

    # pass 1: exact synonym hit
    for canonical, options in synonyms.items():
        for option in options:
            src = norm_to_source.get(normalise(option))
            if src is not None and src not in taken:
                mapping[src] = canonical
                taken.add(src)
                break

    # pass 2: prefix/substring hit for anything still missing
    for canonical, options in synonyms.items():
        if canonical in mapping.values():
            continue
        for norm, src in norm_to_source.items():
            if src in taken:
                continue
            if any(norm.startswith(normalise(o)) or normalise(o) in norm for o in options):
                mapping[src] = canonical
                taken.add(src)
                break

    resolved = set(mapping.values())
    unresolved = [c for c in synonyms if c not in resolved and c not in OPTIONAL[file_kind]]
    return mapping, unresolved
