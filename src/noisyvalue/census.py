"""Interface to the 2020 Census Noisy Measurement Files (NMF).

The NMF is the raw output of the Census Bureau's TopDown disclosure-avoidance
algorithm: for every geography on the DAS geographic spine, a set of
independently noised queries (discrete Gaussian mechanism) over the PL94
person and housing-unit histograms, plus the exact constraints the Bureau's
own post-processing used.  `get_pl94` reads the Hive-partitioned parquet
release and returns a tidy pandas DataFrame whose `value` column holds
`NoisyInt` posteriors.

`get_dhc` does the same for the 2020 Demographic and Housing Characteristics
person NMF (DHC-P), whose histogram is
``relgq(42) x sex(2) x age(116) x hispanic(2) x cenrace(63)``.  That product
is where sex and age live: PL94 has neither.  Its parquet release is public
but not bundled here; `fetch_dhc` downloads the partitions you ask for into a
local tree with the same layout `get_dhc` reads.

Because every DP query is noised independently, each cell's posterior under a
flat prior is a discrete Gaussian centered at the observed count.  Constraint
rows are not exposed as data; instead they sharpen the posteriors:

- ``total_con`` (housing units, every level; persons, national level): the
  occupied/vacant pair is conditioned on the exact unit total, leaving the
  two cells perfectly negatively correlated; the exact person total is
  returned as a point mass when the ``total`` query is requested nationally.
- ``nurse_nva_0_con``: structural zero -- detailed cells for under-18
  residents of nursing facilities are pinned to exactly zero.
- ``hhgq_total_lb_con`` / ``hhgq_total_ub_con``: interval bounds on the hhgq
  (PL94) or per-facility-type GQ (DHC) category totals, applied by truncating
  the affected posteriors.
- ``pl94_con`` (DHC): the published PL94 table for the geography, exact,
  because it was an invariant of the DHC run.  The DHC histogram
  marginalizes onto it, so each query's cells split into blocks of known
  total; see ``_dhc_pl94_blocks`` for what each block size buys.
- ``relgqN_..._con`` (DHC): structural zeros pinning ages a relationship or
  facility type cannot have -- no 14-year-old householders, nobody over 65 in
  college housing.
- All counts are additionally truncated at zero when ``nonnegative=True``.

Note that the NMF is *pre* post-processing: noisy measurements do not add up
across geography levels or across queries, and they differ from the published
PL94 tables.  That inconsistency is real information about the noise and is
exactly what the returned posteriors quantify.

Conditioning a DHC query on ``pl94_con`` does a great deal of work, because
PL94 resolves every DHC axis but sex: male + female is pinned within each
hispanic level, and in a small county most of the detailed histogram becomes
point masses at zero (PL94 says the county holds nobody in that category at
all).  What it cannot pin is a block of three or more free cells, where the
conditional posterior is a discrete Gaussian on a lattice hyperplane rather
than anything the node graph samples cell by cell; those keep the block total
only as an upper bound.

Geocodes are DAS spine codes, not census GEOIDs.  The spine splits every
state into a non-AIAN portion (prefix ``0``) and, where present, an AIAN
portion (prefix ``1``); both are returned, flagged by the ``aian`` column,
and rows sharing a ``geoid`` can be summed (NoisyInt arithmetic) to get
whole-geography estimates.  Standard GEOIDs are derivable for every level
except block groups, whose spine version ("optimized block groups") does not
correspond to tabulation block groups; pass ``real_block_groups=True`` to
``get_pl94`` to recover real block-group totals instead, by reading at the
block level (where GEOIDs are exact) and summing up to each block's real
block group.  ``get_dhc`` covers the levels whose spine codes carry exact
GEOIDs -- us, state, and county; the DHC spine's lower levels (``Prim``,
``Tract_Subset``, ``Tract_Subset_Group``) are spine constructions with no
tabulation counterpart.
"""

import itertools
import math
import os
import re
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import polars as pl
import sympy as sp

from .core import NoisyInt
from .graph import DerivedNode
from .graph import DiscreteGaussianNode
from .graph import TruncatedDiscreteGaussianNode
from .pandas_ext import NoisyFloatArray
from .pandas_ext import NoisyIntArray

DEFAULT_ROOT = "data/2020-pl94-nmf-parquets"
DEFAULT_DHC_ROOT = "data/2020-dhc-nmf-parquets"

# Public Census release the DHC noisy measurements are mirrored from.
DHC_S3_BASE = "https://uscb-2020-product-releases.s3.us-west-2.amazonaws.com"
DHC_S3_PREFIX = "decennial/dhc/2020/nmf/2020-dhc-nmf"

# ── codebook ─────────────────────────────────────────────────────────────────
# Level orderings follow the 2020 DAS schema definitions and were validated
# against published 2020 PL94 tabulations (test/test_census.py); e.g. the six
# single-race national counts each land within a couple standard deviations
# of their published values only under this ordering.

RACES = ("white", "black", "aian", "asian", "nhopi", "sor")

CENRACE_LEVELS = tuple(
    "-".join(combo)
    for k in range(1, 7)
    for combo in itertools.combinations(RACES, k)
)

HHGQ_LEVELS = (
    "household",
    "correctional",
    "juvenile",
    "nursing",
    "other_institutional",
    "college_housing",
    "military",
    "other_noninstitutional",
)

HHINSTLEVELS_LEVELS = ("household", "institutional", "noninstitutional")
VOTINGAGE_LEVELS = ("under_18", "voting_age")
HISPANIC_LEVELS = ("not_hispanic", "hispanic")
H1_LEVELS = ("vacant", "occupied")

# ── DHC-P codebook ───────────────────────────────────────────────────────────
# The DHC person histogram is relgq x sex x age x hispanic x cenrace; hispanic
# and cenrace reuse the PL94 orderings.  Every ordering below was recovered
# from the national NMF itself and is re-checked in test/test_census.py: the
# coarse queries are marginals of the same histogram `detailed_dpq` measures,
# so aggregating detailed_dpq under a candidate ordering has to reproduce them
# to within the measurement noise.

SEX_LEVELS = ("male", "female")

# relgq 0-17: household relationship (2020 RELSHIP, with the householder split
# by whether they live alone).  relgq 18-41: the 24 major GQ types, in the
# order 101-106, 201-203, 301, 401-405, 501, 601-602, 701, 801-802, 900, 901,
# 997.  The DAS age constraints confirm the ordering: relgq 0-5 are 15+ (a
# householder or spouse/partner cannot be a child), relgq 6-8 are at most 89
# (a householder's child), 10 and 12 are 30+ (parents and parents-in-law), 11
# is at most 74 (grandchild), 16 is at most 20 (foster child), 18-22 are 15+
# (adult correctional), 24-26 are at most 25 (juvenile facilities), 32 is 3-30
# (residential schools), 33 is 16-65 (college housing), 23/31/34/35 are 17-65
# (military), 37-38 are 16+ (adult group homes), 39 is 16-75 (merchant ships).
RELSHIP_LEVELS = (
    "householder_alone",
    "householder_not_alone",
    "opposite_sex_spouse",
    "opposite_sex_partner",
    "same_sex_spouse",
    "same_sex_partner",
    "biological_child",
    "adopted_child",
    "stepchild",
    "sibling",
    "parent",
    "grandchild",
    "parent_in_law",
    "child_in_law",
    "other_relative",
    "roommate",
    "foster_child",
    "other_nonrelative",
)

GQ_TYPE_LEVELS = (
    "federal_detention_center",
    "federal_prison",
    "state_prison",
    "local_jail",
    "correctional_residential_facility",
    "military_disciplinary_barracks",
    "juvenile_group_home",
    "juvenile_residential_treatment",
    "juvenile_correctional_facility",
    "nursing_facility",
    "mental_hospital",
    "hospital_no_usual_home",
    "hospice",
    "military_treatment_facility",
    "residential_school",
    "college_housing",
    "military_quarters",
    "military_ships",
    "shelter",
    "adult_group_home",
    "adult_residential_treatment",
    "maritime_vessel",
    "workers_quarters",
    "other_noninstitutional_facility",
)

RELGQ_LEVELS = RELSHIP_LEVELS + GQ_TYPE_LEVELS

# popSehsdTargetsRelship collapses all 24 GQ types into one level; the
# 8-level-GQ variant collapses them into the 7 PL94 major types instead.
POPSEHSD_RELSHIP_LEVELS = RELSHIP_LEVELS + ("group_quarters",)
RELSHIP_GQ8_LEVELS = RELSHIP_LEVELS + HHGQ_LEVELS[1:]

RELGQ_4_LEVELS = (
    "householder",
    "other_household_member",
    "institutional_gq",
    "noninstitutional_gq",
)

# The six groups the DAS measures against age_10_groups; each collects GQ
# types that share an age restriction (see the relgq comment above).
GQ_CONSTR_LEVELS = (
    "household",
    "residential_school",
    "adult_correctional",
    "college_and_other_noninstitutional",
    "military",
    "other_gq",
)

# Membership of each coarse relgq grouping, as index groups over RELGQ_LEVELS.
_RELGQ_GROUPS = {
    "relgq": tuple((i,) for i in range(42)),
    "relgq_4_groups": (
        (0, 1),
        tuple(range(2, 18)),
        tuple(range(18, 33)),
        tuple(range(33, 42)),
    ),
    "popSehsdTargetsRelship": (
        *((i,) for i in range(18)),
        tuple(range(18, 42)),
    ),
    "relship_and_eight_level_GQ": (
        *((i,) for i in range(18)),
        tuple(range(18, 24)),   # correctional
        (24, 25, 26),           # juvenile
        (27,),                  # nursing
        tuple(range(28, 33)),   # other institutional
        (33,),                  # college housing
        (34, 35),               # military
        tuple(range(36, 42)),   # other noninstitutional
    ),
    "gq_constr_groups": (
        tuple(range(18)),
        (32,),
        tuple(range(18, 23)),
        (33, 37, 38, 39, 40, 41),
        (23, 31, 34, 35),
        (24, 25, 26, 27, 28, 29, 30, 36),
    ),
}


def _age_labels(ranges):
    return tuple(f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in ranges)


AGE_RANGES = tuple((a, a) for a in range(116))
AGE_3_RANGES = ((0, 17), (18, 64), (65, 115))
AGE_10_RANGES = (
    (0, 2), (3, 14), (15, 15), (16, 16), (17, 17),
    (18, 19), (20, 24), (25, 29), (30, 34), (35, 115),
)
_AGE_TAIL = (
    (18, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49),
    (50, 54), (55, 59), (60, 61), (62, 64), (65, 66), (67, 69), (70, 74),
    (75, 79), (80, 84), (85, 89), (90, 94), (95, 99), (100, 104),
    (105, 109), (110, 115),
)
AGE_38_RANGES = tuple((a, a) for a in range(15)) + ((15, 17),) + _AGE_TAIL
AGE_40_RANGES = tuple((a, a) for a in range(18)) + _AGE_TAIL

AGE_LEVELS = _age_labels(AGE_RANGES)
AGE_3_LEVELS = _age_labels(AGE_3_RANGES)
AGE_10_LEVELS = _age_labels(AGE_10_RANGES)
AGE_38_LEVELS = _age_labels(AGE_38_RANGES)
AGE_40_LEVELS = _age_labels(AGE_40_RANGES)


# ── DHC-P constraint tables ──────────────────────────────────────────────────
# What the three families of DHC constraint row mean, as index sets over the
# histogram axes.  All three were checked against the data (test_census.py):
# the ages a relgq rule forbids hold nothing but noise, the gqlevels bounds
# name exactly the facility types a county has, and pl94_con reproduces the
# published PL94 table for the geography.

def _singles(n):
    return tuple((i,) for i in range(n))


def _range_groups(ranges):
    return tuple(tuple(range(lo, hi + 1)) for lo, hi in ranges)


# Index groups over each axis, per axis spec: the base cells every position of
# a query's axis covers.  Parallel to _DHC_AXIS_LEVELS, which labels them.
_DHC_AXIS_GROUPS = {
    "relgq": {
        "*": (tuple(range(42)),),
        "detailed": _singles(42),
        **_RELGQ_GROUPS,
    },
    "sex": {"*": ((0, 1),), "detailed": _singles(2), "sex": _singles(2)},
    "age": {
        "*": (tuple(range(116)),),
        "detailed": _singles(116),
        "age_18_64_116": _range_groups(AGE_3_RANGES),
        "age_10_groups": _range_groups(AGE_10_RANGES),
        "age_38_groups": _range_groups(AGE_38_RANGES),
        "age_40_groups": _range_groups(AGE_40_RANGES),
    },
    "hispanic": {"*": ((0, 1),), "detailed": _singles(2), "hispanic": _singles(2)},
    "cenrace": {
        "*": (tuple(range(63)),),
        "detailed": _singles(63),
        "cenrace": _singles(63),
    },
}

# Ages each `relgqN_..._con` row pins to zero, keyed by its age spec string.
_AGE_ZERO_RULES = {
    "age_lessthan15": frozenset(range(0, 15)),
    "age_lessthan20": frozenset(range(0, 20)),
    "age_lessthan30": frozenset(range(0, 30)),
    "age_lt16": frozenset(range(0, 16)),
    "age_greaterthan25": frozenset(range(26, 116)),
    "age_greaterthan89": frozenset(range(90, 116)),
    "age_gt20": frozenset(range(21, 116)),
    "age_gt74": frozenset(range(75, 116)),
    "age_lt15_gt89": frozenset(range(0, 15)) | frozenset(range(90, 116)),
    "age_lt16_gt75": frozenset(range(0, 16)) | frozenset(range(76, 116)),
    "age_lt16gt65": frozenset(range(0, 16)) | frozenset(range(66, 116)),
    "age_lt17gt65": frozenset(range(0, 17)) | frozenset(range(66, 116)),
    "age_lt3_gt30": frozenset(range(0, 3)) | frozenset(range(31, 116)),
}


def _relgq_to_hhgq8():
    """relgq level -> its PL94 hhgq category, the coarsening pl94_con uses."""
    out = [0] * 42                                  # relgq 0-17 are households
    gq_classes = _RELGQ_GROUPS["relship_and_eight_level_GQ"][18:]
    for category, group in enumerate(gq_classes, start=1):
        for level in group:
            out[level] = category
    return tuple(out)


# pl94_con is an 8 x 2 x 2 x 63 histogram over hhgq x votingage x hispanic x
# cenrace, so it resolves every DHC axis except sex.
_PL94_SHAPE = (8, 2, 2, 63)
_PL94_AXIS_OF = {"relgq": 0, "age": 1, "hispanic": 2, "cenrace": 3}
_PL94_CLASS_OF = {
    "relgq": _relgq_to_hhgq8(),
    "age": tuple(0 if age < 18 else 1 for age in range(116)),
    "hispanic": (0, 1),
    "cenrace": tuple(range(63)),
}

# The gqlevels axis the hhgq bound rows use: one household level, then the 24
# GQ types in relgq order.
_GQLEVEL_MEMBERS = (tuple(range(18)),) + tuple((i,) for i in range(18, 42))

# Per-axis label sets keyed by the spec strings that appear in the NMF's
# hhgq/votingage/hispanic/cenrace/h1 columns.
_PL94_AXIS_LEVELS = {
    "hhgq": {
        "*": ("total",),
        "detailed": HHGQ_LEVELS,
        "hhgq": HHGQ_LEVELS,
        "hhinstlevels": HHINSTLEVELS_LEVELS,
        "gqNursingTotal": ("nursing",),
    },
    "votingage": {
        "*": ("total",),
        "detailed": VOTINGAGE_LEVELS,
        "votingage": VOTINGAGE_LEVELS,
        "nonvoting": ("under_18",),
    },
    "hispanic": {
        "*": ("total",),
        "detailed": HISPANIC_LEVELS,
        "hispanic": HISPANIC_LEVELS,
    },
    "cenrace": {
        "*": ("total",),
        "detailed": CENRACE_LEVELS,
        "cenrace": CENRACE_LEVELS,
    },
    "h1": {
        "*": ("total",),
        "detailed": H1_LEVELS,
    },
}

_DHC_AXIS_LEVELS = {
    "relgq": {
        "*": ("total",),
        "detailed": RELGQ_LEVELS,
        "relgq": RELGQ_LEVELS,
        "relgq_4_groups": RELGQ_4_LEVELS,
        "popSehsdTargetsRelship": POPSEHSD_RELSHIP_LEVELS,
        "relship_and_eight_level_GQ": RELSHIP_GQ8_LEVELS,
        "gq_constr_groups": GQ_CONSTR_LEVELS,
    },
    "sex": {
        "*": ("total",),
        "detailed": SEX_LEVELS,
        "sex": SEX_LEVELS,
    },
    "age": {
        "*": ("total",),
        "detailed": AGE_LEVELS,
        "age_18_64_116": AGE_3_LEVELS,
        "age_10_groups": AGE_10_LEVELS,
        "age_38_groups": AGE_38_LEVELS,
        "age_40_groups": AGE_40_LEVELS,
    },
    "hispanic": {
        "*": ("total",),
        "detailed": HISPANIC_LEVELS,
        "hispanic": HISPANIC_LEVELS,
    },
    "cenrace": {
        "*": ("total",),
        "detailed": CENRACE_LEVELS,
        "cenrace": CENRACE_LEVELS,
    },
}

# Index groups over the 8 base hhgq levels, used to translate the hhgq bound
# constraints onto each query's hhgq axis.
_HHGQ_GROUPS = {
    "detailed": tuple((i,) for i in range(8)),
    "hhgq": tuple((i,) for i in range(8)),
    "hhinstlevels": ((0,), (1, 2, 3, 4), (5, 6, 7)),
    "gqNursingTotal": ((3,),),
}

_PERSON_AXES = ("hhgq", "votingage", "hispanic", "cenrace")
_UNIT_AXES = ("h1",)
_DHC_AXES = ("relgq", "sex", "age", "hispanic", "cenrace")

_PERSON_QUERIES = {
    "total": "total_dpq",
    "hhgq": "hhgq_dpq",
    "hhinstlevels": "hhinstlevels_dpq",
    "votingage": "votingage_dpq",
    "hispanic": "hispanic_dpq",
    "cenrace": "cenrace_dpq",
    "votingage*hispanic": "votingage * hispanic_dpq",
    "votingage*cenrace": "votingage * cenrace_dpq",
    "hispanic*cenrace": "hispanic * cenrace_dpq",
    "votingage*hispanic*cenrace": "votingage * hispanic * cenrace_dpq",
    "detailed": "detailed_dpq",
}

# The unit histogram has a single axis; its only DP query is occupied/vacant.
# The unit total is invariant (exact) at every level, so "total" is served
# from the constraint rows as a point mass.
_UNIT_QUERIES = {
    "total": None,
    "h1": "detailed_dpq",
}

# DHC-P query names list their non-marginalized axis specs in histogram axis
# order, so the canonical name says exactly which binning each axis uses; the
# raw NMF names order the axes differently and are matched verbatim.
_DHC_QUERIES = {
    "sex*hispanic": "hispanic * sex_dpq",
    "sex*age_18_64_116": "age_18_64_116 * sex_dpq",
    "sex*age_38_groups": "age_38_groups * sex_dpq",
    "relgq_4_groups*sex": "sex * relgq_4_groups_dpq",
    "relgq_4_groups*age_18_64_116": "age_18_64_116 * relgq_4_groups_dpq",
    "gq_constr_groups*age_10_groups": "gq_constr_groups * age_10_groups_dpq",
    "popSehsdTargetsRelship": "popSehsdTargetsRelship_dpq",
    "relgq*sex*age_40_groups*hispanic*cenrace":
        "relgq * age_40_groups * hispanic * cenrace * sex_dpq",
    "relship_and_eight_level_GQ*sex*age_40_groups*hispanic*cenrace":
        "hispanic * sex * age_40_groups * relship_and_eight_level_GQ * cenrace_dpq",
    "detailed": "detailed_dpq",
}

_PERSON_CONSTRAINTS = (
    "total_con",
    "nurse_nva_0_con",
    "hhgq_total_lb_con",
    "hhgq_total_ub_con",
)
_UNIT_CONSTRAINTS = ("total_con",)

_PL94_GEO_LEVELS = {
    "us": "US",
    "state": "State",
    "county": "County",
    "tract": "Tract",
    "block group": "Block_Group",
    "block_group": "Block_Group",
    "block": "Block",
}

_DHC_GEO_LEVELS = {
    "us": "US",
    "state": "State",
    "county": "County",
}

# One spec per (product, table): where the parquet lives, what the histogram
# axes are, which queries it serves, and how its cells are built.  `rules`
# names the constraint handling `_emit_cells` applies, and `constraints` the
# rows it needs; DHC does neither yet.
_PL94_PERSON = {
    "dataset": "{}_Person_PL_PROD",
    "axes": _PERSON_AXES,
    "levels": _PL94_AXIS_LEVELS,
    "queries": _PERSON_QUERIES,
    "constraints": _PERSON_CONSTRAINTS,
    "geographies": _PL94_GEO_LEVELS,
    "exact_total_levels": ("US",),
    "rules": "pl94_person",
}

_PL94_UNIT = {
    "dataset": "{}_Unit_PL_PROD",
    "axes": _UNIT_AXES,
    "levels": _PL94_AXIS_LEVELS,
    "queries": _UNIT_QUERIES,
    "constraints": _UNIT_CONSTRAINTS,
    "geographies": _PL94_GEO_LEVELS,
    "exact_total_levels": "all",
    "rules": "pl94_unit",
}

_DHC_PERSON = {
    "dataset": "{}_DHCP_PROD",
    "axes": _DHC_AXES,
    "levels": _DHC_AXIS_LEVELS,
    "queries": _DHC_QUERIES,
    # the relgq zero rules are one row per rule, so take every constraint row
    "constraints": "all",
    "geographies": _DHC_GEO_LEVELS,
    "exact_total_levels": (),
    "rules": "dhc_person",
}

_STATES = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9, "DE": 10,
    "DC": 11, "FL": 12, "GA": 13, "HI": 15, "ID": 16, "IL": 17, "IN": 18,
    "IA": 19, "KS": 20, "KY": 21, "LA": 22, "ME": 23, "MD": 24, "MA": 25,
    "MI": 26, "MN": 27, "MS": 28, "MO": 29, "MT": 30, "NE": 31, "NV": 32,
    "NH": 33, "NJ": 34, "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39,
    "OK": 40, "OR": 41, "PA": 42, "RI": 44, "SC": 45, "SD": 46, "TN": 47,
    "TX": 48, "UT": 49, "VT": 50, "VA": 51, "WA": 53, "WV": 54, "WI": 55,
    "WY": 56, "PR": 72,
}

_STATE_NAMES = {
    "alabama": 1, "alaska": 2, "arizona": 4, "arkansas": 5, "california": 6,
    "colorado": 8, "connecticut": 9, "delaware": 10,
    "district of columbia": 11, "florida": 12, "georgia": 13, "hawaii": 15,
    "idaho": 16, "illinois": 17, "indiana": 18, "iowa": 19, "kansas": 20,
    "kentucky": 21, "louisiana": 22, "maine": 23, "maryland": 24,
    "massachusetts": 25, "michigan": 26, "minnesota": 27, "mississippi": 28,
    "missouri": 29, "montana": 30, "nebraska": 31, "nevada": 32,
    "new hampshire": 33, "new jersey": 34, "new mexico": 35, "new york": 36,
    "north carolina": 37, "north dakota": 38, "ohio": 39, "oklahoma": 40,
    "oregon": 41, "pennsylvania": 42, "rhode island": 44,
    "south carolina": 45, "south dakota": 46, "tennessee": 47, "texas": 48,
    "utah": 49, "vermont": 50, "virginia": 51, "washington": 53,
    "west virginia": 54, "wisconsin": 55, "wyoming": 56, "puerto rico": 72,
}

_MATERIALIZE_WARN_CELLS = 2_000_000


# ── public helpers ───────────────────────────────────────────────────────────

def pl94_queries(table="person"):
    """Tabulate the queries available from `get_pl94` for a table.

    Returns a plain pandas DataFrame with one row per query: its name, the
    raw NMF query name, the number of cells per geography, and notes.
    """
    table = _normalize_table(table)
    rows = []
    if table == "person":
        for name, raw in _PERSON_QUERIES.items():
            note = "exact at the national level (total_con)" if name == "total" else ""
            rows.append((name, raw, _query_cells(_PL94_PERSON, name), note))
    else:
        rows.append(("total", "total_con", 1, "exact at every level (invariant)"))
        rows.append(("h1", "detailed_dpq", 2,
                     "conditioned on the exact total when apply_constraints=True"))
    return pd.DataFrame(rows, columns=["query", "nmf_query_name", "cells", "notes"])


def dhc_queries():
    """Tabulate the queries available from `get_dhc`.

    Returns a plain pandas DataFrame with one row per query: its name, the
    raw NMF query name, the number of cells per geography, and notes.  A
    query name lists the binning each histogram axis uses, in axis order
    (relgq, sex, age, hispanic, cenrace); unlisted axes are marginalized out.
    """
    rows = []
    for name, raw in _DHC_QUERIES.items():
        specs = _axis_specs(_DHC_PERSON, name)
        has_sex = specs[_DHC_AXES.index("sex")] != "*"
        rows.append((name, raw, _query_cells(_DHC_PERSON, name), has_sex))
    return pd.DataFrame(rows, columns=["query", "nmf_query_name", "cells", "has_sex"])


def get_pl94(geography, queries, state=None, *, table="person",
             root=DEFAULT_ROOT, nonnegative=True, apply_constraints=True,
             real_block_groups=False):
    """Read PL94 NMF measurements as a tidy frame of noisy values.

    Parameters
    ----------
    geography : one of "us", "state", "county", "tract", "block group",
        "block".  Geographies below county require `state`.
    queries : query name or list of names; see `pl94_queries()`.
    state : state FIPS code (int or str), postal abbreviation, or full name;
        may be a list.  Puerto Rico is a separate NMF product and is routed
        automatically when requested.
    table : "person" or "unit".
    root : directory containing the `*_PL_PROD` parquet datasets.
    nonnegative : truncate every posterior at zero (true counts cannot be
        negative).
    apply_constraints : condition posteriors on the NMF constraint rows
        (exact totals, structural zeros, hhgq bounds).
    real_block_groups : only valid with `geography="block group"`.  Instead
        of returning the raw spine ("optimized") block groups, reads at the
        block level and sums block measurements up to real tabulation block
        groups (a block's group digit always duplicates its own leading
        digit, so the real grouping is recoverable exactly).  This reads and
        materializes every block in the requested area, so it is far more
        expensive than the default.

    Returns a pandas DataFrame with one row per cell: `geoid` (standard
    census GEOID where derivable, otherwise <NA>), `geocode` (raw spine
    code, or <NA> when `real_block_groups=True` since a summed row no longer
    corresponds to one spine geography), `aian` (True/False for AIAN-branch
    spine geographies, or <NA> when `real_block_groups=True` and the real
    block group straddles both branches), `query`, one label column per
    histogram axis, `value` (`NoisyInt`, or `NoisyFloat` when
    `real_block_groups=True` since summing promotes it), and `variance` (the
    NMF measurement variance).
    """
    table = _normalize_table(table)
    spec = _PL94_PERSON if table == "person" else _PL94_UNIT
    level = _normalize_geography(geography, spec)
    queries = _normalize_queries(queries, spec["queries"])

    states = _resolve_states(state)
    if level == "US":
        if states:
            raise ValueError('geography "us" does not take a state')
    elif level in ("Tract", "Block_Group", "Block") and not states:
        raise ValueError(f'geography "{geography}" requires a state')

    if real_block_groups and level != "Block_Group":
        raise ValueError(
            'real_block_groups=True only applies to geography="block group"')

    fetch_level = level
    if level == "Block_Group":
        if real_block_groups:
            fetch_level = "Block"
        else:
            warnings.warn(
                "NMF block groups are DAS 'optimized block groups' and do not "
                "correspond to census tabulation block groups; geoid is left "
                "<NA>. Pass real_block_groups=True to recover real "
                "block-group totals by summing block-level measurements.",
                stacklevel=2)

    products = _group_states_by_product(states)
    frames = [
        _load_product(spec, product, fips_list, fetch_level, queries, root,
                      nonnegative, apply_constraints)
        for product, fips_list in products
    ]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if level == "Block_Group" and real_block_groups:
        frame = _aggregate_real_block_groups(frame, spec["axes"])
    return frame


def get_dhc(geography, queries, state=None, county=None, *,
            root=DEFAULT_DHC_ROOT, nonnegative=True, apply_constraints=True):
    """Read DHC person NMF measurements as a tidy frame of noisy values.

    This is the product that resolves sex and single-year age; PL94 has
    neither.  Its histogram is relgq x sex x age x hispanic x cenrace.

    Parameters
    ----------
    geography : "us", "state", or "county".  County requires `state`.
    queries : query name or list of names; see `dhc_queries()`.  Note that
        `detailed` alone is 1,227,744 cells per geography.
    state : state FIPS code (int or str), postal abbreviation, or full name;
        may be a list.  Puerto Rico is a separate NMF product and is routed
        automatically when requested.
    county : county FIPS code (int or str) or list of them, to narrow a
        county-level read; the DHC release partitions by county, so this
        prunes the files that are read.
    root : directory holding the `*_DHCP_PROD` parquet datasets, as laid down
        by `fetch_dhc`.
    nonnegative : truncate every posterior at zero (true counts cannot be
        negative).
    apply_constraints : condition posteriors on the NMF constraint rows --
        the exact PL94 table for the geography (`pl94_con`), the relgq age
        rules, and the per-facility-type GQ bounds.  Cells the PL94 table
        pins are returned as point masses, and cells it ties together in
        pairs are built jointly, so they always sum to the invariant.

    Returns a pandas DataFrame shaped like `get_pl94`'s: `geoid`, `geocode`,
    `aian`, `query`, one label column per histogram axis (`relgq`, `sex`,
    `age`, `hispanic`, `cenrace`), `value` (`NoisyInt`), and `variance`.
    """
    spec = _DHC_PERSON
    level = _normalize_geography(geography, spec)
    queries = _normalize_queries(queries, spec["queries"])

    states = _resolve_states(state)
    counties = _resolve_counties(county)
    if level == "US":
        if states:
            raise ValueError('geography "us" does not take a state')
    elif level == "County" and not states:
        raise ValueError('geography "county" requires a state')
    if counties and level != "County":
        raise ValueError('county only applies to geography="county"')

    products = _group_states_by_product(states)
    frames = [
        _load_product(spec, product, fips_list, level, queries, root,
                      nonnegative, apply_constraints, counties=counties)
        for product, fips_list in products
    ]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def fetch_dhc(geography, state=None, county=None, *, root=DEFAULT_DHC_ROOT,
              overwrite=False):
    """Download DHC person NMF partitions into `root` for `get_dhc` to read.

    The 2020 DHC NMF is public but large, so nothing is bundled with this
    package and nothing is fetched implicitly: name the geographies you want
    and they are mirrored, preserving the release's own directory layout.
    A state-level partition runs about 1.5 MB per state and a county-level
    one about 1.8 MB per county.

    Parameters mirror `get_dhc`: `geography` is "us", "state", or "county",
    `state` selects states (required below the national level), and `county`
    narrows a county-level fetch to particular county FIPS codes.  Files that
    already exist are skipped unless `overwrite=True`.

    Returns the list of local paths that now hold the requested partitions.
    """
    spec = _DHC_PERSON
    level = _normalize_geography(geography, spec)
    states = _resolve_states(state)
    counties = _resolve_counties(county)
    if level == "US":
        if states:
            raise ValueError('geography "us" does not take a state')
    elif not states:
        raise ValueError(f'geography "{geography}" requires a state')
    if counties and level != "County":
        raise ValueError('county only applies to geography="county"')

    paths = []
    for product, fips_list in _group_states_by_product(states):
        dataset = spec["dataset"].format(product)
        for kind in ("DPQuery", "Constraint"):
            base = f"{DHC_S3_PREFIX}/{dataset}/{level}.parquet/{kind}/"
            # each state is split into a non-AIAN and an AIAN spine branch
            prefixes = [base] if level == "US" else [
                f"{base}State={branch:03d}/"
                for fips in fips_list
                for branch in (fips, fips + 100)
            ]
            for prefix in prefixes:
                for key in _list_s3_keys(prefix):
                    if not key.endswith(".parquet"):
                        continue
                    if counties and _county_of_key(key) not in counties:
                        continue
                    paths.append(_download_key(key, root, overwrite))
    if not paths:
        raise RuntimeError(
            "no DHC partitions matched the request; check the state and "
            "county codes")
    return paths


def _county_of_key(key):
    """County FIPS code of a `County=<geocode>`-partitioned object key."""
    for part in key.split("/"):
        if part.startswith("County="):
            return int(part[-4:])
    return None


def _list_s3_keys(prefix):
    keys = []
    token = None
    while True:
        url = f"{DHC_S3_BASE}?list-type=2&prefix={prefix}"
        if token is not None:
            url += f"&continuation-token={urllib.parse.quote(token, safe='')}"
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys.extend(e.text for e in root.findall("s3:Contents/s3:Key", ns)
                    if e.text)
        if root.findtext("s3:IsTruncated", "false", ns).lower() != "true":
            return keys
        token = root.findtext("s3:NextContinuationToken", "", ns)
        if not token:
            raise RuntimeError("truncated S3 listing without a continuation token")


def _download_key(key, root, overwrite):
    """Mirror one object into `root`, keeping its path below the NMF prefix."""
    dest = os.path.join(root, os.path.relpath(key, DHC_S3_PREFIX))
    if os.path.exists(dest) and not overwrite:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    partial = dest + ".part"
    try:
        with urllib.request.urlopen(f"{DHC_S3_BASE}/{key}", timeout=60) as src, \
                open(partial, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
        os.replace(partial, dest)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return dest


# ── input normalization ──────────────────────────────────────────────────────

def _normalize_table(table):
    t = str(table).strip().lower().rstrip("s")
    if t not in ("person", "unit"):
        raise ValueError(f'table must be "person" or "unit", got {table!r}')
    return t


def _normalize_geography(geography, spec):
    levels = spec["geographies"]
    g = str(geography).strip().lower()
    if g not in levels:
        raise ValueError(
            f"unknown geography {geography!r}; expected one of "
            f"{sorted(set(levels))}")
    return levels[g]


def _normalize_queries(queries, qmap):
    if isinstance(queries, str):
        queries = [queries]
    # Query names carry axis specs like popSehsdTargetsRelship, so match
    # case-insensitively rather than requiring the caller to match case.
    by_key = {q.lower().replace(" ", ""): q for q in qmap}
    result = []
    for q in queries:
        key = str(q).strip().lower().replace(" ", "")
        if key.endswith("_dpq"):
            key = key[:-4]
        if key not in by_key:
            raise ValueError(
                f"unknown query {q!r}; expected one of {sorted(qmap)}")
        canonical = by_key[key]
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError("at least one query is required")
    return result


def _resolve_states(state):
    if state is None:
        return []
    if isinstance(state, (str, int, np.integer)):
        state = [state]
    fips = []
    for s in state:
        if isinstance(s, (int, np.integer)):
            f = int(s)
        else:
            text = str(s).strip()
            if text.isdigit():
                f = int(text)
            elif text.upper() in _STATES:
                f = _STATES[text.upper()]
            elif text.lower() in _STATE_NAMES:
                f = _STATE_NAMES[text.lower()]
            else:
                raise ValueError(f"unknown state {s!r}")
        if f not in _STATES.values():
            raise ValueError(f"unknown state FIPS code {f}")
        if f not in fips:
            fips.append(f)
    return fips


def _resolve_counties(county):
    if county is None:
        return []
    if isinstance(county, (str, int, np.integer)):
        county = [county]
    fips = []
    for c in county:
        text = str(c).strip()
        if not text.isdigit():
            raise ValueError(f"county must be a FIPS code, got {c!r}")
        f = int(text)
        if f not in fips:
            fips.append(f)
    return fips


def _group_states_by_product(states):
    """Split requested states into (product, fips_list) pairs."""
    if not states:
        return [("US", [])]
    us = [f for f in states if f != 72]
    groups = []
    if us:
        groups.append(("US", us))
    if 72 in states:
        groups.append(("PR", [72]))
    return groups


def _axis_specs(spec, canonical):
    """Per-axis spec strings for a canonical query name.

    A canonical name lists the spec each resolved axis uses, joined by "*";
    every axis it does not name is marginalized out ("*").
    """
    axes, levels = spec["axes"], spec["levels"]
    if canonical == "detailed":
        return tuple("detailed" for _ in axes)
    parts = canonical.split("*")
    return tuple(
        next((p for p in parts if p in levels[ax] and p != "*"), "*")
        for ax in axes)


def _person_axis_specs(canonical):
    """Axis spec strings for a canonical PL94 person query name."""
    return _axis_specs(_PL94_PERSON, canonical)


def _query_cells(spec, canonical):
    """Cells per geography for a canonical query name."""
    return math.prod(
        len(spec["levels"][ax][s])
        for ax, s in zip(spec["axes"], _axis_specs(spec, canonical)))


# ── data access ──────────────────────────────────────────────────────────────

def _scan(root, spec, product, level):
    path = f"{root}/{spec['dataset'].format(product)}/{level}.parquet"
    # Partitions were written by different jobs: integer widths drift and
    # DPQuery partitions may lack the sign column entirely.
    return pl.scan_parquet(
        path,
        cast_options=pl.ScanCastOptions(integer_cast="upcast"),
        missing_columns="insert",
    )


def _collect_rows(root, spec, product, level, fips_list, query_names,
                  constraints=(), counties=()):
    lf = _scan(root, spec, product, level)
    if fips_list and level != "US":
        partitions = [f for fips in fips_list for f in (fips, fips + 100)]
        lf = lf.filter(pl.col("State").is_in(partitions))
    if counties:
        # The DHC county partition value is the spine geocode, whose last four
        # digits are the county FIPS code; it is read as an integer.
        lf = lf.filter(
            pl.col("County").cast(pl.Utf8).str.slice(-4).cast(pl.Int64)
            .is_in(list(counties)))
    keep = pl.col("query_name").is_in(list(query_names))
    if constraints == "all":
        if "sign" in lf.collect_schema().names():
            keep = keep | pl.col("sign").is_not_null()
    elif constraints:
        keep = keep | pl.col("query_name").is_in(list(constraints))
    return lf.filter(keep).collect()


def _tract_geoids(root, spec, product, fips_list):
    """Map spine tract codes to standard 11-char tract GEOIDs.

    The spine renumbers tracts, but block geocodes embed the standard block
    GEOID in their last 16 characters, and spine tracts map 1:1 onto
    tabulation tracts; scanning one query's block geocodes recovers the
    correspondence.
    """
    lf = _scan(root, spec, product, "Block")
    if fips_list:
        partitions = [f for fips in fips_list for f in (fips, fips + 100)]
        lf = lf.filter(pl.col("State").is_in(partitions))
    geocodes = (
        lf.filter(pl.col("query_name") == "detailed_dpq")
        .select("geocode").unique().collect()["geocode"]
    )
    xwalk = {}
    conflicts = set()
    for g in geocodes:
        spine_tract = g[:-18]
        real = g[-16:][:11]
        if xwalk.setdefault(spine_tract, real) != real:
            conflicts.add(spine_tract)
    for spine_tract in conflicts:
        xwalk[spine_tract] = None
    if conflicts:
        warnings.warn(
            f"{len(conflicts)} spine tracts span multiple tabulation tracts; "
            "their geoid is left <NA>.", stacklevel=2)
    return xwalk


def _geoid(level, geocode, tract_xwalk):
    if level == "US":
        return "1"
    if level == "State":
        return f"{int(geocode) % 100:02d}"
    if level == "County":
        return f"{int(geocode[:3]) % 100:02d}{int(geocode[4:8]):03d}"
    if level == "Tract":
        return tract_xwalk.get(geocode)
    if level == "Block":
        real = geocode[-16:]
        # last 16 chars: state(2) county(3) tract(6) bg(1) block(4); the bg
        # digit duplicates the block code's first digit.
        return real[:11] + real[12:]
    return None  # Block_Group: spine block groups are not tabulation ones


def _is_aian(level, geocode):
    if level == "US":
        return False
    return geocode.startswith("1")


def _aggregate_real_block_groups(frame, axes):
    """Sum block-level rows up to real tabulation block-group totals.

    A block's `geoid` (state+county+tract+block) already contains its real
    block group prefix: the first 12 characters are state+county+tract plus
    the block group digit, since that digit always duplicates the block
    code's own leading digit (see `_geoid`). Summed with plain `+` rather
    than the extension array's own `.sum()` reduction, since pandas' groupby
    machinery collapses that reduction to a constant, silently discarding
    every contributing block's noise.
    """
    frame = frame.copy()
    frame["geoid"] = frame["geoid"].str.slice(0, 12)
    group_cols = ["geoid", "query", *axes]

    rows = []
    for key, group in frame.groupby(group_cols, sort=False):
        values = list(group["value"])
        total = values[0]
        for v in values[1:]:
            total = total + v
        aian_values = group["aian"].unique()
        aian = bool(aian_values[0]) if len(aian_values) == 1 else pd.NA
        rows.append((*key, aian, float(group["variance"].sum()), total))

    out = pd.DataFrame(rows, columns=[*group_cols, "aian", "variance", "value"])
    out["geoid"] = out["geoid"].astype("string")
    out["aian"] = pd.array(out["aian"], dtype="boolean")
    out["geocode"] = pd.array([pd.NA] * len(out), dtype="string")
    out["value"] = NoisyFloatArray(
        np.array([float(v._obs) for v in out["value"]], dtype="float64"),
        np.array([v._root for v in out["value"]], dtype=object))
    return out[["geoid", "geocode", "aian", "query", *axes, "value", "variance"]]


# ── posterior construction ───────────────────────────────────────────────────

def _exact_count(pinned):
    """A NoisyInt with no uncertainty left: a constraint pins the true count.

    The measurement it replaces is dropped rather than kept as the observed
    value, since the posterior is a point mass and the noisy number is known
    to be wrong.
    """
    return NoisyInt(int(pinned), DerivedNode(sp.Integer(int(pinned))))


def _noisy_count(obs, sd, lb=None, ub=None):
    """A NoisyInt whose posterior is a (truncated) discrete Gaussian at obs.

    The posterior is encoded directly as obs - eps rather than through a
    latent-plus-constraint pair, so building and sampling large collections
    of census cells avoids the symbolic solve step entirely.
    """
    obs = int(obs)
    if lb is not None and ub is not None and lb == ub:
        return _exact_count(lb)
    if lb is None and ub is None:
        node = DiscreteGaussianNode.create(loc=sp.Integer(0), scale=sp.Float(sd))
    else:
        # theta = obs - eps lies in [lb, ub] iff eps lies in [obs-ub, obs-lb].
        low = -sp.oo if ub is None else sp.Integer(obs - int(ub))
        high = sp.oo if lb is None else sp.Integer(obs - int(lb))
        node = TruncatedDiscreteGaussianNode.create(
            loc=sp.Integer(0), scale=sp.Float(sd), low=low, high=high)
    root = DerivedNode(sp.Integer(obs) - node.expr, deps=(node,))
    return NoisyInt(obs, root)


def _conditioned_pair(obs_a, obs_b, variance, total, nonnegative):
    """Two cells conditioned on their exact sum.

    With independent measurements obs_a = a + e1 and obs_b = b + e2 (equal
    variance v) and a known exact a + b = total, the posterior of a over the
    integers is a discrete Gaussian centered at (obs_a - obs_b + total)/2
    with variance v/2, and b = total - a exactly (perfect negative
    correlation).  This is the only block size for which the conditional
    posterior is again a discrete Gaussian, so it is the only one built
    jointly; see `_dhc_pl94_blocks`.
    """
    obs_a, obs_b, total = int(obs_a), int(obs_b), int(total)
    scale = sp.sqrt(sp.Float(float(variance)) / 2)
    delta = sp.Rational(obs_a + obs_b - total, 2)
    if nonnegative:
        node = TruncatedDiscreteGaussianNode.create(
            loc=delta, scale=scale,
            low=sp.Integer(obs_a - total), high=sp.Integer(obs_a))
    else:
        node = DiscreteGaussianNode.create(loc=delta, scale=scale)
    a_root = DerivedNode(sp.Integer(obs_a) - node.expr, deps=(node,))
    a = NoisyInt(obs_a, a_root)
    b_root = DerivedNode(sp.Integer(total) - a_root.expr, deps=(a_root,))
    b = NoisyInt(obs_b, b_root)
    return a, b


# ── constraint rules ─────────────────────────────────────────────────────────

def _pl94_person_rules(geocode, specs, canonical, constraints, exact, lb, ub):
    """Fill PL94 person bounds: hhgq category bounds and the nursing zero."""
    bounds = _hhgq_axis_bounds(geocode, specs, constraints)
    if bounds is not None:
        for pos, (g_lb, g_ub) in enumerate(bounds):
            if g_ub is not None:
                ub[pos] = np.minimum(ub[pos], g_ub)
            if g_lb is not None:
                lb[pos] = np.maximum(lb[pos], g_lb)
    nurse = constraints.get((geocode, "nurse_nva_0_con"))
    if canonical == "detailed" and nurse is not None and int(nurse[0]) == 0:
        exact[3, 0] = 0.0                # under-18 residents of nursing homes


def _dhc_person_rules(geocode, specs, sizes, obs, variance, constraints,
                      constraint_specs, nonnegative, exact, lb, ub):
    """Apply the DHC constraint rows to one query's cells.

    Fills `exact`/`lb`/`ub` in place and returns {flat index: NoisyInt} for
    the cells that had to be built jointly.
    """
    _dhc_gq_bounds(geocode, specs, constraints, lb, ub)
    _dhc_structural_zeros(geocode, specs, sizes, constraints, constraint_specs,
                          exact)
    return _dhc_pl94_blocks(geocode, specs, sizes, obs, variance, constraints,
                            nonnegative, exact, ub)


def _dhc_gq_bounds(geocode, specs, constraints, lb, ub):
    """Bound the relgq axis by the per-facility-type hhgq bound rows.

    The bounds count facilities: a GQ type with f facilities of that kind in
    the geography holds at least f people and at most 99999 each.  Upper
    bounds transfer to any sub-slice of a category; lower bounds only apply
    when the cell is the whole category total.
    """
    lows = constraints.get((geocode, "hhgq_total_lb_con"))
    highs = constraints.get((geocode, "hhgq_total_ub_con"))
    groups = _DHC_AXIS_GROUPS["relgq"].get(specs[0])
    if lows is None or highs is None or groups is None:
        return
    whole_category = all(s == "*" for s in specs[1:])
    for pos, group in enumerate(groups):
        members = set(group)
        touched = [level for level, levels in enumerate(_GQLEVEL_MEMBERS)
                   if members & set(levels)]
        ub[pos] = np.minimum(ub[pos], sum(int(highs[l]) for l in touched))
        if whole_category:
            covered = [level for level, levels in enumerate(_GQLEVEL_MEMBERS)
                       if set(levels) <= members]
            lb[pos] = np.maximum(lb[pos], sum(int(lows[l]) for l in covered))


def _dhc_structural_zeros(geocode, specs, sizes, constraints, constraint_specs,
                          exact):
    """Pin cells the relgq age rules forbid (no 14-year-old householders)."""
    relgq_groups = _DHC_AXIS_GROUPS["relgq"].get(specs[0])
    age_groups = _DHC_AXIS_GROUPS["age"].get(specs[2])
    if relgq_groups is None or age_groups is None:
        return
    for name, rule_specs in constraint_specs.items():
        match = re.fullmatch(r"relgq(\d+)", rule_specs[0])
        forbidden = _AGE_ZERO_RULES.get(rule_specs[2])
        if match is None or forbidden is None:
            continue
        value = constraints.get((geocode, name))
        if value is None or int(value[0]) != 0:
            continue
        level = (int(match.group(1)),)
        rows = [pos for pos, g in enumerate(relgq_groups) if g == level]
        cols = [pos for pos, g in enumerate(age_groups)
                if set(g) <= forbidden]
        if rows and cols:
            index = [np.arange(size) for size in sizes]
            index[0], index[2] = np.array(rows), np.array(cols)
            exact[np.ix_(*index)] = 0.0


def _dhc_pl94_blocks(geocode, specs, sizes, obs, variance, constraints,
                     nonnegative, exact, ub):
    """Condition on pl94_con, the exact published PL94 table for the area.

    The DHC histogram marginalizes onto the PL94 one, so a query's cells
    split into blocks whose totals are known exactly (`_pl94_blocks`).  What
    that buys depends on how many cells of a block are still free:

    - a block totalling zero pins every cell in it to zero, which is common
      in the detailed queries because most PL94 cells are empty;
    - one free cell is a point mass at the block total;
    - two are conditioned jointly, so they stay perfectly anti-correlated and
      always sum to the invariant;
    - more than two only take the block total as an upper bound.  The exact
      conditional posterior there is a discrete Gaussian restricted to a
      lattice hyperplane, which does not factor into per-cell distributions
      the node graph can sample independently, so the sum is left free.
    """
    row = constraints.get((geocode, "pl94_con"))
    if row is None:
        return {}
    pl94 = np.asarray(row, dtype=np.int64).reshape(_PL94_SHAPE)
    exact_flat, ub_flat = exact.reshape(-1), ub.reshape(-1)
    prebuilt = {}

    for positions, selector in _pl94_blocks(specs):
        cells = _block_flat_indices(positions, sizes)
        # cells already pinned are structural zeros, so they take nothing off
        # the block total: blocks are disjoint, and nothing else has run yet
        free = cells[np.isnan(exact_flat[cells])]
        if free.size == 0:
            continue
        total = int(pl94[np.ix_(*selector)].sum())
        if total == 0 and nonnegative:
            exact_flat[free] = 0.0
        elif free.size == 1:
            exact_flat[free[0]] = total
        elif free.size == 2:
            first, second = (int(cell) for cell in free)
            prebuilt[first], prebuilt[second] = _conditioned_pair(
                int(obs[first]), int(obs[second]), variance, total, nonnegative)
        elif nonnegative:
            ub_flat[free] = np.minimum(ub_flat[free], total)
    return prebuilt


def _pl94_blocks(specs):
    """Split a DHC query's cells into blocks pl94_con pins the total of.

    Yields (per-axis cell positions, pl94 selector) pairs.  Sex is merged
    away because PL94 does not resolve it; every other axis is merged only
    as far as it must be for its blocks to be whole PL94 categories.
    """
    per_axis = [
        _merged_blocks(_DHC_AXIS_GROUPS[axis][s], _PL94_CLASS_OF.get(axis))
        for axis, s in zip(_DHC_AXES, specs)
    ]
    for combination in itertools.product(*per_axis):
        selector = [None] * len(_PL94_SHAPE)
        for axis, (_, categories) in zip(_DHC_AXES, combination):
            if axis in _PL94_AXIS_OF:
                selector[_PL94_AXIS_OF[axis]] = categories
        yield tuple(positions for positions, _ in combination), tuple(selector)


def _merged_blocks(groups, category_of):
    """Merge an axis's groups until each block is whole PL94 categories.

    Returns (group positions, category indices) pairs.  An axis PL94 does not
    resolve merges into one block, as do groups finer than a PL94 category
    (the two householder levels, say, which PL94 only knows in total).
    """
    if category_of is None:
        return [(tuple(range(len(groups))), ())]
    blocks = []
    for position, group in enumerate(groups):
        positions, categories = [position], {category_of[i] for i in group}
        disjoint = []
        for other_positions, other_categories in blocks:
            if categories & other_categories:
                positions.extend(other_positions)
                categories |= other_categories
            else:
                disjoint.append((other_positions, other_categories))
        blocks = disjoint + [(positions, categories)]
    return [(tuple(sorted(p)), tuple(sorted(c))) for p, c in blocks]


def _block_flat_indices(positions, sizes):
    grids = np.meshgrid(*(np.asarray(p) for p in positions), indexing="ij")
    return np.ravel_multi_index(tuple(g.ravel() for g in grids), sizes)


# ── assembly ─────────────────────────────────────────────────────────────────

def _load_product(spec, product, fips_list, level, queries, root, nonnegative,
                  apply_constraints, counties=()):
    axes, qmap = spec["axes"], spec["queries"]
    raw_names = {qmap[q] for q in queries if qmap[q] is not None}
    exact_levels = spec["exact_total_levels"]
    exact_total = "total" in queries and (
        exact_levels == "all" or level in exact_levels)
    con_names = spec["constraints"]

    rows = _collect_rows(
        root, spec, product, level, fips_list, raw_names,
        constraints=con_names if (apply_constraints or exact_total) else (),
        counties=counties)

    constraints = {}
    constraint_specs = {}
    if "sign" in rows.columns:
        for r in rows.filter(pl.col("sign").is_not_null()).iter_rows(named=True):
            constraints[(r["geocode"], r["query_name"])] = r["value"]
            # a constraint's axis specs say what it constrains, and are the
            # same wherever the row appears
            constraint_specs.setdefault(
                r["query_name"], tuple(r[ax] for ax in axes))
        rows = rows.filter(pl.col("sign").is_null())
    dpq = rows.sort("query_name", "geocode")

    total_cells = int(dpq.select(pl.col("value").list.len().sum()).item() or 0)
    if total_cells > _MATERIALIZE_WARN_CELLS:
        warnings.warn(
            f"materializing {total_cells:,} noisy values; consider narrowing "
            "the geography, state, or query selection.", stacklevel=3)

    tract_xwalk = {}
    if level == "Tract":
        tract_xwalk = _tract_geoids(root, spec, product, fips_list)

    canonical_by_raw = {v: k for k, v in qmap.items() if v is not None}
    columns = {
        "geoid": [], "geocode": [], "aian": [], "query": [],
        **{ax: [] for ax in axes},
        "variance": [],
    }
    obs_out, roots_out = [], []

    def emit(geocode, query, labels, variance, value):
        columns["geoid"].append(_geoid(level, geocode, tract_xwalk))
        columns["geocode"].append(geocode)
        columns["aian"].append(_is_aian(level, geocode))
        columns["query"].append(query)
        for ax, lab in zip(axes, labels):
            columns[ax].append(lab)
        columns["variance"].append(variance)
        obs_out.append(value._obs)
        roots_out.append(value._root)

    if exact_total:
        for (geocode, name), value in sorted(constraints.items()):
            if name != "total_con":
                continue
            total = int(value[0])
            emit(geocode, "total", ("total",) * len(axes), 0.0,
                 _exact_count(total))

    for r in dpq.iter_rows(named=True):
        canonical = canonical_by_raw[r["query_name"]]
        if spec["rules"] == "pl94_unit" and canonical == "h1":
            _emit_unit_h1(emit, r, constraints, nonnegative, apply_constraints)
        else:
            _emit_cells(emit, r, canonical, spec, constraints,
                        constraint_specs, nonnegative, apply_constraints)

    frame = pd.DataFrame({
        "geoid": pd.array(columns["geoid"], dtype="string"),
        "geocode": columns["geocode"],
        "aian": columns["aian"],
        "query": columns["query"],
        **{ax: columns[ax] for ax in axes},
        "value": NoisyIntArray(
            np.asarray(obs_out, dtype="int64"),
            np.asarray(roots_out, dtype=object)),
        "variance": np.asarray(columns["variance"], dtype="float64"),
    })
    return frame


def _emit_unit_h1(emit, row, constraints, nonnegative, apply_constraints):
    geocode = row["geocode"]
    variance = float(row["variance"])
    obs_vac, obs_occ = (int(x) for x in row["value"])
    total = constraints.get((geocode, "total_con"))
    if apply_constraints and total is not None:
        vac, occ = _conditioned_pair(
            obs_vac, obs_occ, variance, int(total[0]), nonnegative)
    else:
        sd = math.sqrt(variance)
        lb = 0 if nonnegative else None
        vac = _noisy_count(obs_vac, sd, lb=lb)
        occ = _noisy_count(obs_occ, sd, lb=lb)
    emit(geocode, "h1", ("vacant",), variance, vac)
    emit(geocode, "h1", ("occupied",), variance, occ)


def _emit_cells(emit, row, canonical, spec, constraints, constraint_specs,
                nonnegative, apply_constraints):
    axes = spec["axes"]
    geocode = row["geocode"]
    variance = float(row["variance"])
    sd = math.sqrt(variance)
    values = row["value"]
    specs = tuple(row[ax] for ax in axes)
    axis_labels = [spec["levels"][ax][s] for ax, s in zip(axes, specs)]

    sizes = tuple(len(labels) for labels in axis_labels)
    if math.prod(sizes) != len(values):
        raise ValueError(
            f"query {row['query_name']!r} at {geocode!r}: "
            f"{len(values)} cells do not match axis sizes {sizes}")

    # Per-cell posterior plan: a pinned true value, an interval, or (for cells
    # a constraint ties to one another) a value already built jointly.
    exact = np.full(sizes, np.nan)
    lb = np.full(sizes, 0.0 if nonnegative else -np.inf)
    ub = np.full(sizes, np.inf)
    prebuilt = {}
    if apply_constraints:
        if spec["rules"] == "pl94_person":
            _pl94_person_rules(geocode, specs, canonical, constraints,
                               exact, lb, ub)
        elif spec["rules"] == "dhc_person":
            prebuilt = _dhc_person_rules(
                geocode, specs, sizes, values, variance, constraints,
                constraint_specs, nonnegative, exact, lb, ub)

    exact_flat, lb_flat, ub_flat = (a.reshape(-1) for a in (exact, lb, ub))
    for flat, idx in enumerate(itertools.product(*(range(s) for s in sizes))):
        obs = int(values[flat])
        labels = tuple(axis_labels[a][i] for a, i in enumerate(idx))
        if flat in prebuilt:
            value = prebuilt[flat]
        elif not math.isnan(exact_flat[flat]):
            value = _exact_count(int(exact_flat[flat]))
        else:
            low, high = lb_flat[flat], ub_flat[flat]
            value = _noisy_count(
                obs, sd,
                lb=None if low == -np.inf else int(low),
                ub=None if high == np.inf else int(high))
        emit(geocode, canonical, labels, variance, value)


def _hhgq_axis_bounds(geocode, specs, constraints):
    """Per-position (lb, ub) bounds on the query's hhgq axis, or None.

    Upper bounds always transfer to sub-slices of an hhgq category (counts
    are nonnegative); lower bounds only apply when the cell is the entire
    category total, i.e. every other axis is marginalized out.
    """
    groups = _HHGQ_GROUPS.get(specs[0])
    lb = constraints.get((geocode, "hhgq_total_lb_con"))
    ub = constraints.get((geocode, "hhgq_total_ub_con"))
    if groups is None or lb is None or ub is None:
        return None
    whole_category = all(s == "*" for s in specs[1:])
    bounds = []
    for group in groups:
        g_ub = sum(int(ub[i]) for i in group)
        g_lb = sum(int(lb[i]) for i in group) if whole_category else None
        bounds.append((g_lb, g_ub))
    return bounds
