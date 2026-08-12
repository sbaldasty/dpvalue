"""The 2020 Census Noisy Measurement Files, as `dataset` framework products.

This is the library's Census interface: the PL94 person and housing-unit
NMFs, and the DHC person NMF, expressed as what the `dataset` framework asks
a dataset extension for, and nothing else:

1. **A schema**, declared as `Axis`/`Binning`/`Measurement`/`GeoLevel` literals.
   Data, not code.
2. **Physical access** (`NmfSource`): parquet layout, Hive partition pruning,
   S3 mirroring, schema drift.  Genuinely Census-specific.
3. **A translation** of the release's own constraint rows into the framework's
   evidence vocabulary -- `nurse_nva_0_con` *means* "these cells are
   structurally zero", `pl94_con` *means* "the totals of this coarsening are
   exact".  What follows from that is the framework's problem, not this file's.
4. **An atom map** for the one geography whose partition is incomparable with
   the DAS spine: tabulation block groups.  A block geocode embeds its
   tabulation GEOID in its last 16 characters, which is the only reason the
   crosswalk is derivable at all, and is not something core could know.
5. **A noise family**: discrete Gaussian, scale from the `variance` column.

What is *not* here, because it lives in the framework: the block ladder, the
conditional-pair closed form, the merging of a query's binning against a
constraint's, the truncation bookkeeping, and the geographic re-aggregation.
See `docs/dataset-framework.md` for the design rationale.

The headline feature is joint consistency: cells live in a `View` and are
resolved once, so two reads of one physical unknown are the same posterior
rather than two independent ones that disagree when sampled together::

    from noisyvalue import census

    view = census.dhc_2020.view("county", state="VT", county=[7, 9])
    sex = view.query("sex*hispanic")
    ages = view.query("sex*age_38_groups")
    view.unused_evidence()      # what the release knew and we could not use

Two queries against the same view share every latent cell they both name; two
views are independent.  See `dataset/view.py` for the full contract.

`get_pl94` / `get_dhc` are thin convenience wrappers around a throwaway view --
one call, one view, exactly as consistent as the library's original per-cell
interface was.  Open a view directly (as above) when consistency across calls
matters, which is most of the time.
"""

import itertools
import math
import os
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import polars as pl

from .dataset import Axis, ExactHistogram, GeoLevel, Measurement, Partition
from .dataset import Product, Region, Source, Universe, ZeroRegion
from .dataset import BoundedHistogram, MeasurementRow, marginal_coarsening

DEFAULT_PL94_ROOT = "data/2020-pl94-nmf-parquets"
DEFAULT_DHC_ROOT = "data/2020-dhc-nmf-parquets"

DHC_S3_BASE = "https://uscb-2020-product-releases.s3.us-west-2.amazonaws.com"
DHC_S3_PREFIX = "decennial/dhc/2020/nmf/2020-dhc-nmf"

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


# ── codebook ─────────────────────────────────────────────────────────────────
# Level orderings follow the 2020 DAS schema definitions and were validated
# against published 2020 PL94 tabulations (test/test_census.py); e.g. the six
# single-race national counts each land within a couple standard deviations
# of their published values only under this ordering.  The DHC-P orderings
# were recovered from the national NMF itself -- every coarse query is a
# marginal of `detailed_dpq`, so a wrong grouping disagrees far beyond the
# noise -- and are re-checked the same way.

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


# ── schema: axes ─────────────────────────────────────────────────────────────
# Every axis gets "*" (marginalized) and "detailed" for free from `Axis`; what
# is declared here is the release's named binnings.  Each is a partition of the
# axis's base levels except where noted, and the framework checks it: groups
# must be disjoint, and labels must be parallel to groups.

def _singles(n):
    return tuple((i,) for i in range(n))


def _pl94_person_universe():
    hhgq = Axis("hhgq", HHGQ_LEVELS)
    hhgq.alias("hhgq", "detailed")
    hhgq.declare("hhinstlevels", ((0,), (1, 2, 3, 4), (5, 6, 7)),
                 HHINSTLEVELS_LEVELS)
    # a sub-selection, not a partition: it names nursing homes and nothing else
    hhgq.declare("gqNursingTotal", ((3,),), ("nursing",))

    votingage = Axis("votingage", VOTINGAGE_LEVELS)
    votingage.alias("votingage", "detailed")
    votingage.declare("nonvoting", ((0,),), ("under_18",))

    hispanic = Axis("hispanic", HISPANIC_LEVELS)
    hispanic.alias("hispanic", "detailed")

    cenrace = Axis("cenrace", CENRACE_LEVELS)
    cenrace.alias("cenrace", "detailed")

    return Universe("pl94 person", (hhgq, votingage, hispanic, cenrace))


def _pl94_unit_universe():
    h1 = Axis("h1", H1_LEVELS)
    h1.alias("h1", "detailed")
    return Universe("pl94 unit", (h1,))


def _dhc_person_universe():
    relgq = Axis("relgq", RELGQ_LEVELS)
    relgq.alias("relgq", "detailed")
    relgq.declare(
        "relgq_4_groups",
        ((0, 1), tuple(range(2, 18)), tuple(range(18, 33)), tuple(range(33, 42))),
        RELGQ_4_LEVELS)
    relgq.declare(
        "popSehsdTargetsRelship",
        (*_singles(18), tuple(range(18, 42))),
        POPSEHSD_RELSHIP_LEVELS)
    relgq.declare(
        "relship_and_eight_level_GQ",
        (*_singles(18),
         tuple(range(18, 24)),    # correctional
         (24, 25, 26),            # juvenile
         (27,),                   # nursing
         tuple(range(28, 33)),    # other institutional
         (33,),                   # college housing
         (34, 35),                # military
         tuple(range(36, 42))),   # other noninstitutional
        RELSHIP_GQ8_LEVELS)
    # the six groups the DAS measures against age_10_groups; each collects GQ
    # types sharing an age restriction
    relgq.declare(
        "gq_constr_groups",
        (tuple(range(18)),
         (32,),
         tuple(range(18, 23)),
         (33, 37, 38, 39, 40, 41),
         (23, 31, 34, 35),
         (24, 25, 26, 27, 28, 29, 30, 36)),
        GQ_CONSTR_LEVELS)

    sex = Axis("sex", SEX_LEVELS)
    sex.alias("sex", "detailed")

    age = Axis("age", AGE_LEVELS)
    for name, ranges, labels in (
            ("age_18_64_116", AGE_3_RANGES, AGE_3_LEVELS),
            ("age_10_groups", AGE_10_RANGES, AGE_10_LEVELS),
            ("age_38_groups", AGE_38_RANGES, AGE_38_LEVELS),
            ("age_40_groups", AGE_40_RANGES, AGE_40_LEVELS)):
        age.declare(name, tuple(tuple(range(lo, hi + 1)) for lo, hi in ranges),
                    labels)

    hispanic = Axis("hispanic", HISPANIC_LEVELS)
    hispanic.alias("hispanic", "detailed")

    cenrace = Axis("cenrace", CENRACE_LEVELS)
    cenrace.alias("cenrace", "detailed")

    return Universe("dhc person", (relgq, sex, age, hispanic, cenrace))


PL94_PERSON = _pl94_person_universe()
PL94_UNIT = _pl94_unit_universe()
DHC_PERSON = _dhc_person_universe()


# ── schema: measurements ─────────────────────────────────────────────────────
# A canonical name lists the binning each resolved axis uses, in universe axis
# order; every axis it does not name is marginalized.  The raw NMF names order
# the axes differently and are matched verbatim.

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

# The unit histogram has a single axis and one DP query, occupied/vacant.  Its
# total is invariant at every level, so "total" has no released query at all
# and is served purely from evidence -- which the framework handles without a
# special case, since a one-cell block with an exact total is a point mass.
_UNIT_QUERIES = {
    "total": None,
    "h1": "detailed_dpq",
}

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

_NOTES = {
    ("pl94 person", "total"): "exact at the national level (total_con)",
    ("pl94 unit", "total"): "exact at every level (invariant); no released query",
    ("pl94 unit", "h1"): "conditioned on the exact unit total",
}


def _measurements(universe, queries):
    out = []
    for name, native in queries.items():
        note = _NOTES.get((universe.name, name), "")
        if name == "total":
            specs = tuple("*" for _ in universe.axes)
            out.append(Measurement(name, universe, specs, native, note))
        else:
            out.append(Measurement.parse(name, universe, native, note))
    return out


# ── schema: geography ────────────────────────────────────────────────────────
# Geocodes are DAS spine codes, not census GEOIDs.  The spine splits every
# state into a non-AIAN portion (prefix 0) and, where present, an AIAN portion
# (prefix 1); both are returned, flagged by `aian`, and rows sharing a geoid
# can be summed to get whole-geography estimates.

def _block_group_key(geocode):
    """Tabulation block group of a spine block geocode.

    The last 16 characters are the standard block GEOID: state(2) county(3)
    tract(6) blockgroup(1) block(4).  This is the atom map the framework needs
    to compute the meet of two incomparable geographic partitions, and it is
    exactly the kind of fact core cannot derive on its own.
    """
    return geocode[-16:][:12]


def _pl94_geographies():
    return (
        GeoLevel("us", native="US"),
        GeoLevel("state", native="State"),
        GeoLevel("county", native="County"),
        GeoLevel("tract", native="Tract", requires=("state",)),
        GeoLevel("block group", native="Block_Group", requires=("state",),
                 aliases=("block_group",),
                 note="DAS 'optimized block groups'; these are not tabulation "
                      "block groups, so geoid is left missing. Ask for "
                      "'tabulation block group' instead."),
        GeoLevel("tabulation block group", derive_from="block",
                 key=_block_group_key, requires=("state",),
                 aliases=("real block group", "real_block_group"),
                 note="incomparable with the spine's own block groups; "
                      "recovered by reading blocks and re-aggregating, which "
                      "is far more expensive than any direct level."),
        GeoLevel("block", native="Block", requires=("state",)),
    )


def _dhc_geographies():
    return (
        GeoLevel("us", native="US"),
        GeoLevel("state", native="State"),
        GeoLevel("county", native="County", requires=("state",)),
        # declared so asking for them reports why, rather than silently
        # answering with something coarser
        GeoLevel("tract", native="Tract", requires=("state",),
                 note="the DHC spine's lower levels are spine constructions "
                      "with no tabulation counterpart and are not released."),
        GeoLevel("block", native="Block", requires=("state",),
                 note="not released for DHC."),
    )


# ── evidence: what the constraint rows mean ──────────────────────────────────
# Each rule reads one geography's native constraint rows and says only what
# they assert.  Consequences -- pinning, pairing, bounding -- are the
# framework's job.

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

# The gqlevels axis the DHC hhgq bound rows use: one household level, then the
# 24 GQ types in relgq order.
_GQLEVEL_MEMBERS = (tuple(range(18)),) + tuple((i,) for i in range(18, 42))

# pl94_con is an 8 x 2 x 2 x 63 histogram over hhgq x votingage x hispanic x
# cenrace, so it resolves every DHC axis except sex.
_PL94_CON_SHAPE = (8, 2, 2, 63)


def _relgq_to_hhgq8():
    """relgq level -> its PL94 hhgq category, the coarsening pl94_con uses."""
    out = [0] * 42                                  # relgq 0-17 are households
    gq_classes = DHC_PERSON.axis("relgq").binning(
        "relship_and_eight_level_GQ").partition.groups[18:]
    for category, group in enumerate(gq_classes, start=1):
        for level in group:
            out[level] = category
    return tuple(out)


_PL94_CON_COARSENING = marginal_coarsening(DHC_PERSON, {
    "relgq": _relgq_to_hhgq8(),
    "age": tuple(0 if age < 18 else 1 for age in range(116)),
    "hispanic": (0, 1),
    "cenrace": tuple(range(63)),
})

_DHC_GQ_COARSENING = marginal_coarsening(DHC_PERSON, {
    "relgq": Partition(_GQLEVEL_MEMBERS, 42, "gqlevels"),
})

_PL94_HHGQ_COARSENING = marginal_coarsening(PL94_PERSON, {
    "hhgq": Partition(_singles(8), 8, "hhgq"),
})

_PL94_TOTAL_COARSENING = marginal_coarsening(PL94_PERSON, {})
_UNIT_TOTAL_COARSENING = marginal_coarsening(PL94_UNIT, {})


def _bounds(rows, coarsening, shape, name):
    lo = rows.get(f"{name}_lb_con")
    hi = rows.get(f"{name}_ub_con")
    if lo is None and hi is None:
        return ()
    return (BoundedHistogram(
        coarsening,
        lo=None if lo is None else np.asarray(lo, dtype=np.int64).reshape(shape),
        hi=None if hi is None else np.asarray(hi, dtype=np.int64).reshape(shape),
        source=f"{name}_lb_con/{name}_ub_con"),)


def pl94_person_evidence(rows, specs):
    """PL94 person: the national total, hhgq category bounds, nursing zeros."""
    out = []
    total = rows.get("total_con")
    if total is not None:
        out.append(ExactHistogram(
            _PL94_TOTAL_COARSENING, np.array(int(total[0])).reshape(1, 1, 1, 1),
            source="total_con"))
    out.extend(_bounds(rows, _PL94_HHGQ_COARSENING, (8, 1, 1, 1), "hhgq_total"))
    nurse = rows.get("nurse_nva_0_con")
    if nurse is not None and int(nurse[0]) == 0:
        # under-18 residents of nursing facilities
        out.append(ZeroRegion(
            Region(({3}, {0}, None, None)), source="nurse_nva_0_con"))
    return tuple(out)


def pl94_unit_evidence(rows, specs):
    """PL94 units: the housing-unit total, invariant at every level."""
    total = rows.get("total_con")
    if total is None:
        return ()
    return (ExactHistogram(
        _UNIT_TOTAL_COARSENING, np.array(int(total[0])).reshape(1),
        source="total_con"),)


def dhc_person_evidence(rows, specs):
    """DHC persons: GQ bounds, the relgq age rules, and the PL94 invariant."""
    out = list(_bounds(rows, _DHC_GQ_COARSENING, (25, 1, 1, 1, 1), "hhgq_total"))

    for name, value in rows.items():
        rule = specs.get(name)
        if rule is None or not rule[0].startswith("relgq") or not rule[0][5:].isdigit():
            continue
        forbidden = _AGE_ZERO_RULES.get(rule[2])
        if forbidden is None or int(value[0]) != 0:
            continue
        out.append(ZeroRegion(
            Region(({int(rule[0][5:])}, None, forbidden, None, None)), source=name))

    pl94 = rows.get("pl94_con")
    if pl94 is not None:
        # published PL94 table for the area: an invariant of the DHC run, and
        # the DHC histogram marginalizes onto it.  Sex is inserted as a
        # marginalized axis, since PL94 does not resolve it.
        values = np.asarray(pl94, dtype=np.int64).reshape(_PL94_CON_SHAPE)
        out.append(ExactHistogram(
            _PL94_CON_COARSENING, values[:, None, :, :, :], source="pl94_con"))
    return tuple(out)


# ── physical access ──────────────────────────────────────────────────────────

class NmfSource(Source):
    """Reads the Hive-partitioned parquet NMF releases.

    One scan serves both measurements and evidence for a selection, so the
    result is cached per (level, selection): the framework asks for
    measurements first and evidence second, and a 21 GB release should not be
    walked twice for that.  A product is a module-level singleton and outlives
    any one view, so the cache keeps only the few most recent selections.
    """

    CACHE_ENTRIES = 4

    def __init__(self, universe, dataset, spine_levels, evidence_rule,
                 default_root, needs_state=()):
        self.universe = universe
        self.dataset = dataset
        self.spine_levels = tuple(spine_levels)
        self.evidence_rule = evidence_rule
        self.default_root = default_root
        self.needs_state = frozenset(needs_state)
        self._cache = {}
        self._tract_xwalk = {}

    def levels(self):
        return self.spine_levels

    # ── selection handling ──────────────────────────────────────────────────

    def check_selection(self, level, selection):
        unknown = set(selection) - {"state", "county", "root"}
        if unknown:
            raise TypeError(f"unexpected selection key(s): {sorted(unknown)}")
        states = resolve_states(selection.get("state"))
        counties = resolve_counties(selection.get("county"))
        if level == "US" and states:
            raise ValueError('geography "us" does not take a state')
        if level != "US" and level in self.needs_state and not states:
            raise ValueError(f"geography {level!r} requires a state")
        if counties and level != "County":
            raise ValueError('county only applies to geography="county"')

    def _root(self, selection):
        return selection.get("root") or self.default_root

    def _selection_key(self, selection):
        return (self._root(selection),
                tuple(resolve_states(selection.get("state"))),
                tuple(resolve_counties(selection.get("county"))))

    # ── scanning ────────────────────────────────────────────────────────────

    def _scan(self, root, product, level):
        path = f"{root}/{self.dataset.format(product)}/{level}.parquet"
        # Partitions were written by different jobs: integer widths drift and
        # DPQuery partitions may lack the sign column entirely.
        return pl.scan_parquet(
            path,
            cast_options=pl.ScanCastOptions(integer_cast="upcast"),
            missing_columns="insert")

    def _collect(self, level, selection, natives):
        key = (level, self._selection_key(selection))
        cached = self._cache.get(key)
        natives = frozenset(natives)
        if cached is not None and natives <= cached["natives"]:
            return cached
        natives = natives | (cached["natives"] if cached else frozenset())

        root = self._root(selection)
        states = resolve_states(selection.get("state"))
        counties = resolve_counties(selection.get("county"))

        dpq, constraints, specs = [], {}, {}
        for product, fips_list in group_states_by_product(states):
            rows = self._collect_product(
                root, product, level, fips_list, counties, natives)
            for row in rows.filter(pl.col("sign").is_not_null()).iter_rows(named=True):
                constraints.setdefault(row["geocode"], {})[row["query_name"]] = row["value"]
                # a constraint's axis specs say what it constrains, and are the
                # same wherever the row appears
                specs.setdefault(
                    row["query_name"],
                    tuple(row[axis] for axis in self.universe.axis_names))
            dpq.append(rows.filter(pl.col("sign").is_null()).sort("query_name", "geocode"))

        entry = {"natives": natives, "dpq": dpq, "constraints": constraints,
                 "specs": specs}
        self._cache.pop(key, None)
        self._cache[key] = entry
        while len(self._cache) > self.CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))
        return entry

    def _collect_product(self, root, product, level, fips_list, counties, natives):
        lf = self._scan(root, product, level)
        if fips_list and level != "US":
            partitions = [f for fips in fips_list for f in (fips, fips + 100)]
            lf = lf.filter(pl.col("State").is_in(partitions))
        if counties:
            # the County partition value is the spine geocode, whose last four
            # digits are the county FIPS code; it is read as an integer
            lf = lf.filter(
                pl.col("County").cast(pl.Utf8).str.slice(-4).cast(pl.Int64)
                .is_in(list(counties)))
        # `sign` marks a constraint row.  A release that ships no constraint
        # partition at all lacks the column outright, which is not the same as
        # the column being null, and polars will not filter on what is absent.
        if "sign" not in lf.collect_schema().names():
            rows = lf.filter(pl.col("query_name").is_in(sorted(natives))).collect()
            return rows.with_columns(pl.lit(None, dtype=pl.Utf8).alias("sign"))
        keep = pl.col("sign").is_not_null()
        if natives:
            keep = keep | pl.col("query_name").is_in(sorted(natives))
        return lf.filter(keep).collect()

    # ── the Source interface ────────────────────────────────────────────────

    def read_measurements(self, level, selection, measurements):
        by_native = {m.native: m for m in measurements if m.native is not None}
        entry = self._collect(level, selection, by_native)
        checked = set()
        for frame in entry["dpq"]:
            for row in frame.iter_rows(named=True):
                measurement = by_native.get(row["query_name"])
                if measurement is None:
                    continue
                if measurement.name not in checked:
                    self._check_binning(row, measurement)
                    checked.add(measurement.name)
                yield MeasurementRow(row["geocode"], measurement.name,
                                     row["value"], row["variance"])

    def _check_binning(self, row, measurement):
        """Validate the catalog against the file's own axis spec columns.

        Compared as partitions rather than as names, since a release spells
        the same binning several ways ("detailed" and "h1" are one thing).
        This is the check that would catch a codebook drifting from a reissued
        release, so it runs once per measurement rather than not at all.
        """
        for axis, spec, declared in zip(
                self.universe.axes,
                (row[a] for a in self.universe.axis_names),
                measurement.binnings):
            if not axis.has(spec):
                raise ValueError(
                    f"{row['query_name']!r} at {row['geocode']!r} names an "
                    f"unknown {axis.name} binning {spec!r}")
            if axis.binning(spec).partition.groups != declared.partition.groups:
                raise ValueError(
                    f"{row['query_name']!r} at {row['geocode']!r} bins "
                    f"{axis.name} as {spec!r}, which is not the catalog's "
                    f"{declared.name!r}")

    def read_evidence(self, level, selection):
        entry = self._collect(level, selection, ())
        specs = entry["specs"]
        return {geo: self.evidence_rule(rows, specs)
                for geo, rows in entry["constraints"].items()}

    def geo_columns(self, level, selection, geo_keys):
        unique = pd.unique(np.asarray(geo_keys, dtype=object))
        xwalk = self._tracts(level, selection)
        geoid = {g: _geoid(level, g, xwalk) for g in unique}
        aian = {g: _is_aian(level, g) for g in unique}
        return {
            "geoid": pd.array([geoid[g] for g in geo_keys], dtype="string"),
            "geocode": pd.array(list(geo_keys), dtype="string"),
            "aian": pd.array([aian[g] for g in geo_keys], dtype="boolean"),
        }

    def _tracts(self, level, selection):
        """Map spine tract codes to standard 11-char tract GEOIDs.

        The spine renumbers tracts, but block geocodes embed the standard block
        GEOID in their last 16 characters, and spine tracts map 1:1 onto
        tabulation tracts; scanning one query's block geocodes recovers the
        correspondence.
        """
        if level != "Tract" or "Block" not in self.spine_levels:
            return {}
        key = self._selection_key(selection)
        if key in self._tract_xwalk:
            return self._tract_xwalk[key]

        root = self._root(selection)
        states = resolve_states(selection.get("state"))
        geocodes = []
        for product, fips_list in group_states_by_product(states):
            lf = self._scan(root, product, "Block")
            if fips_list:
                partitions = [f for fips in fips_list for f in (fips, fips + 100)]
                lf = lf.filter(pl.col("State").is_in(partitions))
            geocodes.extend(
                lf.filter(pl.col("query_name") == "detailed_dpq")
                .select("geocode").unique().collect()["geocode"])

        xwalk, conflicts = {}, set()
        for geocode in geocodes:
            spine_tract = geocode[:-18]
            real = geocode[-16:][:11]
            if xwalk.setdefault(spine_tract, real) != real:
                conflicts.add(spine_tract)
        for spine_tract in conflicts:
            xwalk[spine_tract] = None
        if conflicts:
            warnings.warn(
                f"{len(conflicts)} spine tracts span multiple tabulation "
                "tracts; their geoid is left <NA>.", stacklevel=2)
        self._tract_xwalk[key] = xwalk
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
        # digit duplicates the block code's first digit
        return real[:11] + real[12:]
    return None  # Block_Group: spine block groups are not tabulation ones


def _is_aian(level, geocode):
    if level == "US":
        return False
    return geocode.startswith("1")


# ── selection helpers ────────────────────────────────────────────────────────

def resolve_states(state):
    """FIPS codes for a state given as code, postal abbreviation, or name."""
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


def resolve_counties(county):
    if county is None:
        return []
    if isinstance(county, (str, int, np.integer)):
        county = [county]
    fips = []
    for c in county:
        text = str(c).strip()
        if not text.isdigit():
            raise ValueError(f"county must be a FIPS code, got {c!r}")
        if int(text) not in fips:
            fips.append(int(text))
    return fips


def group_states_by_product(states):
    """Split requested states into (product, fips_list) pairs.

    Puerto Rico is a separate NMF product and is routed automatically.
    """
    if not states:
        return [("US", [])]
    groups = []
    mainland = [f for f in states if f != 72]
    if mainland:
        groups.append(("US", mainland))
    if 72 in states:
        groups.append(("PR", [72]))
    return groups


# ── the products ─────────────────────────────────────────────────────────────

_PL94_SPINE = ("US", "State", "County", "Tract", "Block_Group", "Block")
_PL94_NEEDS_STATE = ("Tract", "Block_Group", "Block")

pl94_2020 = Product(
    "pl94 2020 person",
    PL94_PERSON,
    _measurements(PL94_PERSON, _PERSON_QUERIES),
    _pl94_geographies(),
    NmfSource(PL94_PERSON, "{}_Person_PL_PROD", _PL94_SPINE,
              pl94_person_evidence, DEFAULT_PL94_ROOT, _PL94_NEEDS_STATE),
    note="2020 PL94-171 person noisy measurements")

pl94_2020_units = Product(
    "pl94 2020 unit",
    PL94_UNIT,
    _measurements(PL94_UNIT, _UNIT_QUERIES),
    _pl94_geographies(),
    NmfSource(PL94_UNIT, "{}_Unit_PL_PROD", _PL94_SPINE,
              pl94_unit_evidence, DEFAULT_PL94_ROOT, _PL94_NEEDS_STATE),
    note="2020 PL94-171 housing-unit noisy measurements")

dhc_2020 = Product(
    "dhc 2020 person",
    DHC_PERSON,
    _measurements(DHC_PERSON, _DHC_QUERIES),
    _dhc_geographies(),
    NmfSource(DHC_PERSON, "{}_DHCP_PROD", ("US", "State", "County"),
              dhc_person_evidence, DEFAULT_DHC_ROOT, ("County",)),
    note="2020 DHC person noisy measurements; the product that carries sex and age")

PRODUCTS = {
    "pl94": pl94_2020,
    "pl94 units": pl94_2020_units,
    "dhc": dhc_2020,
}


# ── mirroring the DHC release ────────────────────────────────────────────────

def fetch_dhc(geography, state=None, county=None, *, root=DEFAULT_DHC_ROOT,
              overwrite=False):
    """Download DHC person NMF partitions into `root` for `dhc_2020` to read.

    The 2020 DHC NMF is public but large, so nothing is bundled and nothing is
    fetched implicitly: name the geographies you want and they are mirrored,
    preserving the release's own directory layout.  A state-level partition
    runs about 1.5 MB per state and a county-level one about 1.8 MB per county.

    Returns the list of local paths that now hold the requested partitions.
    """
    plan = dhc_2020.plan(geography)
    level = plan.read_level
    states = resolve_states(state)
    counties = resolve_counties(county)
    if level == "US":
        if states:
            raise ValueError('geography "us" does not take a state')
    elif not states:
        raise ValueError(f'geography "{geography}" requires a state')
    if counties and level != "County":
        raise ValueError('county only applies to geography="county"')

    paths = []
    for product, fips_list in group_states_by_product(states):
        dataset = dhc_2020.source.dataset.format(product)
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
        keys.extend(e.text for e in root.findall("s3:Contents/s3:Key", ns) if e.text)
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


# ── one-shot convenience ─────────────────────────────────────────────────────

def read(product, geography, queries, **kwargs):
    """Read measurements through a throwaway view.

    Convenient, and exactly as consistent as the library's original per-cell
    interface was: one call is one view, so values from two calls are
    independent.  Open a view when that matters, which is most of the time.
    """
    options = {k: kwargs.pop(k) for k in
               ("nonnegative", "apply_constraints", "family", "strategy")
               if k in kwargs}
    view = product.view(geography, **options, **kwargs)
    return view.query(queries) if isinstance(queries, str) else view.queries(queries)


def _normalize_table(table):
    t = str(table).strip().lower().rstrip("s")
    if t not in ("person", "unit"):
        raise ValueError(f'table must be "person" or "unit", got {table!r}')
    return t


def get_pl94(geography, queries, state=None, *, table="person", root=None,
             nonnegative=True, apply_constraints=True, real_block_groups=False):
    """Read PL94 NMF measurements through a throwaway view, tidily.

    A thin wrapper over `pl94_2020.view(...)` / `pl94_2020_units.view(...)`:
    one call is one view, so values from two separate calls are independent.
    Open a view directly (`census.pl94_2020.view(...)`) when several reads
    need to share latent cells.

    Parameters
    ----------
    geography : one of "us", "state", "county", "tract", "block group",
        "tabulation block group", "block".  Geographies below county require
        `state`.
    queries : measurement name or list of names; see
        `pl94_2020.measurements()`.
    state : state FIPS code (int or str), postal abbreviation, or full name;
        may be a list.  Puerto Rico is a separate NMF product and is routed
        automatically when requested.
    table : "person" or "unit".
    root : directory containing the `*_PL_PROD` parquet datasets; defaults to
        `DEFAULT_PL94_ROOT`.
    nonnegative : truncate every posterior at zero (true counts cannot be
        negative).
    apply_constraints : condition posteriors on the NMF constraint rows
        (exact totals, structural zeros, hhgq bounds).
    real_block_groups : only valid with `geography="block group"`.  Reads at
        the block level and sums block measurements up to real tabulation
        block groups instead of the DAS spine's "optimized" ones -- far more
        expensive than any direct level, since it reads every block in the
        requested area.

    Returns a pandas DataFrame with one row per cell: `geoid` (standard
    census GEOID where derivable, otherwise <NA>), `geocode` (raw spine
    code, or <NA> when `real_block_groups=True` since a summed row no longer
    corresponds to one spine geography), `aian` (True/False for AIAN-branch
    spine geographies, or <NA> when a summed row straddles both branches),
    `query`, one label column per histogram axis, `value` (`NoisyInt`, or
    `NoisyFloat` when `real_block_groups=True` since summing promotes it),
    and `variance` (the NMF measurement variance).
    """
    product = pl94_2020 if _normalize_table(table) == "person" else pl94_2020_units
    geography = str(geography).strip().lower()
    if real_block_groups:
        if geography not in ("block group", "block_group"):
            raise ValueError(
                'real_block_groups=True only applies to geography="block group"')
        geography = "tabulation block group"
    elif geography in ("block group", "block_group"):
        warnings.warn(
            "NMF block groups are DAS 'optimized block groups' and do not "
            "correspond to census tabulation block groups; geoid is left "
            "<NA>. Pass real_block_groups=True to recover real "
            "block-group totals by summing block-level measurements.",
            stacklevel=2)

    selection = {}
    if state is not None:
        selection["state"] = state
    if root is not None:
        selection["root"] = root
    view = product.view(geography, nonnegative=nonnegative,
                        apply_constraints=apply_constraints, **selection)
    return view.queries(queries).rename(columns={"measurement": "query"})


def get_dhc(geography, queries, state=None, county=None, *, root=None,
            nonnegative=True, apply_constraints=True):
    """Read DHC person NMF measurements through a throwaway view, tidily.

    This is the product that resolves sex and single-year age; PL94 has
    neither.  Its histogram is relgq x sex x age x hispanic x cenrace.  A
    thin wrapper over `dhc_2020.view(...)`: one call is one view, so values
    from two separate calls are independent.  Open a view directly
    (`census.dhc_2020.view(...)`) when several reads need to share latent
    cells -- for example `sex*hispanic` and `sex*age_38_groups` against the
    same area, which share the invariants `pl94_con` pins.

    Parameters
    ----------
    geography : "us", "state", or "county".  County requires `state`.
    queries : measurement name or list of names; see `dhc_2020.measurements()`.
        Note that `detailed` alone is 1,227,744 cells per geography.
    state : state FIPS code (int or str), postal abbreviation, or full name;
        may be a list.  Puerto Rico is a separate NMF product and is routed
        automatically when requested.
    county : county FIPS code (int or str) or list of them, to narrow a
        county-level read; the DHC release partitions by county, so this
        prunes the files that are read.
    root : directory holding the `*_DHCP_PROD` parquet datasets, as laid down
        by `fetch_dhc`; defaults to `DEFAULT_DHC_ROOT`.
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
    selection = {}
    if state is not None:
        selection["state"] = state
    if county is not None:
        selection["county"] = county
    if root is not None:
        selection["root"] = root
    view = dhc_2020.view(geography, nonnegative=nonnegative,
                         apply_constraints=apply_constraints, **selection)
    return view.queries(queries).rename(columns={"measurement": "query"})


def pl94_queries(table="person"):
    """Tabulate the queries available from `get_pl94` for a table.

    Returns a plain pandas DataFrame with one row per query: its name, the
    raw NMF query name, the number of cells per geography, and notes.
    """
    product = pl94_2020 if _normalize_table(table) == "person" else pl94_2020_units
    out = product.measurements()[["measurement", "native", "cells", "note"]]
    return out.rename(columns={
        "measurement": "query", "native": "nmf_query_name", "note": "notes"})


def dhc_queries():
    """Tabulate the queries available from `get_dhc`.

    Returns a plain pandas DataFrame with one row per query: its name, the
    raw NMF query name, the number of cells per geography, and whether the
    query resolves the sex axis.
    """
    df = dhc_2020.measurements()
    out = df[["measurement", "native", "cells"]].rename(
        columns={"measurement": "query", "native": "nmf_query_name"})
    out["has_sex"] = (df["sex"] != "*").to_numpy()
    return out


# Table P12 ("Sex by Age") row order after the Total/Male/Female header
# rows, as published (checked against the Census API's `groups/P12.json`
# for the 2020 DHC release): (label, query, age levels). `query` is a
# `dhc_queries()` name and `age` is the set of that query's age-axis level
# labels to sum -- the coarsest query whose own binning reproduces the
# published bucket exactly, so the reconstructed measurement variance
# matches what was actually released for that shape wherever possible.
# "sex*age_38_groups" binning matches every published row except three:
# it has no way to split its single 20-24 bucket into P12's 20 years/21
# years/22-24 years, since the DP mechanism never measured that split at
# any coarser resolution than single-year age. `query=None` marks those
# three rows (and their female mirrors) as unsupported rather than falling
# back to "detailed": reconstructing them means summing ~5,000 detailed
# cells marginalized over relgq/hispanic/cenrace, and empirically that sum's
# posterior comes out nowhere near its own observed value (off by
# thousands, in either direction depending on `nonnegative`) -- most of
# those cells fall in PL94 constraint blocks with 3+ still-free cells, where
# the block total is only an upper bound and each cell is otherwise modeled
# independently, so summing thousands of them doesn't average out the way a
# tight joint reconstruction would. There is no other DHC query with
# single-year age resolution, so these three rows have no reliable
# reconstruction from the NMF at all right now.
_P12_AGE_BUCKETS = (
    ("Under 5 years", "sex*age_38_groups", ("0", "1", "2", "3", "4")),
    ("5 to 9 years", "sex*age_38_groups", ("5", "6", "7", "8", "9")),
    ("10 to 14 years", "sex*age_38_groups", ("10", "11", "12", "13", "14")),
    ("15 to 17 years", "sex*age_38_groups", ("15-17",)),
    ("18 and 19 years", "sex*age_38_groups", ("18-19",)),
    ("20 years", None, ("20",)),
    ("21 years", None, ("21",)),
    ("22 to 24 years", None, ("22", "23", "24")),
    ("25 to 29 years", "sex*age_38_groups", ("25-29",)),
    ("30 to 34 years", "sex*age_38_groups", ("30-34",)),
    ("35 to 39 years", "sex*age_38_groups", ("35-39",)),
    ("40 to 44 years", "sex*age_38_groups", ("40-44",)),
    ("45 to 49 years", "sex*age_38_groups", ("45-49",)),
    ("50 to 54 years", "sex*age_38_groups", ("50-54",)),
    ("55 to 59 years", "sex*age_38_groups", ("55-59",)),
    ("60 and 61 years", "sex*age_38_groups", ("60-61",)),
    ("62 to 64 years", "sex*age_38_groups", ("62-64",)),
    ("65 and 66 years", "sex*age_38_groups", ("65-66",)),
    ("67 to 69 years", "sex*age_38_groups", ("67-69",)),
    ("70 to 74 years", "sex*age_38_groups", ("70-74",)),
    ("75 to 79 years", "sex*age_38_groups", ("75-79",)),
    ("80 to 84 years", "sex*age_38_groups", ("80-84",)),
    ("85 years and over", "sex*age_38_groups",
     ("85-89", "90-94", "95-99", "100-104", "105-109", "110-115")),
)

_P12_UNSUPPORTED_REASON = (
    "no DHC query workload reconstructs this cell reliably: only the "
    "\"detailed\" query has single-year age resolution, and summing the "
    "thousands of detailed cells this bucket marginalizes over produces a "
    "posterior far from its own observed value (most of those cells fall "
    "in PL94 constraint blocks with 3+ free cells, where the block total "
    "is only an upper bound rather than a tight joint constraint -- see "
    "dhc_table_variables()'s docstring)"
)


def dhc_table_variables(table="P12"):
    """Census variable-code -> `get_dhc` query/filter crosswalk for a published table.

    Maps literal `tidycensus`-style variable codes (e.g. `"P12_002N"`) from
    a published decennial table to the `get_dhc` query name and per-axis
    filter levels that reconstruct that cell from the actual NMF
    measurements. This is what lets a caller pass Census's own variable
    codes instead of naming a query plus axis filters directly -- see
    `_P12_AGE_BUCKETS` above for how each row picks its query.

    Only `table="P12"` (Sex by Age) is implemented so far; other published
    tables would each need their own row-to-query mapping worked out and
    checked the same way before being added here. Six of P12's 49 codes
    (Male/Female 20 years, 21 years, and 22 to 24 years) aren't included
    either, for a different reason: see `_P12_AGE_BUCKETS` and
    `unsupported_dhc_table_variables()`.

    Returns a dict keyed by variable code. Each value is a dict with a
    `"query"` entry (a `dhc_queries()` name), a `"label"` entry (the
    published row label), and zero or more `axis: (level, ...)` entries --
    the cells summed for that code are those matching every listed axis's
    level set; axes not listed are summed over entirely. Multiple levels
    for one axis mean summing several NMF cells to reconstruct one wider
    published bucket.
    """
    if table != "P12":
        raise ValueError(f'dhc_table_variables only supports table="P12" (got "{table}")')

    variables = {
        "P12_001N": {"query": "sex*hispanic", "label": "Total"},
        "P12_002N": {"query": "sex*hispanic", "label": "Male", "sex": ("male",)},
        "P12_026N": {"query": "sex*hispanic", "label": "Female", "sex": ("female",)},
    }
    for sex, total_code in (("male", 2), ("female", 26)):
        for i, (label, query, age) in enumerate(_P12_AGE_BUCKETS):
            if query is None:
                continue
            code = f"P12_{total_code + 1 + i:03d}N"
            variables[code] = {
                "query": query, "label": f"{sex.capitalize()}: {label}",
                "sex": (sex,), "age": age,
            }
    return variables


def unsupported_dhc_table_variables(table="P12"):
    """Codes `dhc_table_variables()` deliberately omits, with why.

    Returns a dict keyed by variable code (a subset of the table's full
    code range that `dhc_table_variables()` doesn't return) to a
    human-readable reason string. Meant for producing a clear error message
    when a caller asks for one of these, rather than an ambiguous "unknown
    code".
    """
    if table != "P12":
        raise ValueError(
            f'unsupported_dhc_table_variables only supports table="P12" (got "{table}")')

    unsupported = {}
    for sex, total_code in (("male", 2), ("female", 26)):
        for i, (label, query, age) in enumerate(_P12_AGE_BUCKETS):
            if query is not None:
                continue
            code = f"P12_{total_code + 1 + i:03d}N"
            unsupported[code] = _P12_UNSUPPORTED_REASON
    return unsupported
