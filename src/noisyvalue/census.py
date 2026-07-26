"""Interface to the 2020 Census PL94-171 Noisy Measurement File (NMF).

The NMF is the raw output of the Census Bureau's TopDown disclosure-avoidance
algorithm: for every geography on the DAS geographic spine, a set of
independently noised queries (discrete Gaussian mechanism) over the PL94
person and housing-unit histograms, plus the exact constraints the Bureau's
own post-processing used.  `get_pl94` reads the Hive-partitioned parquet
release and returns a tidy pandas DataFrame whose `value` column holds
`NoisyInt` posteriors.

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
  category totals, applied by truncating the affected posteriors.
- All counts are additionally truncated at zero when ``nonnegative=True``.

Note that the NMF is *pre* post-processing: noisy measurements do not add up
across geography levels or across queries, and they differ from the published
PL94 tables.  That inconsistency is real information about the noise and is
exactly what the returned posteriors quantify.

Geocodes are DAS spine codes, not census GEOIDs.  The spine splits every
state into a non-AIAN portion (prefix ``0``) and, where present, an AIAN
portion (prefix ``1``); both are returned, flagged by the ``aian`` column,
and rows sharing a ``geoid`` can be summed (NoisyInt arithmetic) to get
whole-geography estimates.  Standard GEOIDs are derivable for every level
except block groups, whose spine version ("optimized block groups") does not
correspond to tabulation block groups; pass ``real_block_groups=True`` to
``get_pl94`` to recover real block-group totals instead, by reading at the
block level (where GEOIDs are exact) and summing up to each block's real
block group.
"""

import itertools
import math
import warnings

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

# Per-axis label sets keyed by the spec strings that appear in the NMF's
# hhgq/votingage/hispanic/cenrace/h1 columns.
_AXIS_LEVELS = {
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

_PERSON_CONSTRAINTS = (
    "total_con",
    "nurse_nva_0_con",
    "hhgq_total_lb_con",
    "hhgq_total_ub_con",
)
_UNIT_CONSTRAINTS = ("total_con",)

_GEO_LEVELS = {
    "us": "US",
    "state": "State",
    "county": "County",
    "tract": "Tract",
    "block group": "Block_Group",
    "block_group": "Block_Group",
    "block": "Block",
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
            specs = _person_axis_specs(name)
            cells = math.prod(
                len(_AXIS_LEVELS[ax][spec]) for ax, spec in zip(_PERSON_AXES, specs)
            )
            note = "exact at the national level (total_con)" if name == "total" else ""
            rows.append((name, raw, cells, note))
    else:
        rows.append(("total", "total_con", 1, "exact at every level (invariant)"))
        rows.append(("h1", "detailed_dpq", 2,
                     "conditioned on the exact total when apply_constraints=True"))
    return pd.DataFrame(rows, columns=["query", "nmf_query_name", "cells", "notes"])


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
    level = _normalize_geography(geography)
    axes = _PERSON_AXES if table == "person" else _UNIT_AXES
    qmap = _PERSON_QUERIES if table == "person" else _UNIT_QUERIES
    queries = _normalize_queries(queries, qmap)

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
        _load_product(product, fips_list, fetch_level, table, axes, qmap,
                      queries, root, nonnegative, apply_constraints)
        for product, fips_list in products
    ]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if level == "Block_Group" and real_block_groups:
        frame = _aggregate_real_block_groups(frame, axes)
    return frame


# ── input normalization ──────────────────────────────────────────────────────

def _normalize_table(table):
    t = str(table).strip().lower().rstrip("s")
    if t not in ("person", "unit"):
        raise ValueError(f'table must be "person" or "unit", got {table!r}')
    return t


def _normalize_geography(geography):
    g = str(geography).strip().lower()
    if g not in _GEO_LEVELS:
        raise ValueError(
            f"unknown geography {geography!r}; expected one of "
            f"{sorted(set(_GEO_LEVELS))}")
    return _GEO_LEVELS[g]


def _normalize_queries(queries, qmap):
    if isinstance(queries, str):
        queries = [queries]
    result = []
    for q in queries:
        key = str(q).strip().lower().replace(" ", "")
        if key.endswith("_dpq"):
            key = key[:-4]
        if key not in qmap:
            raise ValueError(
                f"unknown query {q!r}; expected one of {sorted(qmap)}")
        if key not in result:
            result.append(key)
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


def _person_axis_specs(canonical):
    """Axis spec strings for a canonical person query name."""
    if canonical == "detailed":
        return ("detailed", "detailed", "detailed", "detailed")
    parts = canonical.split("*") if canonical != "total" else []
    return tuple(ax if ax in parts else "*" for ax in _PERSON_AXES)


# ── data access ──────────────────────────────────────────────────────────────

def _scan(root, product, table, level):
    kind = "Person" if table == "person" else "Unit"
    path = f"{root}/{product}_{kind}_PL_PROD/{level}.parquet"
    # Partitions were written by different jobs: integer widths drift and
    # Constraint partitions may lack the sign column entirely.
    return pl.scan_parquet(
        path,
        cast_options=pl.ScanCastOptions(integer_cast="upcast"),
        missing_columns="insert",
    )


def _collect_rows(root, product, table, level, fips_list, query_names):
    lf = _scan(root, product, table, level)
    if fips_list and level != "US":
        partitions = [f for fips in fips_list for f in (fips, fips + 100)]
        lf = lf.filter(pl.col("State").is_in(partitions))
    lf = lf.filter(pl.col("query_name").is_in(list(query_names)))
    return lf.collect()


def _tract_geoids(root, product, table, fips_list):
    """Map spine tract codes to standard 11-char tract GEOIDs.

    The spine renumbers tracts, but block geocodes embed the standard block
    GEOID in their last 16 characters, and spine tracts map 1:1 onto
    tabulation tracts; scanning one query's block geocodes recovers the
    correspondence.
    """
    lf = _scan(root, product, table, "Block")
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

def _exact_count(obs, pinned):
    return NoisyInt(int(obs), DerivedNode(sp.Integer(int(pinned))))


def _noisy_count(obs, sd, lb=None, ub=None):
    """A NoisyInt whose posterior is a (truncated) discrete Gaussian at obs.

    The posterior is encoded directly as obs - eps rather than through a
    latent-plus-constraint pair, so building and sampling large collections
    of census cells avoids the symbolic solve step entirely.
    """
    obs = int(obs)
    if lb is not None and ub is not None and lb == ub:
        return _exact_count(obs, lb)
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


def _conditioned_unit_pair(obs_vac, obs_occ, variance, total, nonnegative):
    """Occupied/vacant posteriors conditioned on the exact unit total.

    With independent measurements obs_vac = vac + e1 and obs_occ = occ + e2
    (equal variance v) and the invariant vac + occ = total, the posterior of
    vac over the integers is a discrete Gaussian centered at
    (obs_vac - obs_occ + total)/2 with variance v/2, and occ = total - vac
    exactly (perfect negative correlation).
    """
    obs_vac, obs_occ, total = int(obs_vac), int(obs_occ), int(total)
    scale = sp.sqrt(sp.Float(float(variance)) / 2)
    delta = sp.Rational(obs_vac + obs_occ - total, 2)
    if nonnegative:
        node = TruncatedDiscreteGaussianNode.create(
            loc=delta, scale=scale,
            low=sp.Integer(obs_vac - total), high=sp.Integer(obs_vac))
    else:
        node = DiscreteGaussianNode.create(loc=delta, scale=scale)
    vac_root = DerivedNode(sp.Integer(obs_vac) - node.expr, deps=(node,))
    vac = NoisyInt(obs_vac, vac_root)
    occ_root = DerivedNode(sp.Integer(total) - vac_root.expr, deps=(vac_root,))
    occ = NoisyInt(obs_occ, occ_root)
    return vac, occ


# ── assembly ─────────────────────────────────────────────────────────────────

def _load_product(product, fips_list, level, table, axes, qmap, queries,
                  root, nonnegative, apply_constraints):
    raw_names = {qmap[q] for q in queries if qmap[q] is not None}
    exact_total = "total" in queries and (
        table == "unit" or (table == "person" and level == "US"))
    con_names = _PERSON_CONSTRAINTS if table == "person" else _UNIT_CONSTRAINTS

    need_constraints = apply_constraints or exact_total
    wanted = set(raw_names) | (set(con_names) if need_constraints else set())
    rows = _collect_rows(root, product, table, level, fips_list, wanted)

    constraints = {}
    for r in rows.filter(pl.col("sign").is_not_null()).iter_rows(named=True):
        constraints[(r["geocode"], r["query_name"])] = r["value"]

    dpq = rows.filter(pl.col("sign").is_null()).sort("query_name", "geocode")

    total_cells = int(dpq.select(pl.col("value").list.len().sum()).item() or 0)
    if total_cells > _MATERIALIZE_WARN_CELLS:
        warnings.warn(
            f"materializing {total_cells:,} noisy values; consider narrowing "
            "the geography, state, or query selection.", stacklevel=3)

    tract_xwalk = {}
    if level == "Tract":
        tract_xwalk = _tract_geoids(root, product, table, fips_list)

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
                 _exact_count(total, total))

    for r in dpq.iter_rows(named=True):
        canonical = canonical_by_raw[r["query_name"]]
        if table == "unit" and canonical == "h1":
            _emit_unit_h1(emit, r, constraints, nonnegative, apply_constraints)
        else:
            _emit_person_query(emit, r, canonical, constraints,
                               nonnegative, apply_constraints)

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
        vac, occ = _conditioned_unit_pair(
            obs_vac, obs_occ, variance, int(total[0]), nonnegative)
    else:
        sd = math.sqrt(variance)
        lb = 0 if nonnegative else None
        vac = _noisy_count(obs_vac, sd, lb=lb)
        occ = _noisy_count(obs_occ, sd, lb=lb)
    emit(geocode, "h1", ("vacant",), variance, vac)
    emit(geocode, "h1", ("occupied",), variance, occ)


def _emit_person_query(emit, row, canonical, constraints,
                       nonnegative, apply_constraints):
    geocode = row["geocode"]
    variance = float(row["variance"])
    sd = math.sqrt(variance)
    values = row["value"]
    specs = tuple(row[ax] for ax in _PERSON_AXES)
    axis_labels = [_AXIS_LEVELS[ax][spec]
                   for ax, spec in zip(_PERSON_AXES, specs)]

    sizes = tuple(len(labels) for labels in axis_labels)
    if math.prod(sizes) != len(values):
        raise ValueError(
            f"query {row['query_name']!r} at {geocode!r}: "
            f"{len(values)} cells do not match axis sizes {sizes}")

    base_lb = 0 if nonnegative else None

    hhgq_bounds = None
    structural_zero = False
    if apply_constraints:
        hhgq_bounds = _hhgq_axis_bounds(geocode, specs, constraints)
        nurse = constraints.get((geocode, "nurse_nva_0_con"))
        structural_zero = (
            canonical == "detailed"
            and nurse is not None and int(nurse[0]) == 0)

    for flat, idx in enumerate(itertools.product(*(range(s) for s in sizes))):
        obs = int(values[flat])
        labels = tuple(axis_labels[a][i] for a, i in enumerate(idx))
        if structural_zero and idx[0] == 3 and idx[1] == 0:
            # under-18 residents of nursing facilities: exactly zero
            emit(geocode, canonical, labels, variance, _exact_count(obs, 0))
            continue
        lb, ub = base_lb, None
        if hhgq_bounds is not None:
            g_lb, g_ub = hhgq_bounds[idx[0]]
            ub = g_ub
            if g_lb is not None:
                lb = max(lb or 0, g_lb)
        emit(geocode, canonical, labels, variance,
             _noisy_count(obs, sd, lb=lb, ub=ub))


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
