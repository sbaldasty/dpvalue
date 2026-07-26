import math
import os

import numpy as np
import polars as pl
import pytest
import sympy as sp

from noisyvalue.census import (
    AGE_3_RANGES,
    AGE_10_RANGES,
    AGE_38_RANGES,
    AGE_40_RANGES,
    AGE_LEVELS,
    CENRACE_LEVELS,
    DEFAULT_DHC_ROOT,
    DEFAULT_ROOT,
    GQ_CONSTR_LEVELS,
    HHGQ_LEVELS,
    RELGQ_4_LEVELS,
    RELGQ_LEVELS,
    RELSHIP_GQ8_LEVELS,
    RELSHIP_LEVELS,
    SEX_LEVELS,
    _DHC_PERSON,
    _RELGQ_GROUPS,
    _axis_specs,
    _person_axis_specs,
    dhc_queries,
    get_dhc,
    get_pl94,
    pl94_queries,
)
from noisyvalue.core import NoisyInt, sample_noisy_values
from noisyvalue.graph import TruncatedDiscreteGaussianNode


# ── codebook ─────────────────────────────────────────────────────────────────

def test_cenrace_codebook_has_63_distinct_levels_in_combination_order():
    assert len(CENRACE_LEVELS) == 63
    assert len(set(CENRACE_LEVELS)) == 63
    assert CENRACE_LEVELS[:6] == ("white", "black", "aian", "asian", "nhopi", "sor")
    assert CENRACE_LEVELS[6] == "white-black"
    assert CENRACE_LEVELS[-1] == "white-black-aian-asian-nhopi-sor"


def test_person_axis_specs_marginalize_unnamed_axes():
    assert _person_axis_specs("total") == ("*", "*", "*", "*")
    assert _person_axis_specs("detailed") == ("detailed",) * 4
    assert _person_axis_specs("hispanic*cenrace") == ("*", "*", "hispanic", "cenrace")
    assert _person_axis_specs("hhgq") == ("hhgq", "*", "*", "*")


def test_pl94_queries_cell_counts():
    person = pl94_queries("person")
    assert dict(zip(person["query"], person["cells"]))["detailed"] == 2016
    unit = pl94_queries("unit")
    assert set(unit["query"]) == {"total", "h1"}


# ── DHC codebook ─────────────────────────────────────────────────────────────

def test_dhc_relgq_codebook_is_18_household_levels_plus_24_gq_types():
    assert len(RELGQ_LEVELS) == 42
    assert len(set(RELGQ_LEVELS)) == 42
    assert RELGQ_LEVELS[:18] == RELSHIP_LEVELS
    assert RELGQ_LEVELS[0] == "householder_alone"
    assert RELGQ_LEVELS[27] == "nursing_facility"
    assert RELGQ_LEVELS[33] == "college_housing"
    # the 8-level-GQ variant keeps the relationship levels and collapses the
    # 24 GQ types into the 7 PL94 major types
    assert RELSHIP_GQ8_LEVELS == RELSHIP_LEVELS + HHGQ_LEVELS[1:]
    assert len(RELSHIP_GQ8_LEVELS) == 25


def test_dhc_relgq_groupings_partition_the_42_levels():
    for spec, groups in _RELGQ_GROUPS.items():
        flat = sorted(i for group in groups for i in group)
        assert flat == list(range(42)), spec
    assert len(_RELGQ_GROUPS["relgq_4_groups"]) == len(RELGQ_4_LEVELS)
    assert len(_RELGQ_GROUPS["gq_constr_groups"]) == len(GQ_CONSTR_LEVELS)
    assert len(_RELGQ_GROUPS["relship_and_eight_level_GQ"]) == 25
    # the household/GQ boundary sits between relgq 17 and 18 in every grouping
    householders, others, institutional, noninstitutional = \
        _RELGQ_GROUPS["relgq_4_groups"]
    assert householders == (0, 1)
    assert others == tuple(range(2, 18))
    assert institutional + noninstitutional == tuple(range(18, 42))


def test_dhc_age_groupings_partition_single_years_zero_to_115():
    for ranges, size in ((AGE_3_RANGES, 3), (AGE_10_RANGES, 10),
                         (AGE_38_RANGES, 38), (AGE_40_RANGES, 40)):
        assert len(ranges) == size
        covered = [age for lo, hi in ranges for age in range(lo, hi + 1)]
        assert covered == list(range(116))
    assert len(AGE_LEVELS) == 116
    assert AGE_LEVELS[0] == "0" and AGE_LEVELS[-1] == "115"


def test_dhc_axis_specs_marginalize_unnamed_axes():
    assert _axis_specs(_DHC_PERSON, "sex*hispanic") == \
        ("*", "sex", "*", "hispanic", "*")
    assert _axis_specs(_DHC_PERSON, "sex*age_38_groups") == \
        ("*", "sex", "age_38_groups", "*", "*")
    assert _axis_specs(_DHC_PERSON, "detailed") == ("detailed",) * 5


def test_dhc_queries_cell_counts_and_sex_coverage():
    frame = dhc_queries()
    cells = dict(zip(frame["query"], frame["cells"]))
    assert cells["detailed"] == 1_227_744
    assert cells["sex*hispanic"] == 4
    assert cells["sex*age_38_groups"] == 2 * 38
    assert cells["gq_constr_groups*age_10_groups"] == 60
    with_sex = set(frame[frame["has_sex"]]["query"])
    assert "sex*hispanic" in with_sex
    assert "popSehsdTargetsRelship" not in with_sex


# ── truncated discrete gaussian ──────────────────────────────────────────────

def test_truncated_discrete_gaussian_respects_bounds():
    rng = np.random.default_rng(7)
    node = TruncatedDiscreteGaussianNode.create(loc=0, scale=5, low=-3, high=2)
    draws = node.sample(rng, size=4000)
    assert draws.min() >= -3
    assert draws.max() <= 2


def test_truncated_discrete_gaussian_point_mass():
    rng = np.random.default_rng(7)
    node = TruncatedDiscreteGaussianNode.create(loc=0, scale=5, low=4, high=4)
    draws = node.sample(rng, size=100)
    assert np.all(draws == 4)


def test_truncated_discrete_gaussian_window_far_from_loc():
    rng = np.random.default_rng(7)
    node = TruncatedDiscreteGaussianNode.create(loc=0, scale=2, low=1000, high=1010)
    draws = node.sample(rng, size=200)
    assert draws.min() >= 1000
    assert draws.max() <= 1010
    # mass should pile up at the boundary nearest loc
    assert np.median(draws) == 1000


def test_observe_posterior_centers_on_observation():
    from noisyvalue.graph import DiscreteGaussianNode

    node = DiscreteGaussianNode.create(loc=0, scale=3)
    value = NoisyInt.observe(100, node)
    batch = value.sample(4000, rng=11)
    assert int(value) == 100
    assert abs(batch.mean() - 100) < 0.5
    assert np.all(batch.draws == batch.draws.astype(int))


# ── synthetic NMF fixture ────────────────────────────────────────────────────

VT_COUNTY = "050100011"  # spine code for county FIPS 50001


def _write(path, frame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.write_parquet(path)


@pytest.fixture
def nmf_root(tmp_path):
    root = str(tmp_path / "nmf")
    person_axes = {"hhgq": "*", "votingage": "*", "hispanic": "*", "cenrace": "*"}

    detailed = list(range(1, 2017))
    dpq = pl.DataFrame({
        "geocode": [VT_COUNTY] * 3,
        "query_name": ["total_dpq", "hhgq_dpq", "detailed_dpq"],
        "hhgq": ["*", "hhgq", "detailed"],
        "votingage": ["*", "*", "detailed"],
        "hispanic": ["*", "*", "detailed"],
        "cenrace": ["*", "*", "detailed"],
        "query_shape": [[1, 1, 1, 1], [8, 1, 1, 1], [8, 2, 2, 63]],
        "value": [[5000], [4000, 3, 0, 40, 0, 200, -4, 30], detailed],
        "variance": pl.Series([100.0, 100.0, 4.0], dtype=pl.Float32),
    })
    _write(f"{root}/US_Person_PL_PROD/County.parquet/DPQuery/State=50/part-0.parquet", dpq)

    con = pl.DataFrame({
        "geocode": [VT_COUNTY] * 3,
        "query_name": ["hhgq_total_lb_con", "hhgq_total_ub_con", "nurse_nva_0_con"],
        "sign": [">=", "<=", "="],
        "hhgq": ["hhgq", "hhgq", "gqNursingTotal"],
        "votingage": ["*", "*", "nonvoting"],
        "hispanic": ["*", "*", "*"],
        "cenrace": ["*", "*", "*"],
        "query_shape": [[8, 1, 1, 1], [8, 1, 1, 1], [1, 1, 1, 1]],
        "value": [[0, 5, 0, 20, 0, 0, 0, 0], [6000] * 8, [0]],
        "variance": pl.Series([0.0, 0.0, 0.0], dtype=pl.Float32),
    })
    _write(f"{root}/US_Person_PL_PROD/County.parquet/Constraint/State=50/part-0.parquet", con)

    unit_dpq = pl.DataFrame({
        "geocode": [VT_COUNTY],
        "query_name": ["detailed_dpq"],
        "h1": ["detailed"],
        "query_shape": [[2]],
        "value": [[30, 80]],
        "variance": pl.Series([25.0], dtype=pl.Float32),
    })
    _write(f"{root}/US_Unit_PL_PROD/County.parquet/DPQuery/State=50/part-0.parquet", unit_dpq)

    unit_con = pl.DataFrame({
        "geocode": [VT_COUNTY],
        "query_name": ["total_con"],
        "sign": ["="],
        "h1": ["*"],
        "query_shape": [[1]],
        "value": [[100]],
        "variance": pl.Series([0.0], dtype=pl.Float32),
    })
    _write(f"{root}/US_Unit_PL_PROD/County.parquet/Constraint/State=50/part-0.parquet", unit_con)
    return root


def test_get_pl94_tidy_shape_and_geoids(nmf_root):
    df = get_pl94("county", ["total", "hhgq"], state="VT", root=nmf_root)
    assert list(df.columns) == [
        "geoid", "geocode", "aian", "query",
        "hhgq", "votingage", "hispanic", "cenrace", "value", "variance"]
    assert len(df) == 9
    assert set(df["geoid"]) == {"50001"}
    assert not df["aian"].any()
    hhgq_rows = df[df["query"] == "hhgq"]
    assert list(hhgq_rows["hhgq"]) == list(HHGQ_LEVELS)
    assert [int(v) for v in hhgq_rows["value"]] == [4000, 3, 0, 40, 0, 200, -4, 30]


def test_get_pl94_applies_hhgq_bounds(nmf_root):
    df = get_pl94("county", "hhgq", state=50, root=nmf_root)
    correctional = df[df["hhgq"] == "correctional"]["value"].array[0]
    draws = correctional.sample(2000, rng=3).draws
    # obs 3 with exact lower bound 5: posterior lives in [5, 6000]
    assert draws.min() >= 5
    military = df[df["hhgq"] == "military"]["value"].array[0]
    draws = military.sample(2000, rng=3).draws
    # obs -4: nonnegativity truncates the posterior at zero
    assert draws.min() >= 0


def test_get_pl94_nonnegative_false_allows_negative_posterior(nmf_root):
    df = get_pl94("county", "hhgq", state=50, root=nmf_root,
                  nonnegative=False, apply_constraints=False)
    military = df[df["hhgq"] == "military"]["value"].array[0]
    draws = military.sample(2000, rng=3).draws
    assert draws.min() < 0


def test_get_pl94_structural_zero_pins_nursing_minors(nmf_root):
    df = get_pl94("county", "detailed", state="VT", root=nmf_root)
    assert len(df) == 2016
    slab = df[(df["hhgq"] == "nursing") & (df["votingage"] == "under_18")]
    assert len(slab) == 126
    for value in slab["value"]:
        assert np.all(value.sample(20, rng=5).draws == 0)
    other = df[(df["hhgq"] == "nursing") & (df["votingage"] == "voting_age")]
    assert any(value.sample(20, rng=5).draws.std() > 0 for value in other["value"])


def test_get_pl94_unit_conditioning_matches_exact_total(nmf_root):
    df = get_pl94("county", ["h1", "total"], state="VT", table="unit", root=nmf_root)
    vac = df[df["h1"] == "vacant"]["value"].array[0]
    occ = df[df["h1"] == "occupied"]["value"].array[0]
    total = df[df["query"] == "total"]["value"].array[0]
    assert int(total) == 100

    batch_vac, batch_occ = sample_noisy_values(vac, occ, n=500, rng=9)
    assert np.all(batch_vac.draws + batch_occ.draws == 100)
    # posterior center is (obs_vac - obs_occ + total) / 2 = 25
    assert abs(batch_vac.mean() - 25) < 0.5


def test_get_pl94_unit_without_constraints_stays_independent(nmf_root):
    df = get_pl94("county", "h1", state="VT", table="unit", root=nmf_root,
                  apply_constraints=False)
    vac = df[df["h1"] == "vacant"]["value"].array[0]
    occ = df[df["h1"] == "occupied"]["value"].array[0]
    batch_vac, batch_occ = sample_noisy_values(vac, occ, n=500, rng=9)
    sums = batch_vac.draws + batch_occ.draws
    assert len(np.unique(sums)) > 1
    assert abs(batch_vac.mean() - 30) < 1.0


@pytest.fixture
def block_nmf_root(tmp_path):
    root = str(tmp_path / "nmf")

    def block_geocode(bg, block):
        # "0" (non-AIAN branch) + state(2) + county(3) + tract(6) + bg(1) + block(4).
        # bg must equal block's own leading digit, matching real NMF encoding.
        return f"0{50:02d}001000100{bg}{block}"

    # Blocks 1001 and 1002 share real block group 500010001001 (bg digit "1");
    # block 2001 falls in a different real block group (bg digit "2").
    geocodes = [block_geocode(1, "1001"), block_geocode(1, "1002"),
                block_geocode(2, "2001")]
    dpq = pl.DataFrame({
        "geocode": geocodes,
        "query_name": ["total_dpq"] * 3,
        "hhgq": ["*"] * 3,
        "votingage": ["*"] * 3,
        "hispanic": ["*"] * 3,
        "cenrace": ["*"] * 3,
        "query_shape": [[1, 1, 1, 1]] * 3,
        "value": [[10], [20], [5]],
        "variance": pl.Series([100.0, 100.0, 100.0], dtype=pl.Float32),
    })
    _write(f"{root}/US_Person_PL_PROD/Block.parquet/DPQuery/State=50/part-0.parquet", dpq)

    # A constraint row purely so the "sign" column exists in the scanned
    # schema; its geocode/query don't match anything "total" consults.
    con = pl.DataFrame({
        "geocode": [geocodes[0]],
        "query_name": ["nurse_nva_0_con"],
        "sign": ["="],
        "hhgq": ["gqNursingTotal"],
        "votingage": ["nonvoting"],
        "hispanic": ["*"],
        "cenrace": ["*"],
        "query_shape": [[1, 1, 1, 1]],
        "value": [[0]],
        "variance": pl.Series([0.0], dtype=pl.Float32),
    })
    _write(f"{root}/US_Person_PL_PROD/Block.parquet/Constraint/State=50/part-0.parquet", con)
    return root


def test_get_pl94_real_block_groups_sums_blocks(block_nmf_root):
    df = get_pl94("block group", "total", state="VT", root=block_nmf_root,
                  real_block_groups=True)
    assert set(df["geoid"]) == {"500010001001", "500010001002"}
    assert df["geocode"].isna().all()

    combined = df[df["geoid"] == "500010001001"]["value"].array[0]
    assert int(combined) == 30
    assert bool(df[df["geoid"] == "500010001001"]["aian"].iloc[0]) is False
    draws = combined.sample(3000, rng=1).draws
    assert draws.std() > 5  # noise from both summed blocks survives

    solo = df[df["geoid"] == "500010001002"]["value"].array[0]
    assert int(solo) == 5


def test_get_pl94_real_block_groups_requires_block_group_geography(nmf_root):
    with pytest.raises(ValueError, match="real_block_groups"):
        get_pl94("county", "total", state="VT", root=nmf_root,
                 real_block_groups=True)


# ── synthetic DHC fixture ────────────────────────────────────────────────────

@pytest.fixture
def dhc_root(tmp_path):
    """Two Vermont counties of DHC person measurements, county-partitioned."""
    root = str(tmp_path / "dhc")

    def county_rows(geocode, sexhisp, age3):
        return {
            "geocode": [geocode] * 2,
            "query_name": ["hispanic * sex_dpq", "age_18_64_116 * sex_dpq"],
            "relgq": ["*", "*"],
            "sex": ["sex", "sex"],
            "age": ["*", "age_18_64_116"],
            "hispanic": ["hispanic", "*"],
            "cenrace": ["*", "*"],
            "query_shape": [[1, 2, 1, 2, 1], [1, 2, 3, 1, 1]],
            "value": [sexhisp, age3],
            "variance": pl.Series([16.0, 16.0], dtype=pl.Float32),
        }

    for geocode, county, sexhisp, age3 in (
            # (male/not-hispanic, male/hispanic, female/not, female/hispanic)
            ("05010007", "05010007", [900, 40, 950, 60], [300, 700, -6, 320, 750, 90]),
            ("05010009", "05010009", [500, 10, 520, 15], [100, 300, 110, 105, 310, 120])):
        _write(f"{root}/US_DHCP_PROD/County.parquet/DPQuery/"
               f"State=050/County={county}/part-0.parquet",
               pl.DataFrame(county_rows(geocode, sexhisp, age3)))
    return root


def test_get_dhc_tidy_shape_and_labels(dhc_root):
    df = get_dhc("county", "sex*hispanic", state="VT", root=dhc_root)
    assert list(df.columns) == [
        "geoid", "geocode", "aian", "query",
        "relgq", "sex", "age", "hispanic", "cenrace", "value", "variance"]
    assert set(df["geoid"]) == {"50007", "50009"}
    assert not df["aian"].any()

    essex = df[df["geoid"] == "50009"]
    assert list(zip(essex["sex"], essex["hispanic"])) == [
        ("male", "not_hispanic"), ("male", "hispanic"),
        ("female", "not_hispanic"), ("female", "hispanic")]
    assert [int(v) for v in essex["value"]] == [500, 10, 520, 15]
    # axes the query marginalizes out are labelled "total"
    assert set(essex["relgq"]) == {"total"}
    assert set(essex["age"]) == {"total"}


def test_get_dhc_county_filter_prunes_partitions(dhc_root):
    df = get_dhc("county", "sex*hispanic", state="VT", county="009",
                 root=dhc_root)
    assert set(df["geoid"]) == {"50009"}
    df = get_dhc("county", "sex*hispanic", state="VT", county=[7, 9],
                 root=dhc_root)
    assert set(df["geoid"]) == {"50007", "50009"}


def test_get_dhc_age_group_labels_follow_the_codebook(dhc_root):
    df = get_dhc("county", "sex*age_18_64_116", state="VT", county=9,
                 root=dhc_root)
    assert list(df["age"]) == ["0-17", "18-64", "65-115"] * 2
    assert [int(v) for v in df["value"]] == [100, 300, 110, 105, 310, 120]


def test_get_dhc_truncates_posteriors_at_zero(dhc_root):
    df = get_dhc("county", "sex*age_18_64_116", state="VT", county=7,
                 root=dhc_root)
    negative = df[(df["sex"] == "male") & (df["age"] == "65-115")]
    value = negative["value"].array[0]
    assert int(value) == -6
    assert value.sample(2000, rng=3).draws.min() >= 0

    unbounded = get_dhc("county", "sex*age_18_64_116", state="VT", county=7,
                        root=dhc_root, nonnegative=False)
    value = unbounded[(unbounded["sex"] == "male")
                      & (unbounded["age"] == "65-115")]["value"].array[0]
    assert value.sample(2000, rng=3).draws.min() < 0


def test_get_dhc_posterior_centers_on_the_measurement(dhc_root):
    df = get_dhc("county", "sex*hispanic", state="VT", county=9, root=dhc_root)
    value = df[(df["sex"] == "female")
               & (df["hispanic"] == "not_hispanic")]["value"].array[0]
    batch = value.sample(4000, rng=2)
    assert abs(batch.mean() - 520) < 1.0
    assert 3.0 < batch.draws.std() < 5.0    # variance 16 -> sd 4


def test_get_dhc_rejects_bad_inputs(dhc_root):
    with pytest.raises(ValueError, match="unknown geography"):
        get_dhc("tract", "sex*hispanic", state="VT", root=dhc_root)
    with pytest.raises(ValueError, match="unknown query"):
        get_dhc("county", "hhgq", state="VT", root=dhc_root)
    with pytest.raises(ValueError, match="requires a state"):
        get_dhc("county", "sex*hispanic", root=dhc_root)
    with pytest.raises(ValueError, match="does not take a state"):
        get_dhc("us", "sex*hispanic", state="VT", root=dhc_root)
    with pytest.raises(ValueError, match="county only applies"):
        get_dhc("state", "sex*hispanic", state="VT", county=9, root=dhc_root)


def test_get_pl94_rejects_bad_inputs(nmf_root):
    with pytest.raises(ValueError, match="unknown query"):
        get_pl94("county", "age_by_income", state="VT", root=nmf_root)
    with pytest.raises(ValueError, match="requires a state"):
        get_pl94("tract", "total")
    with pytest.raises(ValueError, match="unknown state"):
        get_pl94("county", "total", state="Narnia", root=nmf_root)
    with pytest.raises(ValueError, match="does not take a state"):
        get_pl94("us", "total", state="VT", root=nmf_root)


# ── integration against the real NMF download ───────────────────────────────

requires_data = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_ROOT), reason="census NMF download not present")

# Published 2020 PL94 national single-race counts (P1); NMF sd is ~27.6.
PUBLISHED_RACE_ALONE = {
    "white": 204277273,
    "black": 41104200,
    "aian": 3727135,
    "asian": 19886049,
    "nhopi": 689966,
    "sor": 27915715,
}


@requires_data
def test_real_data_cenrace_ordering_matches_published_counts():
    df = get_pl94("us", "cenrace")
    by_race = {race: int(value) for race, value in zip(df["cenrace"], df["value"])}
    sd = float(np.sqrt(df["variance"].iloc[0]))
    for race, published in PUBLISHED_RACE_ALONE.items():
        assert abs(by_race[race] - published) < 5 * sd


@requires_data
def test_real_data_national_person_total_is_exact():
    df = get_pl94("us", "total")
    total = df["value"].array[0]
    assert int(total) == 331449281
    assert np.all(total.sample(50, rng=1).draws == 331449281)


@requires_data
def test_real_data_unit_conditioning_reproduces_invariant():
    df = get_pl94("us", ["h1", "total"], table="unit")
    vac = df[df["h1"] == "vacant"]["value"].array[0]
    occ = df[df["h1"] == "occupied"]["value"].array[0]
    total = df[df["query"] == "total"]["value"].array[0]
    assert int(total) == 140498736
    batch_vac, batch_occ = sample_noisy_values(vac, occ, n=200, rng=4)
    assert np.all(batch_vac.draws + batch_occ.draws == 140498736)


@requires_data
def test_real_data_vermont_county_geoids():
    df = get_pl94("county", "total", state="Vermont")
    geoids = sorted(set(df["geoid"]))
    assert geoids[0] == "50001"
    assert len(geoids) == 14


@requires_data
def test_real_data_tract_geoids_resolve_via_block_crosswalk():
    df = get_pl94("tract", "total", state="VT")
    assert df["geoid"].notna().all()
    assert all(g.startswith("50") and len(g) == 11 for g in df["geoid"])


# ── integration against the real DHC download ───────────────────────────────
#
# These need the partitions `fetch_dhc` mirrors, and are skipped without them:
#   fetch_dhc("us"); fetch_dhc("county", state="VT")

def _dhc_partition(*parts):
    return os.path.join(DEFAULT_DHC_ROOT, "US_DHCP_PROD", *parts)


requires_dhc_us = pytest.mark.skipif(
    not os.path.isdir(_dhc_partition("US.parquet")),
    reason='DHC national partition not present; run fetch_dhc("us")')

requires_dhc_vt = pytest.mark.skipif(
    not os.path.isdir(_dhc_partition("County.parquet", "DPQuery", "State=050")),
    reason='DHC Vermont counties not present; '
           'run fetch_dhc("county", state="VT")')

# Published 2020 census sex counts; the national NMF sd is about 12 per cell.
PUBLISHED_SEX = {"male": 162_685_811, "female": 168_763_470}


def _total_and_sd(frame):
    """Summed measurement and its standard deviation over a frame's cells."""
    return (sum(int(v) for v in frame["value"]),
            math.sqrt(float(frame["variance"].sum())))


@requires_dhc_us
def test_real_dhc_sex_levels_match_published_national_counts():
    df = get_dhc("us", "sex*hispanic")
    for sex, published in PUBLISHED_SEX.items():
        total, sd = _total_and_sd(df[df["sex"] == sex])
        assert abs(total - published) < 5 * sd, sex

    total, sd = _total_and_sd(df)
    assert abs(total - 331_449_281) < 5 * sd


@requires_dhc_us
def test_real_dhc_age_groups_split_the_national_population():
    df = get_dhc("us", "sex*age_18_64_116")
    by_age = {age: sum(int(v) for v in group["value"])
              for age, group in df.groupby("age")}
    assert set(by_age) == {"0-17", "18-64", "65-115"}
    # published 2020 counts: 73.1M under 18 and 55.8M at 65 or older
    assert abs(by_age["0-17"] - 73_106_000) < 50_000
    assert abs(by_age["65-115"] - 55_792_000) < 50_000
    total, sd = _total_and_sd(df)
    assert abs(total - 331_449_281) < 5 * sd


@requires_dhc_us
def test_real_dhc_relgq_groupings_agree_across_independent_queries():
    """The coarse relgq queries measure the same histogram from two sides.

    relgq_4_groups, popSehsdTargetsRelship and gq_constr_groups bin the 42
    relgq levels differently; if the codebook's group membership were wrong
    the corresponding totals would disagree by far more than the noise.
    """
    four = get_dhc("us", "relgq_4_groups*sex")
    relship = get_dhc("us", "popSehsdTargetsRelship")
    constr = get_dhc("us", "gq_constr_groups*age_10_groups")

    householder = relship[relship["relgq"].isin(RELSHIP_LEVELS[:2])]
    other_household = relship[relship["relgq"].isin(RELSHIP_LEVELS[2:])]
    gq = relship[relship["relgq"] == "group_quarters"]
    household = relship[relship["relgq"] != "group_quarters"]

    for left, right in (
            (four[four["relgq"] == "householder"], householder),
            (four[four["relgq"] == "other_household_member"], other_household),
            (four[four["relgq"].isin(("institutional_gq",
                                      "noninstitutional_gq"))], gq),
            (constr[constr["relgq"] == "household"], household),
            (constr[constr["relgq"] != "household"], gq)):
        left_total, left_sd = _total_and_sd(left)
        right_total, right_sd = _total_and_sd(right)
        sd = math.hypot(left_sd, right_sd)
        assert abs(left_total - right_total) < 5 * sd, (left_total, right_total)


@requires_dhc_us
def test_real_dhc_group_quarters_population_matches_published():
    df = get_dhc("us", "relgq_4_groups*sex")
    gq = df[df["relgq"].isin(("institutional_gq", "noninstitutional_gq"))]
    total, sd = _total_and_sd(gq)
    assert abs(total - 8_239_000) < 5 * sd + 500      # published is rounded

    householders, sd = _total_and_sd(df[df["relgq"] == "householder"])
    assert abs(householders - 126_817_000) < 5 * sd + 1_000


@requires_dhc_vt
def test_real_dhc_vermont_counties_resolve_to_geoids():
    df = get_dhc("county", "sex*hispanic", state="Vermont")
    geoids = sorted(set(df["geoid"]))
    assert geoids[0] == "50001"
    assert len(geoids) == 14
    assert not df["aian"].any()
    assert set(df["sex"]) == set(SEX_LEVELS)

    essex = df[df["geoid"] == "50009"]
    total, sd = _total_and_sd(essex)
    assert abs(total - 5_920) < 5 * sd               # published Essex County


@requires_dhc_vt
def test_real_dhc_county_filter_matches_a_full_state_read():
    whole = get_dhc("county", "sex*age_18_64_116", state="VT")
    one = get_dhc("county", "sex*age_18_64_116", state="VT", county=9)
    assert set(one["geoid"]) == {"50009"}
    expected = whole[whole["geoid"] == "50009"]
    assert [int(v) for v in one["value"]] == [int(v) for v in expected["value"]]
