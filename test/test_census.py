import os

import numpy as np
import polars as pl
import pytest
import sympy as sp

from noisyvalue.census import (
    CENRACE_LEVELS,
    DEFAULT_ROOT,
    HHGQ_LEVELS,
    _person_axis_specs,
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
