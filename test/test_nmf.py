"""The 2020 Census NMF products, built on the dataset framework.

The framework itself is tested in test_dataset.py; what is checked here is
that the Census schema is declared correctly, that the constraint rows are
translated into the right evidence, and that a view keeps its promise.
"""

import math
import os

import numpy as np
import polars as pl
import pytest

from noisyvalue import nmf
from noisyvalue.core import sample_noisy_values
from noisyvalue.dataset import GeographyUnavailable
from noisyvalue.graph import DiscreteGaussianNode, NoiseNode
from sympy import oo


def _only_noise(value):
    """The single noise node behind a cell's posterior."""
    nodes = [n for n in value._root.closure() if isinstance(n, NoiseNode)]
    assert len(nodes) == 1
    return nodes[0]


# ── catalog ──────────────────────────────────────────────────────────────────

def test_universes_have_the_shapes_the_release_documents():
    assert nmf.PL94_PERSON.shape == (8, 2, 2, 63)
    assert nmf.PL94_UNIT.shape == (2,)
    assert nmf.DHC_PERSON.shape == (42, 2, 116, 2, 63)
    assert nmf.DHC_PERSON.cells == 1_227_744


def test_every_relgq_binning_partitions_the_42_levels():
    relgq = nmf.DHC_PERSON.axis("relgq")
    for name in relgq.binnings:
        partition = relgq.binning(name).partition
        assert partition.is_cover, name
        assert len(partition) == len(relgq.binning(name).labels), name


def test_age_binnings_partition_single_years_zero_to_115():
    age = nmf.DHC_PERSON.axis("age")
    for name in ("age_18_64_116", "age_10_groups", "age_38_groups",
                 "age_40_groups"):
        assert age.binning(name).partition.is_cover, name
    assert age.refines("detailed", "age_38_groups")
    assert age.refines("age_40_groups", "age_18_64_116")
    assert not age.refines("age_18_64_116", "age_40_groups")


def test_the_hhgq_sub_selections_are_declared_as_partial_covers():
    hhgq = nmf.PL94_PERSON.axis("hhgq")
    assert hhgq.binning("hhinstlevels").is_cover
    assert not hhgq.binning("gqNursingTotal").is_cover
    assert not nmf.PL94_PERSON.axis("votingage").binning("nonvoting").is_cover


def test_measurement_cell_counts_match_the_release():
    cells = dict(zip(*[nmf.dhc_2020.measurements()[c]
                       for c in ("measurement", "cells")]))
    assert cells["sex*hispanic"] == 4
    assert cells["sex*age_38_groups"] == 76
    assert cells["gq_constr_groups*age_10_groups"] == 60
    assert cells["relgq*sex*age_40_groups*hispanic*cenrace"] == 423_360
    assert cells["detailed"] == 1_227_744

    person = dict(zip(*[nmf.pl94_2020.measurements()[c]
                        for c in ("measurement", "cells")]))
    assert person["total"] == 1 and person["detailed"] == 2016


def test_measurement_names_are_matched_loosely():
    m = nmf.dhc_2020.measurement("SEX * Hispanic_dpq")
    assert m.name == "sex*hispanic"
    with pytest.raises(KeyError, match="unknown measurement"):
        nmf.dhc_2020.measurement("income")


def test_dhc_has_no_geography_below_county_and_says_so():
    assert nmf.dhc_2020.plan("county").read_level == "County"
    with pytest.raises(GeographyUnavailable, match="not in this release"):
        nmf.dhc_2020.plan("tract")


def test_spine_and_tabulation_block_groups_are_different_geographies():
    spine = nmf.pl94_2020.plan("block group")
    assert spine.read_level == "Block_Group" and not spine.aggregate
    tabulation = nmf.pl94_2020.plan("tabulation block group")
    assert tabulation.read_level == "Block" and tabulation.aggregate


# ── synthetic PL94 fixture ───────────────────────────────────────────────────

VT_COUNTY = "050100011"  # spine code for county FIPS 50001


def _write(path, frame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.write_parquet(path)


@pytest.fixture
def pl94_root(tmp_path):
    root = str(tmp_path / "nmf")

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


def _view(product, root, geography="county", **kwargs):
    return product.view(geography, state="VT", root=root, **kwargs)


def test_pl94_person_frame_is_tidy_and_carries_geoids(pl94_root):
    df = _view(nmf.pl94_2020, pl94_root).queries(["total", "hhgq"])
    assert list(df.columns) == [
        "geoid", "geocode", "aian", "measurement",
        "hhgq", "votingage", "hispanic", "cenrace", "value", "variance"]
    assert len(df) == 9
    assert set(df["geoid"]) == {"50001"}
    assert not df["aian"].any()
    hhgq = df[df["measurement"] == "hhgq"]
    assert list(hhgq["hhgq"]) == list(nmf.PL94_PERSON.axis("hhgq").levels)
    assert [int(v) for v in hhgq["value"]] == [4000, 3, 0, 40, 0, 200, -4, 30]


def test_hhgq_lower_bounds_apply_only_to_whole_category_totals(pl94_root):
    df = _view(nmf.pl94_2020, pl94_root).query("hhgq")
    correctional = df[df["hhgq"] == "correctional"]["value"].array[0]
    # obs 3 with an exact lower bound of 5: the posterior lives in [5, 6000]
    assert correctional.sample(2000, rng=3).draws.min() >= 5
    military = df[df["hhgq"] == "military"]["value"].array[0]
    assert military.sample(2000, rng=3).draws.min() >= 0   # obs -4, nonnegative


def test_hhgq_lower_bounds_do_not_reach_a_sliced_category(pl94_root):
    # nonnegativity off, so the only floor that could appear is the constraint's
    view = _view(nmf.pl94_2020, pl94_root, nonnegative=False)

    whole = view.query("hhgq")
    node = _only_noise(whole[whole["hhgq"] == "correctional"]["value"].array[0])
    # the cell *is* the category total, so its lower bound of 5 binds; in
    # eps space that is a ceiling of obs - 5 = -2
    assert isinstance(node, DiscreteGaussianNode)
    assert node.high != oo
    assert float(node.high) == -2

    # the detailed query slices the category by age, ethnicity and race, and a
    # slice is not bounded below by what the whole category must hold
    detailed = view.query("detailed")
    sliced_node = _only_noise(detailed[detailed["hhgq"] == "correctional"]["value"].array[0])
    assert isinstance(sliced_node, DiscreteGaussianNode)
    assert sliced_node.low == -oo and sliced_node.high == oo


def test_nonnegativity_can_be_declined(pl94_root):
    df = _view(nmf.pl94_2020, pl94_root, nonnegative=False,
               apply_constraints=False).query("hhgq")
    military = df[df["hhgq"] == "military"]["value"].array[0]
    assert military.sample(2000, rng=3).draws.min() < 0


def test_the_nursing_structural_zero_pins_under_18_residents(pl94_root):
    df = _view(nmf.pl94_2020, pl94_root).query("detailed")
    assert len(df) == 2016
    slab = df[(df["hhgq"] == "nursing") & (df["votingage"] == "under_18")]
    assert len(slab) == 126
    for value in slab["value"]:
        assert np.all(value.sample(20, rng=5).draws == 0)
    other = df[(df["hhgq"] == "nursing") & (df["votingage"] == "voting_age")]
    assert any(v.sample(20, rng=5).draws.std() > 0 for v in other["value"])


def test_the_unit_total_is_served_from_evidence_and_conditions_the_pair(pl94_root):
    view = _view(nmf.pl94_2020_units, pl94_root)
    df = view.queries(["h1", "total"])
    total = df[df["measurement"] == "total"]["value"].array[0]
    assert int(total) == 100
    assert np.all(total.sample(20, rng=1).draws == 100)

    vac = df[df["h1"] == "vacant"]["value"].array[0]
    occ = df[df["h1"] == "occupied"]["value"].array[0]
    batch_vac, batch_occ = sample_noisy_values(vac, occ, n=500, rng=9)
    assert np.all(batch_vac.draws + batch_occ.draws == 100)
    # posterior centre is (obs_vac - obs_occ + total) / 2 = 25
    assert abs(batch_vac.mean() - 25) < 0.5


def test_without_evidence_the_unit_pair_is_independent_and_the_total_is_gone(pl94_root):
    view = _view(nmf.pl94_2020_units, pl94_root, apply_constraints=False)
    df = view.query("h1")
    vac, occ = (df[df["h1"] == side]["value"].array[0]
                for side in ("vacant", "occupied"))
    batch_vac, batch_occ = sample_noisy_values(vac, occ, n=500, rng=9)
    assert len(np.unique(batch_vac.draws + batch_occ.draws)) > 1
    # the unit total was never a released query, so with the invariants off
    # there is nothing left to answer it with, and that is said out loud
    with pytest.warns(UserWarning, match="no released query"):
        assert len(view.query("total")) == 0


# ── the view contract ────────────────────────────────────────────────────────

def test_the_same_measurement_twice_returns_the_very_same_nodes(pl94_root):
    view = _view(nmf.pl94_2020, pl94_root)
    first, second = view.query("hhgq"), view.query("hhgq")
    assert all(a._root is b._root
               for a, b in zip(first["value"], second["value"]))


def test_separate_views_are_independent(pl94_root):
    a = _view(nmf.pl94_2020, pl94_root).query("hhgq")["value"].array[0]
    b = _view(nmf.pl94_2020, pl94_root).query("hhgq")["value"].array[0]
    assert a._root is not b._root
    draws_a, draws_b = sample_noisy_values(a, b, n=400, rng=2)
    assert not np.array_equal(draws_a.draws, draws_b.draws)


def test_a_rollup_stays_consistent_with_the_cells_it_came_from(pl94_root):
    view = _view(nmf.pl94_2020, pl94_root)
    coarse = view.rollup("detailed", hhgq="hhinstlevels", votingage="*",
                         hispanic="*", cenrace="*")
    household = coarse[coarse["hhgq"] == "household"]["value"].array[0]
    # detailed cell values run 1..2016 in cell order, so hhgq=household is 1..252
    assert int(household) == 252 * 253 // 2

    detailed = view.query("detailed")
    parts = list(detailed[detailed["hhgq"] == "household"]["value"])
    batches = sample_noisy_values(household, *parts, n=50, rng=6)
    assert np.array_equal(batches[0].draws,
                          sum(b.draws for b in batches[1:]))


def test_a_rollup_must_actually_be_a_coarsening(pl94_root):
    view = _view(nmf.pl94_2020, pl94_root)
    with pytest.raises(ValueError, match="does not refine"):
        view.rollup("hhgq", hhgq="detailed", votingage="detailed")


def test_evidence_used_in_full_leaves_nothing_to_report(pl94_root):
    # PL94 person carries only bounds and a structural zero at county level,
    # and both are imposed exactly, so the report is empty
    view = _view(nmf.pl94_2020, pl94_root)
    view.queries(["hhgq", "detailed"])
    assert view.unused_evidence().empty


# ── geography ────────────────────────────────────────────────────────────────

@pytest.fixture
def block_root(tmp_path):
    root = str(tmp_path / "nmf")

    def block_geocode(bg, block):
        # "0" (non-AIAN branch) + state(2) + county(3) + tract(6) + bg(1) +
        # block(4); bg duplicates the block's own leading digit, as in the
        # real encoding
        return f"0{50:02d}001000100{bg}{block}"

    geocodes = [block_geocode(1, "1001"), block_geocode(1, "1002"),
                block_geocode(2, "2001")]
    dpq = pl.DataFrame({
        "geocode": geocodes,
        "query_name": ["total_dpq"] * 3,
        "hhgq": ["*"] * 3, "votingage": ["*"] * 3,
        "hispanic": ["*"] * 3, "cenrace": ["*"] * 3,
        "query_shape": [[1, 1, 1, 1]] * 3,
        "value": [[10], [20], [5]],
        "variance": pl.Series([100.0] * 3, dtype=pl.Float32),
    })
    _write(f"{root}/US_Person_PL_PROD/Block.parquet/DPQuery/State=50/part-0.parquet", dpq)

    # one constraint row purely so the "sign" column exists in the schema
    con = pl.DataFrame({
        "geocode": [geocodes[0]],
        "query_name": ["nurse_nva_0_con"],
        "sign": ["="],
        "hhgq": ["gqNursingTotal"], "votingage": ["nonvoting"],
        "hispanic": ["*"], "cenrace": ["*"],
        "query_shape": [[1, 1, 1, 1]],
        "value": [[0]],
        "variance": pl.Series([0.0], dtype=pl.Float32),
    })
    _write(f"{root}/US_Person_PL_PROD/Block.parquet/Constraint/State=50/part-0.parquet", con)
    return root


def test_tabulation_block_groups_are_recovered_by_descending_to_blocks(block_root):
    df = _view(nmf.pl94_2020, block_root, "tabulation block group").query("total")
    assert set(df["geoid"]) == {"500010001001", "500010001002"}
    assert df["geocode"].isna().all()

    combined = df[df["geoid"] == "500010001001"]["value"].array[0]
    assert int(combined) == 30
    assert bool(df[df["geoid"] == "500010001001"]["aian"].iloc[0]) is False
    # noise from both summed blocks survives; it is not collapsed to a constant
    assert combined.sample(3000, rng=1).draws.std() > 5

    solo = df[df["geoid"] == "500010001002"]["value"].array[0]
    assert int(solo) == 5


def test_selection_errors_are_caught_before_any_io():
    with pytest.raises(ValueError, match="requires a state"):
        nmf.pl94_2020.view("tract")
    with pytest.raises(ValueError, match="does not take a state"):
        nmf.pl94_2020.view("us", state="VT")
    with pytest.raises(ValueError, match="county only applies"):
        nmf.dhc_2020.view("state", state="VT", county=[1])
    with pytest.raises(TypeError, match="unexpected selection"):
        nmf.pl94_2020.view("state", stat="VT")
    with pytest.raises(ValueError, match="unknown state"):
        nmf.pl94_2020.view("state", state="Atlantis")


# ── synthetic DHC fixture ────────────────────────────────────────────────────

DHC_COUNTY = "05010009"


@pytest.fixture
def dhc_root(tmp_path):
    """One county whose DHC measurements come with their constraint rows.

    The PL94 invariant is deliberately lopsided -- nobody in group quarters
    at all except 40 people in college housing, all of them hispanic -- so
    blocks of every size show up in one small fixture.
    """
    root = str(tmp_path / "dhc")

    pl94 = np.zeros((8, 2, 2, 63), dtype=np.int64)
    pl94[0, 0, 0, 0] = 200          # household, under 18, not hispanic
    pl94[0, 1, 0, 0] = 700          # household, voting age, not hispanic
    pl94[0, 1, 1, 0] = 60           # household, voting age, hispanic
    pl94[5, 1, 1, 0] = 40           # college housing, voting age, hispanic

    dpq = pl.DataFrame({
        "geocode": [DHC_COUNTY] * 3,
        "query_name": ["hispanic * sex_dpq", "age_18_64_116 * sex_dpq",
                       "sex * relgq_4_groups_dpq"],
        "relgq": ["*", "*", "relgq_4_groups"],
        "sex": ["sex", "sex", "sex"],
        "age": ["*", "age_18_64_116", "*"],
        "hispanic": ["hispanic", "*", "*"],
        "cenrace": ["*", "*", "*"],
        "query_shape": [[1, 2, 1, 2, 1], [1, 2, 3, 1, 1], [4, 2, 1, 1, 1]],
        "value": [[460, 44, 450, 62],
                  [95, 300, 210, 110, 290, 195],
                  [260, 240, 250, 230, 3, -2, 18, 25]],
        "variance": pl.Series([16.0] * 3, dtype=pl.Float32),
    })
    _write(f"{root}/US_DHCP_PROD/County.parquet/DPQuery/"
           f"State=050/County={DHC_COUNTY}/part-0.parquet", dpq)

    con = pl.DataFrame({
        "geocode": [DHC_COUNTY] * 4,
        "query_name": ["pl94_con", "hhgq_total_lb_con", "hhgq_total_ub_con",
                       "relgq0_lt15_con"],
        "sign": ["=", ">=", "<=", "="],
        "relgq": ["hhgq8lev", "gqlevels", "gqlevels", "relgq0"],
        "sex": ["*", "*", "*", "*"],
        "age": ["votingage", "*", "*", "age_lessthan15"],
        "hispanic": ["hispanic", "*", "*", "*"],
        "cenrace": ["cenrace", "*", "*", "*"],
        "query_shape": [[8, 1, 2, 2, 63], [25, 1, 1, 1, 1],
                        [25, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "value": [list(pl94.ravel()), [0] * 25,
                  [999999] + [0] * 15 + [99999] + [0] * 8, [0]],
        "variance": pl.Series([0.0] * 4, dtype=pl.Float32),
    })
    _write(f"{root}/US_DHCP_PROD/County.parquet/Constraint/"
           f"State=050/County={DHC_COUNTY}/part-0.parquet", con)
    return root


def test_pl94_invariant_ties_male_and_female_within_each_hispanic_level(dhc_root):
    df = _view(nmf.dhc_2020, dhc_root).query("sex*hispanic")
    by_ethnicity = {h: list(g["value"]) for h, g in df.groupby("hispanic")}

    male, female = by_ethnicity["not_hispanic"]
    a, b = sample_noisy_values(male, female, n=300, rng=5)
    assert np.all(a.draws + b.draws == 900)   # 200 + 700, and PL94 has no sex
    assert len(np.unique(a.draws)) > 1

    male, female = by_ethnicity["hispanic"]
    a, b = sample_noisy_values(male, female, n=300, rng=5)
    assert np.all(a.draws + b.draws == 100)   # 60 in households + 40 in college


def test_a_zero_pl94_block_empties_every_cell_it_covers(dhc_root):
    df = _view(nmf.dhc_2020, dhc_root).query("relgq_4_groups*sex")
    # the invariant puts nobody in institutional group quarters at all
    slab = df[df["relgq"] == "institutional_gq"]["value"]
    assert len(slab) == 2
    for value in slab:
        assert np.all(value.sample(20, rng=5).draws == 0)
    # but noninstitutional GQ holds 40, so those cells stay free
    other = df[df["relgq"] == "noninstitutional_gq"]["value"]
    assert any(v.sample(50, rng=5).draws.std() > 0 for v in other)


def test_blocks_of_three_or_more_keep_the_total_only_as_a_bound(dhc_root):
    view = _view(nmf.dhc_2020, dhc_root)
    df = view.query("sex*age_18_64_116")
    # six cells (2 sexes x 3 age groups) against a PL94 invariant that only
    # resolves under-18 vs voting-age: three free cells per block
    draws = [v.sample(400, rng=4).draws for v in df["value"]]
    assert all(d.std() > 0 for d in draws)
    assert all(d.max() <= 1000 for d in draws)

    report = view.unused_evidence()
    weakened = report[report["kind"].str.startswith("exact total")]
    assert not weakened.empty
    assert weakened["source"].iloc[0] == "pl94_con"
    assert "bound only" in weakened["detail"].iloc[0]


def test_without_the_invariants_nothing_is_tied_together(dhc_root):
    df = _view(nmf.dhc_2020, dhc_root, apply_constraints=False).query("sex*hispanic")
    values = list(df[df["hispanic"] == "not_hispanic"]["value"])
    a, b = sample_noisy_values(*values, n=300, rng=5)
    assert len(np.unique(a.draws + b.draws)) > 1


def test_the_county_filter_prunes_partitions(dhc_root):
    matched = _view(nmf.dhc_2020, dhc_root, county=[9]).query("sex*hispanic")
    assert len(matched) == 4
    with pytest.warns(UserWarning, match="no released query"):
        missed = _view(nmf.dhc_2020, dhc_root, county=[7]).query("sex*hispanic")
    assert len(missed) == 0


def test_the_catalog_is_checked_against_the_file(dhc_root, tmp_path):
    """A codebook that drifts from a reissued release must not pass silently."""
    bad = pl.DataFrame({
        "geocode": [DHC_COUNTY],
        "query_name": ["hispanic * sex_dpq"],
        "relgq": ["*"], "sex": ["sex"], "age": ["*"],
        "hispanic": ["*"],                      # the file now says otherwise
        "cenrace": ["*"],
        "query_shape": [[1, 2, 1, 1, 1]],
        "value": [[900, 950]],
        "variance": pl.Series([16.0], dtype=pl.Float32),
    })
    root = str(tmp_path / "drift")
    _write(f"{root}/US_DHCP_PROD/County.parquet/DPQuery/"
           f"State=050/County={DHC_COUNTY}/part-0.parquet", bad)
    with pytest.raises(ValueError, match="which is not the catalog's"):
        _view(nmf.dhc_2020, root).query("sex*hispanic")


# ── integration against the real NMF download ────────────────────────────────

requires_pl94 = pytest.mark.skipif(
    not os.path.isdir(nmf.DEFAULT_PL94_ROOT),
    reason="census NMF download not present")


def _dhc_partition(*parts):
    return os.path.join(nmf.DEFAULT_DHC_ROOT, "US_DHCP_PROD", *parts)


requires_dhc_us = pytest.mark.skipif(
    not os.path.isdir(_dhc_partition("US.parquet", "DPQuery")),
    reason='DHC national partition not present; run fetch_dhc("us")')

requires_dhc_vt = pytest.mark.skipif(
    not os.path.isdir(_dhc_partition("County.parquet", "DPQuery", "State=050")),
    reason='DHC Vermont counties not present; run fetch_dhc("county", state="VT")')

PUBLISHED_RACE_ALONE = {
    "white": 204277273, "black": 41104200, "aian": 3727135,
    "asian": 19886049, "nhopi": 689966, "sor": 27915715,
}


@requires_pl94
def test_real_cenrace_ordering_matches_published_counts():
    df = nmf.pl94_2020.view("us").query("cenrace")
    by_race = {race: int(v) for race, v in zip(df["cenrace"], df["value"])}
    sd = float(np.sqrt(df["variance"].iloc[0]))
    for race, published in PUBLISHED_RACE_ALONE.items():
        assert abs(by_race[race] - published) < 5 * sd, race


@requires_pl94
def test_real_national_person_total_comes_from_the_invariant():
    # there is no total_dpq nationally; the exact total is a constraint row,
    # and the framework serves the measurement from evidence alone
    df = nmf.pl94_2020.view("us").query("total")
    total = df["value"].array[0]
    assert int(total) == 331449281
    assert np.all(total.sample(50, rng=1).draws == 331449281)


@requires_pl94
def test_real_unit_conditioning_reproduces_the_invariant():
    df = nmf.pl94_2020_units.view("us").queries(["h1", "total"])
    vac, occ = (df[df["h1"] == side]["value"].array[0]
                for side in ("vacant", "occupied"))
    total = df[df["measurement"] == "total"]["value"].array[0]
    assert int(total) == 140498736
    a, b = sample_noisy_values(vac, occ, n=200, rng=4)
    assert np.all(a.draws + b.draws == 140498736)


@requires_pl94
def test_real_vermont_county_geoids():
    df = nmf.pl94_2020.view("county", state="Vermont").query("total")
    geoids = sorted(set(df["geoid"]))
    assert geoids[0] == "50001" and len(geoids) == 14


@requires_pl94
def test_real_tract_geoids_resolve_via_the_block_crosswalk():
    df = nmf.pl94_2020.view("tract", state="VT").query("total")
    assert df["geoid"].notna().all()
    assert all(g.startswith("50") and len(g) == 11 for g in df["geoid"])


PUBLISHED_SEX = {"male": 162_685_811, "female": 168_763_470}


def _total_and_sd(frame):
    return (sum(int(v) for v in frame["value"]),
            math.sqrt(float(frame["variance"].sum())))


@requires_dhc_us
def test_real_dhc_sex_matches_published_national_counts():
    df = nmf.dhc_2020.view("us").query("sex*hispanic")
    for sex, published in PUBLISHED_SEX.items():
        total, sd = _total_and_sd(df[df["sex"] == sex])
        assert abs(total - published) < 5 * sd, sex


@requires_dhc_vt
def test_real_dhc_one_view_answers_two_queries_consistently():
    view = nmf.dhc_2020.view("county", state="VT", county=[1])
    first = view.query("sex*hispanic")
    second = view.query("sex*hispanic")
    assert all(a._root is b._root
               for a, b in zip(first["value"], second["value"]))

    # the invariant makes male + female exact within each hispanic level
    for _, group in first.groupby("hispanic"):
        a, b = sample_noisy_values(*list(group["value"]), n=100, rng=8)
        assert len(np.unique(a.draws + b.draws)) == 1


@requires_dhc_vt
def test_real_dhc_relgq_age_rules_empty_impossible_cells():
    view = nmf.dhc_2020.view("county", state="VT", county=[1])
    df = view.query("relgq*sex*age_40_groups*hispanic*cenrace")
    # relgq0 (householder living alone) cannot be under 15
    young = df[(df["relgq"] == "householder_alone") &
               (df["age"].isin([str(a) for a in range(15)]))]
    assert len(young) > 0
    assert all(int(v) == 0 for v in young["value"])
