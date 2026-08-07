"""Framework tests: the lattice, the catalog, the evidence vocabulary, the solver.

Nothing here touches a dataset.  The Census products built on this are
exercised in test_nmf.py.
"""

import numpy as np
import pytest

from noisyvalue.dataset import (
    Axis,
    Binning,
    BlockStrategy,
    BoundedHistogram,
    BoundOnly,
    DiscreteGaussianFamily,
    ExactHistogram,
    GeoLevel,
    GeographyUnavailable,
    Measurement,
    Partition,
    Product,
    Region,
    Source,
    Universe,
    ZeroRegion,
    marginal_coarsening,
    solve,
)
from noisyvalue.dataset.solve import build_values
from noisyvalue.graph import DiscreteGaussianNode, NoiseNode
from sympy import oo


# ── partitions ───────────────────────────────────────────────────────────────

def test_partition_rejects_overlapping_groups():
    with pytest.raises(ValueError, match="appears in groups"):
        Partition(((0, 1), (1, 2)), 3)


def test_partition_rejects_atoms_outside_the_base_set():
    with pytest.raises(ValueError, match="outside range"):
        Partition(((0, 5),), 3)


def test_partition_distinguishes_a_cover_from_a_sub_selection():
    assert Partition(((0, 1), (2,)), 3).is_cover
    # "nursing homes only" names three of eight categories and pins nothing
    # about the rest; the distinction decides whether a total is known
    assert not Partition(((1,),), 3).is_cover


def test_partition_group_order_is_preserved_but_atom_order_is_not():
    p = Partition(((2, 1), (0,)), 3)
    assert p.groups == ((1, 2), (0,))


def test_refines_follows_the_coarsening_chain():
    fine = Partition.discrete(4)
    mid = Partition(((0, 1), (2, 3)), 4)
    coarse = Partition.trivial(4)
    assert fine.refines(mid) and mid.refines(coarse) and fine.refines(coarse)
    assert not mid.refines(fine)


def test_join_merges_positions_until_both_sides_agree():
    # query bins {0,1},{2},{3}; evidence bins {0},{1,2},{3}
    query = Partition(((0, 1), (2,), (3,)), 4)
    evidence = Partition(((0,), (1, 2), (3,)), 4)
    blocks = query.join(evidence)
    assert len(blocks) == 2
    first, second = blocks
    assert first.a_positions == (0, 1) and first.b_positions == (0, 1)
    assert first.exact
    assert second.a_positions == (2,) and second.b_positions == (2,)
    assert second.exact


def test_join_marks_a_partial_cover_as_inexact_but_contained():
    query = Partition(((0,),), 4)             # asks about one atom
    evidence = Partition(((0, 1), (2, 3)), 4)  # knows totals of pairs
    blocks = query.join(evidence)
    block = next(b for b in blocks if b.a_positions)
    assert not block.exact          # the pair total is not this cell's total
    assert block.contained          # but it does bound it


def test_join_against_a_marginalized_partition_collapses_to_one_block():
    query = Partition.discrete(5)
    blocks = query.join(Partition.trivial(5))
    assert len(blocks) == 1
    assert blocks[0].a_positions == (0, 1, 2, 3, 4)
    assert blocks[0].exact


def test_meet_is_the_coarsest_common_refinement():
    a = Partition(((0, 1, 2), (3, 4, 5)), 6)
    b = Partition(((0, 1), (2, 3), (4, 5)), 6)
    assert a.meet(b).groups == ((0, 1), (2,), (3,), (4, 5))


# ── catalog ──────────────────────────────────────────────────────────────────

def _axis():
    axis = Axis("size", ("s", "m", "l", "xl"))
    axis.declare("pairs", ((0, 1), (2, 3)), ("small", "large"))
    return axis


def test_axis_supplies_marginal_and_detailed_binnings_for_free():
    axis = _axis()
    assert set(axis.binnings) >= {"*", "detailed", "pairs"}
    assert axis.binning("*").labels == ("total",)
    assert axis.binning("detailed").labels == ("s", "m", "l", "xl")


def test_binning_rejects_labels_that_do_not_match_its_groups():
    with pytest.raises(ValueError, match="labels for"):
        Binning("bad", Partition(((0,), (1,)), 2), ("only one",))


def test_axis_rejects_a_binning_over_the_wrong_atom_set():
    with pytest.raises(ValueError, match="over 2 atoms"):
        _axis().add(Binning("x", Partition(((0,), (1,)), 2), ("a", "b")))


def test_axis_refines_reports_the_lattice_relation():
    axis = _axis()
    assert axis.refines("detailed", "pairs")
    assert not axis.refines("pairs", "detailed")


def test_measurement_parse_marginalizes_the_axes_a_name_omits():
    universe = Universe("u", (_axis(), Axis("colour", ("red", "blue"))))
    m = Measurement.parse("pairs", universe)
    assert m.specs == ("pairs", "*")
    assert m.shape == (2, 1)
    assert m.labels == (("small", "large"), ("total",))


def test_measurement_parse_rejects_a_binning_no_axis_declares():
    universe = Universe("u", (_axis(),))
    with pytest.raises(ValueError, match="no axis has binning"):
        Measurement.parse("weight", universe)


def test_detailed_resolves_every_axis():
    universe = Universe("u", (_axis(), Axis("colour", ("red", "blue"))))
    assert Measurement.parse("detailed", universe).cells == 8


# ── geography ────────────────────────────────────────────────────────────────

class _StubSource(Source):
    def __init__(self, levels):
        self._levels = tuple(levels)

    def levels(self):
        return self._levels


def _product(source, geographies):
    universe = Universe("u", (_axis(),))
    return Product("p", universe, [Measurement.parse("pairs", universe)],
                   geographies, source)


def test_a_level_the_source_carries_is_read_directly():
    p = _product(_StubSource(("County", "Block")),
                 (GeoLevel("county", native="County"),
                  GeoLevel("block", native="Block")))
    plan = p.plan("county")
    assert plan.read_level == "County" and not plan.aggregate


def test_an_incomparable_level_descends_to_the_atoms():
    p = _product(_StubSource(("Block",)),
                 (GeoLevel("block", native="Block"),
                  GeoLevel("group", derive_from="block", key=lambda g: g[:2])))
    plan = p.plan("group")
    assert plan.aggregate and plan.read_level == "Block"


def test_computable_in_principle_but_unavailable_here_is_reported():
    # the meet always exists mathematically; it is usable only when the
    # atoms are measured, and DHC-style releases stop partway down
    p = _product(_StubSource(("County",)),
                 (GeoLevel("county", native="County"),
                  GeoLevel("group", derive_from="block", key=lambda g: g[:2]),
                  GeoLevel("block", native="Block")))
    with pytest.raises(GeographyUnavailable, match="re-aggregated"):
        p.plan("group")
    with pytest.raises(GeographyUnavailable, match="not in this release"):
        p.plan("block")


# ── evidence ─────────────────────────────────────────────────────────────────

def _universe():
    hhgq = Axis("hhgq", ("house", "prison", "dorm", "ship"))
    hhgq.declare("pairs", ((0, 1), (2, 3)), ("a", "b"))
    age = Axis("age", ("young", "old"))
    return Universe("u", (hhgq, age))


def test_zero_region_pins_only_cells_wholly_inside_it():
    universe = _universe()
    zero = ZeroRegion(Region(({1}, {0})))          # prison, young
    detailed = Measurement.parse("detailed", universe)
    mask = zero.pinned(detailed.binnings)
    assert mask[1, 0] and mask.sum() == 1

    # a cell that merges prison with houses is not wholly inside the region
    merged = Measurement("m", universe, ("pairs", "detailed"))
    assert not zero.pinned(merged.binnings).any()


def test_upper_bounds_transfer_to_sub_slices_but_lower_bounds_do_not():
    universe = _universe()
    coarsening = marginal_coarsening(universe, {"hhgq": (0, 1, 2, 3)})
    bounds = BoundedHistogram(coarsening,
                              lo=np.array([[5], [7], [0], [0]]),
                              hi=np.array([[50], [70], [0], [0]]))

    whole = Measurement.parse("detailed", _universe())
    whole = Measurement("whole", universe, ("detailed", "*"))
    lo, hi = bounds.bounds(whole.binnings)
    assert list(hi[:, 0]) == [50, 70, 0, 0]
    assert list(lo[:, 0]) == [5, 7, 0, 0]      # the cell *is* the category

    sliced = Measurement("sliced", universe, ("detailed", "detailed"))
    lo, hi = bounds.bounds(sliced.binnings)
    assert list(hi[:, 0]) == [50, 70, 0, 0]    # a slice is still at most this
    assert np.isnan(lo).all()                  # but is not at least it


def test_a_bound_on_a_partial_cover_says_nothing_about_what_it_omits():
    universe = _universe()
    # a rule stated over prisons only, leaving the other three unaccounted
    coarsening = (Partition(((1,),), 4), Partition.trivial(2))
    bounds = BoundedHistogram(coarsening, hi=np.array([[9]]))
    marginal = Measurement("m", universe, ("*", "*"))
    lo, hi = bounds.bounds(marginal.binnings)
    assert np.isnan(hi).all()


def test_exact_histogram_splits_a_measurement_into_pinned_blocks():
    universe = _universe()
    coarsening = marginal_coarsening(universe, {"hhgq": (0, 0, 1, 1)})
    known = ExactHistogram(coarsening, np.array([[100], [20]]))
    detailed = Measurement.parse("detailed", universe)
    blocks = list(known.blocks(detailed.binnings, detailed.shape))
    assert len(blocks) == 2
    assert all(b.exact for b in blocks)
    assert sorted(b.total for b in blocks) == [20, 100]
    assert sorted(len(b.cells) for b in blocks) == [4, 4]


# ── the solver ───────────────────────────────────────────────────────────────

def _cells(universe, specs, values, evidence, **kwargs):
    m = Measurement("m", universe, specs)
    resolution = solve(m, values, 16.0, evidence, **kwargs)
    obs, roots = build_values(resolution, values, 16.0)
    from noisyvalue.core import NoisyInt
    return resolution, [NoisyInt(o, r) for o, r in zip(obs, roots)]


def _noise(value):
    nodes = [n for n in value._root.closure() if isinstance(n, NoiseNode)]
    return nodes[0] if nodes else None


def test_a_block_totalling_zero_empties_every_cell_in_it():
    universe = _universe()
    known = ExactHistogram(
        marginal_coarsening(universe, {"hhgq": (0, 0, 1, 1)}),
        np.array([[0], [20]]))
    _, values = _cells(universe, ("detailed", "*"), [3, -1, 8, 9], [known])
    assert [int(v) for v in values[:2]] == [0, 0]
    assert all(_noise(v) is None for v in values[:2])


def test_one_free_cell_becomes_a_point_mass_at_the_block_total():
    universe = _universe()
    known = ExactHistogram(
        marginal_coarsening(universe, {"hhgq": (0, 1, 2, 3)}),
        np.array([[40], [0], [0], [0]]))
    _, values = _cells(universe, ("detailed", "*"), [37, 1, 0, 2], [known])
    assert int(values[0]) == 40 and _noise(values[0]) is None


def test_two_free_cells_stay_anti_correlated_around_the_exact_total():
    from noisyvalue.core import sample_noisy_values
    universe = _universe()
    known = ExactHistogram(
        marginal_coarsening(universe, {}), np.array([[100]]))
    _, values = _cells(universe, ("*", "detailed"), [30, 80], [known])
    a, b = sample_noisy_values(*values, n=300, rng=7)
    assert np.all(a.draws + b.draws == 100)
    assert len(np.unique(a.draws)) > 1
    # posterior centre is (obs_a - obs_b + total) / 2
    assert abs(a.mean() - 25) < 1.0


def test_three_free_cells_keep_the_total_only_as_a_bound_and_say_so():
    universe = _universe()
    known = ExactHistogram(
        marginal_coarsening(universe, {"hhgq": (0, 0, 0, 1)}),
        np.array([[12], [5]]))
    resolution, values = _cells(universe, ("detailed", "*"), [4, 5, 6, 5], [known])
    weakened = [n for n in resolution.notes if n["kind"].startswith("exact total")]
    assert weakened and weakened[0]["cells"] == 3
    assert "bound only" in weakened[0]["detail"]
    draws = values[0].sample(500, rng=2).draws
    assert draws.max() <= 12 and draws.min() >= 0


class _EqualSplit(BlockStrategy):
    """A stand-in for the lattice sampler the design leaves as future work."""

    name = "equal split"

    def build(self, family, obs, variance, total, lo, hi, nonnegative):
        from noisyvalue.dataset.solve import exact_count
        share = total // len(obs)
        return {i: exact_count(share) for i in range(len(obs))}


def test_the_large_block_construction_is_pluggable():
    universe = _universe()
    known = ExactHistogram(
        marginal_coarsening(universe, {"hhgq": (0, 0, 0, 1)}),
        np.array([[12], [5]]))
    resolution, values = _cells(universe, ("detailed", "*"), [4, 5, 6, 5],
                                [known], strategy=_EqualSplit())
    assert [int(v) for v in values[:3]] == [4, 4, 4]
    assert not [n for n in resolution.notes if n["kind"].startswith("exact total")]


def test_structural_zeros_are_taken_off_a_block_total_before_the_ladder():
    universe = _universe()
    zero = ZeroRegion(Region(({1}, None)), source="rule")   # no prisons here
    known = ExactHistogram(
        marginal_coarsening(universe, {"hhgq": (0, 0, 0, 1)}),
        np.array([[70], [0]]), source="known")
    # the block spans house, prison and dorm, but the rule empties prison, so
    # what is left is a two-cell block and gets the closed form, not a bound
    from noisyvalue.core import sample_noisy_values
    _, values = _cells(universe, ("detailed", "*"), [30, 2, 45, 1],
                       [zero, known])
    assert int(values[1]) == 0
    a, b = sample_noisy_values(values[0], values[2], n=200, rng=3)
    assert np.all(a.draws + b.draws == 70)
    assert len(np.unique(a.draws)) > 1
    assert abs(a.mean() - 27.5) < 1.0


def test_nonnegativity_is_the_default_and_can_be_declined():
    universe = _universe()
    _, values = _cells(universe, ("*", "*"), [-40], [])
    assert values[0].sample(200, rng=1).draws.min() >= 0
    _, values = _cells(universe, ("*", "*"), [-40], [], nonnegative=False)
    assert values[0].sample(200, rng=1).draws.min() < 0


def test_a_bound_outside_the_sampled_support_is_dropped_as_inert():
    universe = _universe()
    huge = BoundedHistogram(marginal_coarsening(universe, {}),
                            hi=np.array([[10 ** 9]]))
    _, values = _cells(universe, ("*", "*"), [500], [huge])
    # 0 and 10**9 are both hundreds of sd away, so neither can bind
    node = _noise(values[0])
    assert isinstance(node, DiscreteGaussianNode)
    assert node.low == -oo and node.high == oo

    tight = BoundedHistogram(marginal_coarsening(universe, {}),
                             hi=np.array([[505]]))
    _, values = _cells(universe, ("*", "*"), [500], [tight])
    node = _noise(values[0])
    assert isinstance(node, DiscreteGaussianNode)
    # theta <= 505 becomes a lower bound in eps-space (eps = obs - theta)
    assert node.low != -oo
    assert values[0].sample(400, rng=1).draws.max() <= 505


def test_dropping_an_inert_bound_leaves_the_posterior_unchanged():
    universe = _universe()
    family = DiscreteGaussianFamily()
    plain = family.cell(500, 16.0)
    bounded = family.cell(500, 16.0, lo=0, hi=10 ** 9)
    assert np.array_equal(plain.sample(500, rng=4).draws,
                          bounded.sample(500, rng=4).draws)


def test_a_measurement_with_no_released_query_must_be_fully_pinned():
    universe = _universe()
    m = Measurement("m", universe, ("*", "*"))
    resolution = solve(m, None, 0.0, ())
    with pytest.raises(ValueError, match="every cell must be pinned"):
        build_values(resolution, None, 0.0)


def test_solve_rejects_a_value_array_that_does_not_match_the_shape():
    universe = _universe()
    m = Measurement("m", universe, ("detailed", "*"))
    with pytest.raises(ValueError, match="do not match its shape"):
        solve(m, [1, 2, 3], 1.0, ())


def test_bound_only_is_the_default_strategy():
    assert BoundOnly().build(None, [1, 2, 3], 1.0, 6, None, None, True) is None


# ── the view ─────────────────────────────────────────────────────────────────

class _ListSource(Source):
    """A source whose measurements are handed to it, for testing `View` alone."""

    def __init__(self, values, evidence=()):
        self.values = values          # {measurement name: [cell values]}
        self.evidence = tuple(evidence)
        self.reads = 0

    def levels(self):
        return ("here",)

    def read_measurements(self, level, selection, measurements):
        from noisyvalue.dataset import MeasurementRow
        self.reads += 1
        for m in measurements:
            if m.name in self.values:
                yield MeasurementRow("area", m.name, self.values[m.name], 4.0)

    def read_evidence(self, level, selection):
        return {"area": self.evidence}

    def geo_columns(self, level, selection, geo_keys):
        return {"area": list(geo_keys)}


def _view_product(source, measurements):
    universe = _universe()
    return Product("p", universe, measurements,
                   (GeoLevel("here", native="here"),), source)


def test_a_view_returns_the_same_nodes_for_a_repeated_measurement():
    universe = _universe()
    m = Measurement("m", universe, ("detailed", "*"), native="m_dpq")
    source = _ListSource({"m": [1, 2, 3, 4]})
    view = _view_product(source, [m]).view("here")
    first, second = view.query("m"), view.query("m")
    assert all(a._root is b._root
               for a, b in zip(first["value"], second["value"]))
    assert source.reads == 1          # and the source is not walked twice


def test_two_measurements_naming_the_same_cells_share_them():
    universe = _universe()
    a = Measurement("a", universe, ("pairs", "*"), native="a_dpq")
    b = Measurement("b", universe, ("pairs", "*"), native="b_dpq")  # same cells
    source = _ListSource({"a": [10, 20], "b": [11, 19]})
    view = _view_product(source, [a, b]).view("here")
    first, second = view.query("a"), view.query("b")
    assert all(x._root is y._root
               for x, y in zip(first["value"], second["value"]))
    # b's own measurement was therefore not used, and the view says so
    report = view.unused_evidence()
    assert report.empty


def test_differently_binned_measurements_share_nothing_and_report_nothing():
    universe = _universe()
    fine = Measurement("fine", universe, ("detailed", "*"), native="f")
    coarse = Measurement("coarse", universe, ("pairs", "*"), native="c")
    source = _ListSource({"fine": [1, 2, 3, 4], "coarse": [3, 7]})
    view = _view_product(source, [fine, coarse]).view("here")
    view.queries(["fine", "coarse"])
    assert view.cells == 6                     # no cell key is named twice
    assert view.unused_evidence().empty


def test_partially_overlapping_measurements_are_reported_not_reconciled():
    universe = _universe()
    universe.axis("hhgq").declare("split", ((0, 1), (2,), (3,)),
                                  ("a", "dorm", "ship"))
    pairs = Measurement("pairs", universe, ("pairs", "*"), native="p")
    split = Measurement("split", universe, ("split", "*"), native="s")
    # both name the house+prison cell; only `split` breaks dorm from ship, so
    # one of the three cells collides and two are new
    source = _ListSource({"pairs": [30, 12], "split": [31, 5, 8]})
    view = _view_product(source, [pairs, split]).view("here")
    first = view.query("pairs")
    second = view.query("split")

    report = view.unused_evidence()
    assert list(report["kind"]) == ["overlapping measurements not reconciled"]
    assert int(report["cells"].iloc[0]) == 1
    # each measurement keeps its own posterior for the cell they both name
    assert first["value"].array[0]._root is not second["value"].array[0]._root
    assert int(first["value"].array[0]) == 30
    assert int(second["value"].array[0]) == 31


def test_a_view_holds_only_the_cells_it_has_been_asked_for():
    universe = _universe()
    small = Measurement("small", universe, ("*", "*"), native="s")
    big = Measurement("big", universe, ("detailed", "detailed"), native="b")
    source = _ListSource({"small": [100], "big": list(range(8))})
    view = _view_product(source, [small, big]).view("here")
    assert view.cells == 0
    view.query("small")
    assert view.cells == 1
    view.query("big")
    assert view.cells == 9
