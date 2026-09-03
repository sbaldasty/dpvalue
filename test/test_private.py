import math

import numpy as np
import pytest

dp = pytest.importorskip("opendp.prelude")
dp.enable_features("contrib", "honest-but-curious")

from noisyvalue.core import NoisyFloat, NoisyInt
from noisyvalue.private import PrivacyUnit, PrivateDataset, rho_for_epsilon_delta


def make_units(n, rng):
    units = [PrivacyUnit(i) for i in range(n)]
    terms = []
    for u in units:
        w = u.private_float(rng.uniform(0.5, 2.0), lo=0.0, hi=2.0)
        x = u.private_float(rng.uniform(0.0, 100.0), lo=0.0, hi=100.0)
        terms.append(w * x)
    return units, terms


def test_weighted_sum_release_matches_calibrated_scale():
    rng = np.random.default_rng(0)
    units, terms = make_units(20, rng)
    ds = PrivateDataset(units)

    value = ds.release(terms, rho=0.1)

    assert isinstance(value, NoisyFloat)
    expected_scale = 200.0 / math.sqrt(2 * 0.1)  # sensitivity = hi_w * hi_x
    draws = value.sample(n=20000, rng=np.random.default_rng(1)).draws
    assert abs(draws.std() - expected_scale) < 0.05 * expected_scale
    assert abs(draws.mean() - float(value)) < 0.05 * expected_scale


def test_bool_count_release_is_discrete_with_sensitivity_one():
    rng = np.random.default_rng(0)
    units = [PrivacyUnit(i) for i in range(30)]
    flags = [u.private_bool(rng.random() < 0.4) for u in units]
    terms = [f.as_int() for f in flags]
    ds = PrivateDataset(units)

    value = ds.release(terms, rho=0.2)

    assert isinstance(value, NoisyInt)
    expected_scale = 1.0 / math.sqrt(2 * 0.2)
    draws = value.sample(n=20000, rng=np.random.default_rng(2)).draws
    assert abs(draws.std() - expected_scale) < 0.05 * expected_scale + 0.05


def test_release_debits_rho_and_enforces_budget():
    rng = np.random.default_rng(0)
    units, terms = make_units(5, rng)
    ds = PrivateDataset(units, rho_budget=0.5)

    ds.release(terms, rho=0.3)
    assert ds.rho_spent == pytest.approx(0.3)

    with pytest.raises(ValueError, match="budget"):
        ds.release(terms, rho=0.3)
    assert ds.rho_spent == pytest.approx(0.3)  # failed release does not debit


def test_release_rejects_terms_spanning_multiple_units():
    rng = np.random.default_rng(0)
    units, terms = make_units(3, rng)
    ds = PrivateDataset(units)
    mixed = terms[0] + terms[1]

    with pytest.raises(ValueError, match="exactly one unit"):
        ds.release([mixed], rho=0.1)


def test_release_rejects_units_outside_the_dataset():
    rng = np.random.default_rng(0)
    units, _ = make_units(3, rng)
    ds = PrivateDataset(units)
    outsider = PrivacyUnit("outsider").private_float(1.0, lo=0.0, hi=1.0)

    with pytest.raises(ValueError, match="not a member"):
        ds.release([outsider], rho=0.1)


def test_release_rejects_empty_terms():
    ds = PrivateDataset([PrivacyUnit(0)])
    with pytest.raises(ValueError):
        ds.release([], rho=0.1)


def test_shared_key_tightens_posterior_across_releases():
    # A registry-backed value is a live view of its key's current posterior
    # (noisyvalue.opendp.SharedLatentRegistry), so `first` tightens in place
    # once a second measurement of the same key is folded in -- there is no
    # separate "first" and "second" posterior to compare after the fact.
    rng = np.random.default_rng(0)
    units, terms = make_units(20, rng)
    ds = PrivateDataset(units)

    first = ds.release(terms, rho=0.1, key="total")
    before_std = first.sample(n=20000, rng=np.random.default_rng(3)).draws.std()

    ds.release(terms, rho=0.1, key="total")
    after_std = first.sample(n=20000, rng=np.random.default_rng(4)).draws.std()

    assert after_std < before_std  # folding in the second measurement tightens it
    assert ds.rho_spent == pytest.approx(0.2)


def test_interval_arithmetic_bounds_propagate():
    unit = PrivacyUnit(0)
    a = unit.private_float(3.0, lo=0.0, hi=5.0)
    b = unit.private_float(2.0, lo=0.0, hi=5.0)

    total = a + b
    assert (total.lo, total.hi) == (0.0, 10.0)

    diff = a - b
    assert (diff.lo, diff.hi) == (-5.0, 5.0)

    scaled = a * 3
    assert (scaled.lo, scaled.hi) == (0.0, 15.0)

    negated = -a
    assert (negated.lo, negated.hi) == (-5.0, -0.0)


def test_clamp_tightens_bounds_and_clips_value():
    unit = PrivacyUnit(0)
    a = unit.private_float(9.0, lo=0.0, hi=10.0)

    clamped = a.clamp(0.0, 5.0)

    assert (clamped.lo, clamped.hi) == (0.0, 5.0)
    assert clamped.value == 5.0


def test_clamp_rejects_disjoint_bounds():
    unit = PrivacyUnit(0)
    a = unit.private_float(1.0, lo=0.0, hi=1.0)

    with pytest.raises(ValueError):
        a.clamp(2.0, 3.0)


def test_leaf_value_is_clipped_to_declared_bounds():
    unit = PrivacyUnit(0)
    v = unit.private_float(150.0, lo=0.0, hi=100.0)
    assert v.value == 100.0


def test_rho_for_epsilon_delta_round_trips_through_the_zcdp_bound():
    epsilon, delta = 1.0, 1e-6
    rho = rho_for_epsilon_delta(epsilon, delta)
    achieved = rho + 2 * math.sqrt(rho * math.log(1.0 / delta))
    assert achieved == pytest.approx(epsilon)


def test_rho_for_epsilon_delta_validates_inputs():
    with pytest.raises(ValueError):
        rho_for_epsilon_delta(1.0, 0.0)
    with pytest.raises(ValueError):
        rho_for_epsilon_delta(0.0, 1e-6)
