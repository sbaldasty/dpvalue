"""Codebook tests for the Wikimedia pageview releases.

Every boundary asserted here was validated against the released files
themselves: the minimum released count in a group of rows reveals its
threshold (90 / 450 / 550 / 1000 / 3500), and hence which mechanism made
it, so era edges, the US gap, the patch days, and each tier-list revision
were all confirmed by probing the days on either side.
"""

import datetime as dt

import numpy as np
import pytest

from noisyvalue import wikimedia as wm
from noisyvalue.dataset import DiscreteLaplaceFamily


def d(s):
    return dt.date.fromisoformat(s)


# ── era routing ──────────────────────────────────────────────────────────────

def test_each_day_routes_to_its_release():
    assert wm.release_for(d("2015-07-01")).name == "pre-2017"
    assert wm.release_for(d("2017-02-08")).name == "pre-2017"
    assert wm.release_for(d("2017-02-09")).name == "historical"
    assert wm.release_for(d("2023-02-05")).name == "historical"
    assert wm.release_for(d("2023-02-06")).name == "current"
    assert wm.release_for(d("2026-08-01")).name == "current"
    assert wm.release_for(d("2015-06-30")) is None


def test_mechanism_refuses_days_before_any_release():
    with pytest.raises(ValueError, match="no pageview release"):
        wm.mechanism(d("2015-06-30"), "FR")


def test_release_urls_point_at_the_daily_tsvs():
    release = wm.release_for(d("2018-01-01"))
    assert release.url(d("2018-01-01")) == (
        "https://analytics.wikimedia.org/published/datasets/"
        "country_project_page_historical/2018-01-01.tsv")


def test_the_never_released_days_are_documented():
    assert d("2020-07-19") in wm.MISSING_DAYS
    assert len(wm.MISSING_DAYS) == 5
    for day in wm.MISSING_DAYS:
        assert wm.release_for(day).name in ("historical",)


# ── the two geometric eras ───────────────────────────────────────────────────

def test_the_historical_eras_are_geometric():
    m = wm.mechanism(d("2018-01-01"), "FR")
    assert isinstance(m.family, DiscreteLaplaceFamily)
    assert m.threshold == 450
    assert m.family.scale_from_variance(m.variance) == pytest.approx(30.0)

    m = wm.mechanism(d("2016-01-01"), "FR")
    assert isinstance(m.family, DiscreteLaplaceFamily)
    assert m.threshold == 3500
    assert m.family.scale_from_variance(m.variance) == pytest.approx(300.0)


# ── the current era's Gaussian tiers ─────────────────────────────────────────

def test_gaussian_variances_reproduce_the_readme_confidence_intervals():
    # sigma^2 = 10 / (2 rho) is not stated in the release documentation, but
    # 1.96 sigma landing on the README's own 95% intervals for all three
    # tiers confirms both the formula and the sqrt(10) L2 sensitivity
    for tier, interval in (("lower", 35.7), ("medium", 176.5),
                           ("higher", 352.5)):
        variance = wm._GAUSSIAN[tier].variance
        assert 1.96 * np.sqrt(variance) == pytest.approx(interval, abs=0.05)


def test_us_rows_in_the_gap_carry_the_historical_mechanism():
    assert wm.mechanism(d("2023-05-01"), "US").threshold == 450
    assert wm.mechanism(d("2023-05-01"), "FR").threshold == 90
    assert wm.mechanism(d("2023-09-19"), "US").threshold == 450
    assert wm.mechanism(d("2023-09-20"), "US").threshold == 90


def test_patch_days_carry_the_historical_mechanism_for_every_country():
    assert wm.mechanism(d("2023-11-17"), "FR").threshold == 450
    assert wm.mechanism(d("2023-11-16"), "FR").threshold == 90


# ── the protection-list revisions ────────────────────────────────────────────

def test_tiered_release_begins_2024_02_15():
    # Bangladesh and Russia first appear on 2024-02-15, at the medium- and
    # higher-risk thresholds; Macao, published until then, vanishes
    assert wm.mechanism(d("2024-02-15"), "BD").threshold == 550
    assert wm.mechanism(d("2024-02-15"), "RU").threshold == 1000
    assert wm.tier(d("2024-02-15"), "MO") == "not published"
    assert wm.tier(d("2024-02-14"), "MO") == "lower"


def test_hong_kong_is_removed_2024_05_19():
    assert wm.tier(d("2024-05-18"), "HK") == "lower"
    assert wm.tier(d("2024-05-19"), "HK") == "not published"


def test_the_2026_recalibration_takes_effect_2026_01_26():
    assert wm.tier(d("2026-01-25"), "PK") == "medium"
    assert wm.tier(d("2026-01-26"), "PK") == "higher"
    assert wm.tier(d("2026-01-25"), "KW") == "medium"
    assert wm.tier(d("2026-01-26"), "KW") == "lower"
    assert wm.tier(d("2026-01-25"), "RU") == "higher"
    assert wm.tier(d("2026-01-26"), "RU") == "not published"


def test_turkey_never_appears_despite_its_listing():
    for day in ("2016-01-01", "2020-01-01", "2024-03-01", "2026-05-01"):
        with pytest.raises(ValueError, match="not published"):
            wm.mechanism(d(day), "TR")


def test_not_published_countries_are_refused():
    with pytest.raises(ValueError, match="not published"):
        wm.mechanism(d("2026-02-01"), "RU")


# ── posteriors ───────────────────────────────────────────────────────────────

def test_a_released_count_becomes_a_nonnegative_posterior_at_the_observation():
    m = wm.mechanism(d("2024-03-01"), "BD")       # medium risk: sigma ~ 90
    cell = m.cell(600)
    draws = cell.sample(4000, rng=7).draws

    assert cell._obs == 600
    assert draws.min() >= 0
    assert draws.mean() == pytest.approx(600.0, rel=0.05)
    assert draws.std() == pytest.approx(np.sqrt(m.variance), rel=0.1)
