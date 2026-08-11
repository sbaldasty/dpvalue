"""Codebook tests for the Wikimedia pageview releases.

Every boundary asserted here was validated against the released files
themselves: the minimum released count in a group of rows reveals its
threshold (90 / 450 / 550 / 1000 / 3500), and hence which mechanism made
it, so era edges, the US gap, the patch days, and each tier-list revision
were all confirmed by probing the days on either side.
"""

import datetime as dt
import io
import os

import numpy as np
import pytest

from noisyvalue import wikimedia as wm
from noisyvalue.core import sample_noisy_values
from noisyvalue.dataset import DiscreteLaplaceFamily
from noisyvalue.pandas_ext import NoisyIntArray


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


# ── reading mirrored files ───────────────────────────────────────────────────

@pytest.fixture
def root(tmp_path):
    def write(dataset, day, rows):
        folder = tmp_path / dataset
        folder.mkdir(parents=True, exist_ok=True)
        text = "".join("\t".join(str(f) for f in row) + "\n" for row in rows)
        (folder / f"{day}.tsv").write_text(text, encoding="utf-8")

    write("country_project_page", "2024-03-01", [
        ("Iceland", "IS", "is.wikipedia", 111, "Reykjavík", "Q1764", 205),
        ("Iceland", "IS", "en.wikipedia", 222, "Golden_Circle_(Iceland)",
         "Q1755509", 149),
        ("Bangladesh", "BD", "bn.wikipedia", 333, "ঢাকা",
         "Q1354", 780),
        ("Russia", "RU", "ru.wikipedia", 444, 'Page_with_"quotes"', "", 1420),
    ])
    write("country_project_page", "2023-05-01", [
        ("United States", "US", "en.wikipedia", 555, "Earth", "Q2", 4021),
        ("France", "FR", "fr.wikipedia", 666, "Terre", "Q2", 512),
    ])
    write("country_project_page_historical", "2018-06-01", [
        ("Germany", "DE", "de.wikipedia", 777, "Erde", "Q2", 913),
    ])
    return str(tmp_path)


def test_get_pageviews_returns_a_tidy_frame(root):
    frame = wm.get_pageviews("2024-03-01", root=root)

    assert list(frame.columns) == [
        "date", "country", "country_code", "project", "page_id",
        "page_title", "item_id", "mechanism", "value", "variance"]
    assert len(frame) == 4
    assert isinstance(frame["value"].array, NoisyIntArray)
    assert list(frame["value"].array._obs) == [205, 149, 780, 1420]
    assert set(frame["date"].dt.date) == {d("2024-03-01")}


def test_variance_follows_the_tier_and_the_era(root):
    frame = wm.get_pageviews("2024-03-01", root=root)
    got = dict(zip(frame["country_code"], frame["variance"]))
    assert got["IS"] == pytest.approx(10 / (2 * 1.505e-2))
    assert got["BD"] == pytest.approx(10 / (2 * 6.166e-4))
    assert got["RU"] == pytest.approx(10 / (2 * 1.546e-4))

    frame = wm.get_pageviews("2018-06-01", root=root)
    assert frame["variance"].iloc[0] == pytest.approx(
        DiscreteLaplaceFamily.variance_from_scale(30.0))


def test_us_gap_rows_carry_the_geometric_mechanism(root):
    frame = wm.get_pageviews("2023-05-01", root=root)
    mechs = dict(zip(frame["country_code"], frame["mechanism"]))
    assert "geometric" in mechs["US"]
    assert "Gaussian" in mechs["FR"]


def test_filters_match_names_codes_titles_projects_and_items(root):
    assert len(wm.get_pageviews("2024-03-01", country="is", root=root)) == 2
    assert len(wm.get_pageviews("2024-03-01", country="Iceland", root=root)) == 2
    page = wm.get_pageviews("2024-03-01", page="Golden Circle (Iceland)",
                            root=root)
    assert list(page["page_id"]) == [222]
    proj = wm.get_pageviews("2024-03-01",
                            project=["is.wikipedia", "bn.wikipedia"], root=root)
    assert len(proj) == 2
    item = wm.get_pageviews("2023-05-01", item="Q2", root=root)
    assert len(item) == 2
    assert wm.get_pageviews("2024-03-01", country="Narnia", root=root).empty


def test_quotes_in_titles_are_read_literally(root):
    frame = wm.get_pageviews("2024-03-01", country="RU", root=root)
    assert list(frame["page_title"]) == ['Page_with_"quotes"']
    assert frame["item_id"].isna().all()


def test_values_sample_around_the_released_count(root):
    value = wm.get_pageviews("2024-03-01", country="BD", root=root)["value"][0]
    draws = value.sample(2000, rng=11).draws

    assert draws.min() >= 0
    assert draws.mean() == pytest.approx(780, abs=10)


def test_repeated_reads_are_one_random_variable(root):
    a = wm.get_pageviews("2024-03-01", country="BD", root=root)["value"][0]
    b = wm.get_pageviews("2024-03-01", country="BD", root=root)["value"][0]
    batch_a, batch_b = sample_noisy_values(a, b, n=200, rng=5)

    assert np.array_equal(batch_a.draws, batch_b.draws)


def test_different_cells_are_independent_random_variables(root):
    frame = wm.get_pageviews("2024-03-01", country="IS", root=root)
    batch_a, batch_b = sample_noisy_values(
        frame["value"][0], frame["value"][1], n=200, rng=5)

    assert not np.array_equal(batch_a.draws, batch_b.draws)


def test_get_refuses_an_unmirrored_day(root):
    with pytest.raises(FileNotFoundError, match="fetch_pageviews"):
        wm.get_pageviews("2024-03-02", root=root)


def test_get_refuses_a_never_released_day(root):
    with pytest.raises(ValueError, match="never released"):
        wm.get_pageviews("2020-07-19", root=root)


def test_days_between_skips_the_never_released_days():
    days = wm.days_between("2020-07-17", "2020-07-22")

    assert d("2020-07-19") not in days
    assert d("2020-07-20") not in days
    assert len(days) == 4


# ── mirroring ────────────────────────────────────────────────────────────────

class _Response:
    def __init__(self, data):
        self._data = io.BytesIO(data)

    def read(self, n=-1):
        return self._data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_mirrors_into_the_release_layout(monkeypatch, tmp_path):
    payload = b"Germany\tDE\tde.wikipedia\t777\tErde\tQ2\t913\n"

    def fake_urlopen(request, timeout=None):
        assert request.full_url == (
            "https://analytics.wikimedia.org/published/datasets/"
            "country_project_page_historical/2018-06-01.tsv")
        return _Response(payload)

    monkeypatch.setattr(wm.urllib.request, "urlopen", fake_urlopen)
    paths = wm.fetch_pageviews("2018-06-01", root=str(tmp_path))

    expected = os.path.join(
        str(tmp_path), "country_project_page_historical", "2018-06-01.tsv")
    assert paths == [expected]
    with open(expected, "rb") as f:
        assert f.read() == payload

    def refuse(*args, **kwargs):
        raise AssertionError("an already-mirrored day was re-downloaded")

    monkeypatch.setattr(wm.urllib.request, "urlopen", refuse)
    assert wm.fetch_pageviews("2018-06-01", root=str(tmp_path)) == paths


def test_fetch_refuses_a_never_released_day(tmp_path):
    with pytest.raises(ValueError, match="never released"):
        wm.fetch_pageviews("2022-10-19", root=str(tmp_path))
