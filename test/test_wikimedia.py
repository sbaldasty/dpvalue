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
from noisyvalue.pandas import NoisyIntArray


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
    assert wm.mechanism(d("2023-05-02"), "US").threshold == 450
    assert wm.mechanism(d("2023-05-02"), "FR").threshold == 90
    assert wm.mechanism(d("2023-09-19"), "US").threshold == 450
    assert wm.mechanism(d("2023-09-20"), "US").threshold == 90


def test_the_us_gap_was_intermittent():
    # on 48 of the gap's 226 days the pipeline succeeded, and US rows carry
    # the ordinary lower-risk Gaussian mechanism
    assert wm.mechanism(d("2023-05-01"), "US").threshold == 90
    assert wm.mechanism(d("2023-09-01"), "US").threshold == 90
    assert len(wm.US_GAUSSIAN_DAYS) == 48
    for day in wm.US_GAUSSIAN_DAYS:
        assert wm.US_GAP[0] <= day <= wm.US_GAP[1]


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
    write("country_project_page", "2023-05-02", [
        ("United States", "US", "en.wikipedia", 555, "Earth", "Q2", 4021),
        ("France", "FR", "fr.wikipedia", 666, "Terre", "Q2", 512),
    ])
    write("country_project_page_historical", "2018-06-01", [
        ("Germany", "DE", "de.wikipedia", 777, "Erde", "Q2", 913),
    ])
    # the pre-2017 layout: six columns, page_id always empty, no item_id
    write("country_project_page_historical_pre_2017", "2016-01-01", [
        ("Canada", "CA", "en.wikipedia", "", "2016", 4360),
    ])
    return str(tmp_path)


def test_get_pageviews_returns_a_tidy_frame(root):
    frame = wm.get_pageviews("2024-03-01", root=root)

    assert list(frame.columns) == [
        "date", "country", "country_code", "project", "page_id",
        "page_title", "item_id", "mechanism", "value", "variance",
        "censored"]
    assert len(frame) == 4
    assert isinstance(frame["value"].array, NoisyIntArray)
    assert list(frame["value"].array._obs) == [205, 149, 780, 1420]
    assert set(frame["date"].dt.date) == {d("2024-03-01")}
    assert not frame["censored"].any()


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
    frame = wm.get_pageviews("2023-05-02", root=root)
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
    item = wm.get_pageviews("2023-05-02", item="Q2", root=root)
    assert len(item) == 2
    assert wm.get_pageviews("2024-03-01", country="Narnia", root=root).empty


def test_pre_2017_files_have_no_page_ids(root):
    frame = wm.get_pageviews("2016-01-01", root=root)

    assert frame["page_id"].isna().all()
    assert frame["item_id"].isna().all()
    assert list(frame["page_title"]) == ["2016"]
    assert frame["variance"].iloc[0] == pytest.approx(
        DiscreteLaplaceFamily.variance_from_scale(300.0))
    # a title-keyed cell is still one random variable across reads
    again = wm.get_pageviews("2016-01-01", page="2016", root=root)
    batch_a, batch_b = sample_noisy_values(
        frame["value"][0], again["value"][0], n=100, rng=9)
    assert np.array_equal(batch_a.draws, batch_b.draws)
    # and an item filter simply selects nothing there
    assert wm.get_pageviews("2016-01-01", item="Q1", root=root).empty


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


def test_missing_cells_can_be_read_as_censored_posteriors(root):
    frame = wm.get_pageviews("2024-03-01", country=["BD", "IS"],
                             project="bn.wikipedia", page="ঢাকা",
                             missing="censored", root=root)
    frame = frame.set_index("country_code")

    # Bangladesh released the row; Iceland's absence becomes a posterior
    assert not frame.loc["BD", "censored"]
    assert frame.loc["IS", "censored"]
    value = frame.loc["IS", "value"]
    draws = value.sample(2000, rng=5).draws
    assert draws.min() >= 0
    assert draws.max() <= 89 + 110               # threshold plus the shoulder grid
    assert draws.mean() == pytest.approx(45, abs=10)
    assert abs(value._obs - np.median(draws)) <= 5   # fabricated obs ~ median


def test_censored_reads_are_one_random_variable(root):
    kwargs = dict(country=["IS"], project="bn.wikipedia", page="ঢাকা",
                  missing="censored", root=root)
    a = wm.get_pageviews("2024-03-01", **kwargs)["value"][0]
    b = wm.get_pageviews("2024-03-01", **kwargs)["value"][0]
    batch_a, batch_b = sample_noisy_values(a, b, n=200, rng=6)
    assert np.array_equal(batch_a.draws, batch_b.draws)


def test_censored_mode_validates_its_request(root):
    with pytest.raises(ValueError, match="explicit countries"):
        wm.get_pageviews("2024-03-01", page="X", missing="censored", root=root)
    with pytest.raises(ValueError, match="exactly one"):
        wm.get_pageviews("2024-03-01", country=["IS"],
                         project="is.wikipedia", missing="censored", root=root)
    with pytest.raises(ValueError, match="ISO alpha-2"):
        wm.get_pageviews("2024-03-01", country=["Iceland"],
                         project="is.wikipedia", page="X",
                         missing="censored", root=root)
    with pytest.raises(ValueError, match="not published"):
        wm.get_pageviews("2024-03-01", country=["MO"],
                         project="zh.wikipedia", page="X",
                         missing="censored", root=root)


def test_censored_mode_refuses_pre_tiers_excluded_countries(root):
    # Russia's absence in 2018 reflects the protection policy of the time,
    # not thresholding, so no censored posterior exists
    with pytest.raises(ValueError, match="protection policy"):
        wm.get_pageviews("2018-06-01", country=["RU"],
                         project="ru.wikipedia", page="X",
                         missing="censored", root=root)


def test_days_beyond_the_codebook_validation_are_warned_about(root):
    future = wm.CODEBOOK_AS_OF + dt.timedelta(days=30)
    with pytest.warns(UserWarning, match="postdates the codebook"):
        with pytest.raises(FileNotFoundError):
            wm.get_pageviews(future, root=root)


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


# ── integration against the real releases ────────────────────────────────────
#
# The minimum released count in a group of rows reveals the threshold that
# suppressed its neighbours -- 90 / 450 / 550 / 1000 / 3500 -- and thereby
# which mechanism made it, so these tests check the codebook's routing
# against the files themselves.  They read days mirrored into the default
# root and are skipped without them; one call prepares everything:
#
#   from noisyvalue import fetch_pageviews
#   fetch_pageviews(["2016-01-01", "2018-06-01", "2023-05-01", "2023-06-01",
#                    "2023-11-17", "2024-02-15", "2026-01-26"])

def _mirrored(day):
    release = wm.release_for(d(day))
    path = os.path.join(wm.DEFAULT_ROOT, release.dataset, f"{day}.tsv")
    return pytest.mark.skipif(
        not os.path.exists(path),
        reason=f"{day} not mirrored; run the fetch_pageviews call in the "
               "comment above this test's section")


def _floor(frame, code=None):
    """Smallest released count, optionally for one country's rows."""
    obs = frame["value"].array._obs
    if code is not None:
        obs = obs[(frame["country_code"] == code).to_numpy()]
    return int(obs.min())


_BIG = ["US", "GB", "DE", "FR", "JP"]        # never below their thresholds


@_mirrored("2016-01-01")
def test_live_pre_2017_floor_sits_at_its_threshold():
    frame = wm.get_pageviews("2016-01-01", country=_BIG)
    assert 3500 <= _floor(frame) < 3700


@_mirrored("2018-06-01")
def test_live_historical_floor_sits_at_its_threshold():
    frame = wm.get_pageviews("2018-06-01", country=_BIG)
    assert 450 <= _floor(frame) < 520


@_mirrored("2023-06-01")
def test_live_us_gap_backfill_days_have_the_geometric_floor():
    frame = wm.get_pageviews("2023-06-01", country=_BIG)
    assert 450 <= _floor(frame, "US") < 520
    assert 90 <= _floor(frame, "FR") < 130
    assert 90 <= _floor(frame, "DE") < 130


@_mirrored("2023-05-01")
def test_live_us_gap_exception_days_have_the_ordinary_floor():
    frame = wm.get_pageviews("2023-05-01", country=_BIG)
    assert 90 <= _floor(frame, "US") < 130
    assert 90 <= _floor(frame, "FR") < 130


@_mirrored("2023-11-17")
def test_live_patch_days_are_wholly_geometric():
    frame = wm.get_pageviews("2023-11-17", country=_BIG)
    assert 450 <= _floor(frame) < 520


@_mirrored("2024-02-15")
def test_live_tier_debut_floors():
    frame = wm.get_pageviews("2024-02-15", country=["US", "BD", "RU"])
    assert 90 <= _floor(frame, "US") < 130
    assert 550 <= _floor(frame, "BD") < 650
    assert 1000 <= _floor(frame, "RU") < 1150


@_mirrored("2026-01-26")
def test_live_recalibration_floors():
    frame = wm.get_pageviews("2026-01-26", country=["PK", "KW", "RU"])
    assert 1000 <= _floor(frame, "PK") < 1150       # medium -> higher
    assert _floor(frame, "KW") < 550                # medium -> lower
    assert not (frame["country_code"] == "RU").any()    # higher -> not published


@_mirrored("2018-06-01")
def test_live_historical_country_sum_is_commensurate_with_the_public_count():
    """Suppression and per-user clipping only remove views, so summing the
    released countries must stay below the public API's exact figure while
    still carrying the bulk of it.  Catches gross misreads (wrong column,
    wrong scale), not subtle ones.

    The ceiling is the *all-agents* figure: the historical pipeline
    evidently did not filter automated traffic (Earth on 2018-06-01 shows
    data-center-sized counts from NL and SG, and the country sum lands on
    the all-agents total at six times the human one)."""
    frame = wm.get_pageviews("2018-06-01", page="Earth", project="en.wikipedia")
    assert len(frame) > 0
    dp_sum = int(frame["value"].array._obs.sum())
    exact = _public_count("en.wikipedia", "Earth", "20180601", "all-agents")
    assert 0.2 * exact < dp_sum < 1.1 * exact


@_mirrored("2023-05-01")
def test_live_current_country_sum_is_commensurate_with_the_public_count():
    """The current era counts each device's first ten unique views via a
    browser cookie, so unlike the historical era it tracks human traffic;
    its sum sits below even the API's user-only figure."""
    frame = wm.get_pageviews("2023-05-01", page="Earth", project="en.wikipedia")
    assert len(frame) > 0
    dp_sum = int(frame["value"].array._obs.sum())
    exact = _public_count("en.wikipedia", "Earth", "20230501", "user")
    assert 0.2 * exact < dp_sum < 1.1 * exact


def _public_count(project, title, yyyymmdd, agent):
    import json
    import urllib.error
    import urllib.request
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           f"{project}/all-access/{agent}/{title}/daily/{yyyymmdd}/{yyyymmdd}")
    request = urllib.request.Request(
        url, headers={"User-Agent":
                      "noisyvalue tests (https://github.com/sbaldasty/noisyvalue)"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        pytest.skip(f"pageviews API unreachable: {error}")
    return int(payload["items"][0]["views"])


@pytest.mark.skipif(not os.path.isdir(wm.DEFAULT_ROOT),
                    reason="wikimedia pageview mirror not present")
def test_live_codebook_matches_the_current_protection_list():
    """The canary for list drift: when Wikimedia recalibrates the Country
    and Territory Protection List, this fails, and the fix is a new dated
    entry in `_TIER_VERSIONS` (pin its effective day by probing released
    files around the revision, as test_wikimedia's header describes)."""
    import csv
    import urllib.error
    import urllib.request
    url = ("https://gitlab.wikimedia.org/repos/movement-insights/"
           "canonical-data/-/raw/main/country/countries.tsv")
    request = urllib.request.Request(
        url, headers={"User-Agent":
                      "noisyvalue tests (https://github.com/sbaldasty/noisyvalue)"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as error:
        pytest.skip(f"canonical-data repository unreachable: {error}")

    listed = {"Medium risk": set(), "Higher risk": set(), "Not published": set()}
    for row in csv.DictReader(text.splitlines(), delimiter="\t"):
        if row["data_risk_classification"] in listed:
            listed[row["data_risk_classification"]].add(row["iso_code"])

    current = wm._TIER_VERSIONS[-1][1]
    assert listed["Medium risk"] == set(current["medium"])
    assert listed["Higher risk"] == set(current["higher"])
    assert listed["Not published"] == set(current["not published"])
