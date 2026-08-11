"""The Wikimedia differentially-private pageview releases: the codebook.

Since 2015 Wikimedia has published daily counts of pageviews per
(country, project, page), protected by differential privacy and served as
plain daily TSVs from analytics.wikimedia.org.  This module knows *what the
numbers mean* -- which noise mechanism produced each row, at what scale,
and under which suppression threshold -- and reads the files in those
terms: `fetch_pageviews` mirrors the days you name into a local root, and
`get_pageviews` returns them as a tidy DataFrame whose `value` column holds
the posterior of each true count.

Unlike `census.py`, repeated reads agree: a cell's noise symbol is derived
from its identity (day, country, project, page), so asking for the same
cell twice yields the same random variable, and expressions combining the
two sample consistently.  Cells never overlap across rows -- each
(country, project, page, day) is measured exactly once, and there are no
marginal queries -- so this is all the joint structure the release has.

What a count measures: each device contributes at most its first 10 unique
pageviews per day (the two historical eras, which predate the client-side
counting cookie, instead assumed per-user daily bounds of 30 and 300).  The
latent quantity is therefore a deduplicated, clipped pageview count --
systematically below the exact global counts of the public pageview API.
The eras also differ in *whose* views they count: the cookie implies a real
browser, and the current era's country sums sit below the API's user-only
figures, while the historical pipeline evidently did not filter automated
traffic -- its sums track the API's all-agents totals, with data-center
countries (NL, SG) sometimes carrying most of a page's count.

There is one release directory per era, each a daily TSV of
(country, country_code, project, page_id, page_title, item_id, count):

| era        | dates                   | noise                    | threshold |
| ---------- | ----------------------- | ------------------------ | --------- |
| pre-2017   | 2015-07-01 - 2017-02-08 | geometric, scale 300     | 3500      |
| historical | 2017-02-09 - 2023-02-05 | geometric, scale 30      | 450       |
| current    | 2023-02-06 - present    | discrete Gaussian (zCDP) | by tier   |

The current era's Gaussian scale and threshold depend on the country's risk
tier on the Country and Territory Protection List, and its files are not
uniformly Gaussian: two groups of rows were backfilled with the *historical*
mechanism after pipeline failures.  `mechanism()` owns all of this routing;
its boundary dates were validated empirically against the released files
(the minimum released count in a group of rows reveals its threshold, and
hence its mechanism -- see test_wikimedia.py).

Suppression: a row is released only when its noisy count clears the
threshold.  For a released row the flat-prior posterior is unaffected by
that selection, since the noisy value itself is observed; it is simply
`obs - eps`, truncated to nonnegative counts, which is what
`Mechanism.cell` builds.  An *absent* row is genuine information -- its
noisy count fell below the threshold -- but modeling that is out of scope
here.
"""

import datetime as dt
import hashlib
import os
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import polars as pl

from .dataset import DiscreteGaussianFamily, DiscreteLaplaceFamily
from .pandas_ext import NoisyIntArray

BASE_URL = "https://analytics.wikimedia.org/published/datasets"
DEFAULT_ROOT = "data/wikimedia-pageviews"

# TSV column order; the files carry no header row.  The pre-2017 release
# predates page ids: its files drop the item_id column outright and leave
# page_id empty, so there a page's identity is its title.
COLUMNS = ("country", "country_code", "project", "page_id", "page_title",
           "item_id", "count")


# ── mechanisms ───────────────────────────────────────────────────────────────

class Mechanism:
    """How one released row was noised, and the posterior that follows.

    `variance` is the noise's true sampling variance, the unit every
    `NoiseFamily` speaks; `threshold` is the release threshold the noisy
    count had to clear.
    """

    def __init__(self, family, variance, threshold, source):
        self.family = family
        self.variance = float(variance)
        self.threshold = int(threshold)
        self.source = source

    def __repr__(self):
        return f"<Mechanism {self.source!r}: threshold {self.threshold}>"

    def cell(self, obs, symbol=None):
        """The flat-prior posterior for a released count: nonnegative,
        centred at the observation."""
        return self.family.cell(int(obs), self.variance, lo=0, symbol=symbol)


# The pure epsilon-DP eras protect m daily pageviews per user at epsilon = 1,
# so the geometric scale is m / epsilon = m.
_LAPLACE_30 = Mechanism(
    DiscreteLaplaceFamily(), DiscreteLaplaceFamily.variance_from_scale(30.0),
    450, "geometric scale 30 (epsilon=1, m=30)")
_LAPLACE_300 = Mechanism(
    DiscreteLaplaceFamily(), DiscreteLaplaceFamily.variance_from_scale(300.0),
    3500, "geometric scale 300 (epsilon=1, m=300)")

# Gaussian zCDP: rho = sensitivity^2 / (2 sigma^2).  The L2 sensitivity is
# sqrt(10) -- one device touches at most 10 distinct cells, by 1 each -- so
# sigma^2 = 10 / (2 rho).  Cross-check: 1.96 sigma reproduces the release
# README's 95% intervals (35.7 / 176.5 / 352.5 pageviews) exactly.
_RHO = {"lower": 1.505e-2, "medium": 6.166e-4, "higher": 1.546e-4}
_TIER_THRESHOLD = {"lower": 90, "medium": 550, "higher": 1000}
_GAUSSIAN = {
    tier: Mechanism(
        DiscreteGaussianFamily(), 10.0 / (2.0 * rho), _TIER_THRESHOLD[tier],
        f"discrete Gaussian zCDP rho={rho} ({tier} risk)")
    for tier, rho in _RHO.items()}


# ── releases ─────────────────────────────────────────────────────────────────

_SCHEMA = {
    "country": pl.Utf8, "country_code": pl.Utf8, "project": pl.Utf8,
    "page_id": pl.Int64, "page_title": pl.Utf8, "item_id": pl.Utf8,
    "count": pl.Int64,
}
_PRE_2017_SCHEMA = {
    "country": pl.Utf8, "country_code": pl.Utf8, "project": pl.Utf8,
    "page_id": pl.Int64, "page_title": pl.Utf8, "count": pl.Int64,
}


class Release:
    """One release directory: a name, a URL path, and the days it serves."""

    def __init__(self, name, dataset, start, end, schema=_SCHEMA):
        self.name = name
        self.dataset = dataset
        self.start = start
        self.end = end                      # None while still being extended
        self.schema = schema

    def __repr__(self):
        return f"<Release {self.name!r}: {self.start} - {self.end or 'present'}>"

    def url(self, day):
        return f"{BASE_URL}/{self.dataset}/{day.isoformat()}.tsv"


RELEASES = (
    Release("pre-2017", "country_project_page_historical_pre_2017",
            dt.date(2015, 7, 1), dt.date(2017, 2, 8),
            schema=_PRE_2017_SCHEMA),
    Release("historical", "country_project_page_historical",
            dt.date(2017, 2, 9), dt.date(2023, 2, 5)),
    Release("current", "country_project_page",
            dt.date(2023, 2, 6), None),
)

# Days inside a release's range whose file was nevertheless never published.
MISSING_DAYS = frozenset({
    dt.date(2017, 6, 26), dt.date(2017, 7, 24),
    dt.date(2020, 7, 19), dt.date(2020, 7, 20),
    dt.date(2022, 10, 19),
})

# Pipeline failures in the current era, repaired afterwards with the
# *historical* mechanism (WMF's retention rules had already destroyed the
# data an exact re-run would have needed): seven whole days, plus most --
# not all -- United States rows for the first months of the era, which a
# database naming error kept dropping.
PATCH_DAYS = frozenset({
    dt.date(2023, 6, 19), dt.date(2023, 10, 25),
    dt.date(2023, 11, 13), dt.date(2023, 11, 17), dt.date(2023, 11, 19),
    dt.date(2023, 11, 23), dt.date(2023, 11, 27),
})

# The US gap is documented as 2023-02-06 through 2023-09-19, but it was not
# solid: on 48 of its 226 days the pipeline succeeded, and those files carry
# ordinary lower-risk Gaussian US rows.  There is no published list, so this
# one was derived by scanning the US minimum released count of every day in
# the range -- 450 marks the geometric backfill, ~90 the ordinary mechanism,
# and the two never blur -- and sampled days are re-checked in
# test_wikimedia.py.
US_GAP = (dt.date(2023, 2, 6), dt.date(2023, 9, 19))
US_GAUSSIAN_DAYS = frozenset(map(dt.date.fromisoformat, (
    "2023-02-19", "2023-02-21", "2023-02-28", "2023-03-05", "2023-03-07",
    "2023-03-09", "2023-03-12", "2023-03-13", "2023-03-14", "2023-03-19",
    "2023-03-26", "2023-03-28", "2023-03-31", "2023-04-01", "2023-04-02",
    "2023-04-03", "2023-04-25", "2023-04-26", "2023-04-28", "2023-05-01",
    "2023-05-08", "2023-05-12", "2023-05-19", "2023-05-23", "2023-05-24",
    "2023-06-11", "2023-06-14", "2023-06-16", "2023-06-28", "2023-07-05",
    "2023-07-07", "2023-07-08", "2023-07-10", "2023-07-11", "2023-07-12",
    "2023-07-25", "2023-07-26", "2023-07-31", "2023-08-05", "2023-08-06",
    "2023-08-11", "2023-08-16", "2023-08-19", "2023-08-23", "2023-08-29",
    "2023-08-30", "2023-08-31", "2023-09-01",
)))


def release_for(day):
    """The release directory serving a day, or None before the first one."""
    for release in RELEASES:
        if day >= release.start and (release.end is None or day <= release.end):
            return release
    return None


# ── country risk tiers ───────────────────────────────────────────────────────
# The Country and Territory Protection List classifies countries as lower,
# medium, or higher risk, or not published; the tier picks the current era's
# rho and threshold.  Medium- and higher-risk countries are released only
# from 2024-02-15.  The list itself is versioned -- Wikimedia's canonical
# countries.tsv carries it, and its git history dates each revision -- and
# every boundary below was confirmed against the data (a tier change moves a
# country's minimum released count between 90, 550, and 1000, or removes the
# country outright).  Only departures from "lower" are recorded here.

TIERS_START = dt.date(2024, 2, 15)

_TIER_VERSIONS = (
    # CTPL of 2024-01-25, in force when tiered release began
    (TIERS_START, {
        "medium": frozenset({
            "AE", "AF", "AZ", "BD", "DJ", "ET", "HN", "IQ", "KW", "KZ",
            "LA", "NI", "OM", "PK", "PS", "SD", "TJ", "UZ", "VE", "YE"}),
        "higher": frozenset({
            "BH", "BY", "EG", "ER", "RU", "SA", "TM", "TR"}),
        "not published": frozenset({
            "CN", "CU", "IR", "KP", "MM", "MO", "SY", "VN"}),
    }),
    # 2024-05-19: Hong Kong reclassified from lower risk to not published
    (dt.date(2024, 5, 19), {
        "medium": frozenset({
            "AE", "AF", "AZ", "BD", "DJ", "ET", "HN", "IQ", "KW", "KZ",
            "LA", "NI", "OM", "PK", "PS", "SD", "TJ", "UZ", "VE", "YE"}),
        "higher": frozenset({
            "BH", "BY", "EG", "ER", "RU", "SA", "TM", "TR"}),
        "not published": frozenset({
            "CN", "CU", "HK", "IR", "KP", "MM", "MO", "SY", "VN"}),
    }),
    # the 2026 recalibration, first reflected in the file of 2026-01-26
    (dt.date(2026, 1, 26), {
        "medium": frozenset({
            "BD", "BT", "ET", "HN", "IQ", "KH", "KZ", "LA", "RW"}),
        "higher": frozenset({
            "AE", "AZ", "BH", "DJ", "EG", "NI", "PK", "PS", "SD", "TJ",
            "TR", "UZ", "YE"}),
        "not published": frozenset({
            "AF", "BY", "CN", "CU", "ER", "HK", "IR", "KP", "MM", "MO",
            "RU", "SA", "SY", "TM", "VE", "VN"}),
    }),
)

# Classified "higher risk" on every list version, yet absent from every
# released file (checked across 2024-2026); the pipeline evidently excludes
# it outright.
NEVER_PUBLISHED = frozenset({"TR"})


def tier(day, country_code):
    """A country's risk tier on a given day: "lower", "medium", "higher",
    or "not published".

    Before tiered release began, every published country was lower risk --
    including some the 2024 list later demoted (Macao appears through
    2024-02-14 at the lower-risk threshold, then vanishes).  Which countries
    the pre-2024 protection list excluded is not reconstructed here: they
    are reported lower like everything else, and their rows simply do not
    occur in the files.
    """
    code = str(country_code).upper()
    if code in NEVER_PUBLISHED:
        return "not published"
    if day < TIERS_START:
        return "lower"
    version = max((v for v in _TIER_VERSIONS if v[0] <= day),
                  key=lambda v: v[0])[1]
    for name, members in version.items():
        if code in members:
            return name
    return "lower"


# ── the routing ──────────────────────────────────────────────────────────────

def mechanism(day, country_code):
    """The mechanism behind a released row for this day and country.

    Raises for days before any release exists and for countries not
    published on the day; a row claiming to be either is not a row this
    codebook can explain.
    """
    release = release_for(day)
    if release is None:
        raise ValueError(f"no pageview release covers {day.isoformat()}")
    which = tier(day, country_code)
    if which == "not published":
        raise ValueError(
            f"{country_code!r} is not published on {day.isoformat()}")
    if release.name == "pre-2017":
        return _LAPLACE_300
    if release.name == "historical":
        return _LAPLACE_30
    if day in PATCH_DAYS:
        return _LAPLACE_30
    if (str(country_code).upper() == "US" and US_GAP[0] <= day <= US_GAP[1]
            and day not in US_GAUSSIAN_DAYS):
        return _LAPLACE_30
    return _GAUSSIAN[which]


# ── access ───────────────────────────────────────────────────────────────────

def resolve_days(days):
    """Days given as dates, datetimes, ISO strings, or an iterable of them."""
    if isinstance(days, (str, dt.date, dt.datetime)):
        days = [days]
    out = []
    for day in days:
        if isinstance(day, str):
            day = dt.date.fromisoformat(day.strip())
        elif isinstance(day, dt.datetime):
            day = day.date()
        elif not isinstance(day, dt.date):
            raise TypeError(f"cannot read {day!r} as a day")
        if day not in out:
            out.append(day)
    return out


def days_between(start, end):
    """Every released day from `start` through `end`, inclusive.

    The five days the historical pipeline never released are omitted, so the
    result can be handed straight to `fetch_pageviews`; ask for one of them
    explicitly to be told why it cannot be served.
    """
    (start,), (end,) = resolve_days(start), resolve_days(end)
    out = []
    day = start
    while day <= end:
        if day not in MISSING_DAYS and release_for(day) is not None:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def _require_release(day):
    release = release_for(day)
    if release is None:
        raise ValueError(f"no pageview release covers {day.isoformat()}")
    if day in MISSING_DAYS:
        raise ValueError(
            f"{day.isoformat()} was never released: the pipeline failed and "
            "the retention-limited source data was gone before it could be "
            "re-run.  `days_between` skips these days.")
    return release


def _local_path(root, release, day):
    return os.path.join(root, release.dataset, f"{day.isoformat()}.tsv")


def fetch_pageviews(days, *, root=DEFAULT_ROOT, overwrite=False):
    """Mirror the daily TSVs for `days` into `root`, for `get_pageviews`.

    The releases are public but sizeable -- roughly 5-60 MB per day, with no
    finer partitioning to prune -- so nothing is bundled and nothing is
    fetched implicitly: name the days you want and they are mirrored,
    preserving the releases' own directory layout.

    Returns the list of local paths that now hold the requested days.
    """
    paths = []
    for day in resolve_days(days):
        release = _require_release(day)
        dest = _local_path(root, release, day)
        if os.path.exists(dest) and not overwrite:
            paths.append(dest)
            continue
        paths.append(_download(release.url(day), dest, day))
    return paths


def _download(url, dest, day):
    request = urllib.request.Request(
        url, headers={"User-Agent":
                      "noisyvalue (https://github.com/sbaldasty/noisyvalue)"})
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    partial = dest + ".part"
    try:
        with urllib.request.urlopen(request, timeout=60) as src, \
                open(partial, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
        os.replace(partial, dest)
    except urllib.error.HTTPError as error:
        if os.path.exists(partial):
            os.remove(partial)
        if error.code == 404:
            raise FileNotFoundError(
                f"{url} does not exist; {day.isoformat()} has not been "
                "published (the current day usually appears with a lag of a "
                "day or two)") from error
        raise
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return dest


def _aslist(value):
    if isinstance(value, (str, int, np.integer)):
        return [value]
    return list(value)


def _filters(country, project, page, page_id, item):
    conditions = []
    if country is not None:
        wanted = [str(c).strip().lower() for c in _aslist(country)]
        conditions.append(
            pl.col("country").str.to_lowercase().is_in(wanted)
            | pl.col("country_code").str.to_lowercase().is_in(wanted))
    if project is not None:
        conditions.append(pl.col("project").is_in(
            [str(p).strip().lower() for p in _aslist(project)]))
    if page is not None:
        conditions.append(pl.col("page_title").is_in(
            [str(p).strip().replace(" ", "_") for p in _aslist(page)]))
    if page_id is not None:
        conditions.append(pl.col("page_id").is_in(
            [int(p) for p in _aslist(page_id)]))
    if item is not None:
        conditions.append(pl.col("item_id").is_in(
            [str(q).strip().upper() for q in _aslist(item)]))
    return conditions


def _symbol_name(day, country_code, project, page_id):
    """A cell's noise symbol: stable across calls, unique across cells."""
    key = f"{day.isoformat()}|{country_code}|{project}|{page_id}"
    return "wmpv_" + hashlib.md5(key.encode()).hexdigest()[:16]


def get_pageviews(days, *, country=None, project=None, page=None,
                  page_id=None, item=None, root=DEFAULT_ROOT):
    """Noisy pageview counts for the selected days, as a tidy DataFrame.

    `country` matches the country name or its ISO code, case-insensitively;
    `project` a project domain like "en.wikipedia"; `page` the page title
    (spaces and underscores are interchangeable); `page_id` and `item` the
    numeric page id and the Wikidata QID.  Each filter takes a scalar or a
    list, and they combine conjunctively.  A filter matching nothing --
    including a country the release excludes -- simply selects no rows.

    Reads only what `fetch_pageviews` has mirrored into `root`.  One row per
    released cell: the `value` column holds the flat-prior posterior of the
    true (deduplicated, clipped) count, nonnegative and centred at the
    released figure, built by the mechanism the `mechanism` column names;
    `variance` is that mechanism's noise variance.  Rows whose noisy count
    fell below the release threshold are absent from the files, so they are
    absent here too -- an absence is a censored observation, not a zero.
    """
    conditions = _filters(country, project, page, page_id, item)
    tables = []
    for day in resolve_days(days):
        release = _require_release(day)
        path = _local_path(root, release, day)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} is not mirrored; "
                f"fetch_pageviews({day.isoformat()!r}) downloads it")
        if item is not None and "item_id" not in release.schema:
            continue                    # pre-2017 rows carry no Wikidata ids
        lf = pl.scan_csv(path, separator="\t", has_header=False,
                         quote_char=None, schema=release.schema)
        for condition in conditions:
            lf = lf.filter(condition)
        tables.append((day, lf.collect()))

    columns = {name: [] for name in
               ("date", *COLUMNS[:-1], "mechanism", "variance")}
    obs, roots = [], []
    for day, table in tables:
        mechs = {}
        for row in table.iter_rows(named=True):
            code = row["country_code"]
            mech = mechs.get(code)
            if mech is None:
                mech = mechs[code] = mechanism(day, code)
            page_key = row["page_id"]
            if page_key is None:            # pre-2017: the title is the identity
                page_key = row["page_title"]
            value = mech.cell(
                row["count"],
                symbol=_symbol_name(day, code, row["project"], page_key))
            columns["date"].append(day)
            for name in COLUMNS[:-1]:
                columns[name].append(row.get(name))
            columns["mechanism"].append(mech.source)
            columns["variance"].append(mech.variance)
            obs.append(value._obs)
            roots.append(value._root)

    return pd.DataFrame({
        "date": pd.to_datetime(columns["date"]),
        "country": pd.array(columns["country"], dtype="string"),
        "country_code": pd.array(columns["country_code"], dtype="string"),
        "project": pd.array(columns["project"], dtype="string"),
        "page_id": pd.array(columns["page_id"], dtype="Int64"),
        "page_title": pd.array(columns["page_title"], dtype="string"),
        "item_id": pd.array(columns["item_id"], dtype="string"),
        "mechanism": pd.array(columns["mechanism"], dtype="string"),
        "value": NoisyIntArray(np.asarray(obs, dtype="int64"),
                               np.asarray(roots, dtype=object)),
        "variance": np.asarray(columns["variance"], dtype="float64"),
    })
