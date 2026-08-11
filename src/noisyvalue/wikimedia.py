"""The Wikimedia differentially-private pageview releases: the codebook.

Since 2015 Wikimedia has published daily counts of pageviews per
(country, project, page), protected by differential privacy and served as
plain daily TSVs from analytics.wikimedia.org.  This module knows *what the
numbers mean*: which noise mechanism produced each row, at what scale, and
under which suppression threshold.  Fetching and reading the files sits on
top of it.

What a count measures: each device contributes at most its first 10 unique
pageviews per day (the two historical eras, which predate the client-side
counting cookie, instead assumed per-user daily bounds of 30 and 300).  The
latent quantity is therefore a deduplicated, clipped pageview count --
systematically below the exact global counts of the public pageview API.

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

from .dataset import DiscreteGaussianFamily, DiscreteLaplaceFamily

BASE_URL = "https://analytics.wikimedia.org/published/datasets"
DEFAULT_ROOT = "data/wikimedia-pageviews"

# TSV column order; the files carry no header row.
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

    def cell(self, obs):
        """The flat-prior posterior for a released count: nonnegative,
        centred at the observation."""
        return self.family.cell(int(obs), self.variance, lo=0)


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

class Release:
    """One release directory: a name, a URL path, and the days it serves."""

    def __init__(self, name, dataset, start, end):
        self.name = name
        self.dataset = dataset
        self.start = start
        self.end = end                      # None while still being extended

    def __repr__(self):
        return f"<Release {self.name!r}: {self.start} - {self.end or 'present'}>"

    def url(self, day):
        return f"{BASE_URL}/{self.dataset}/{day.isoformat()}.tsv"


RELEASES = (
    Release("pre-2017", "country_project_page_historical_pre_2017",
            dt.date(2015, 7, 1), dt.date(2017, 2, 8)),
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
# data an exact re-run would have needed): seven whole days, plus United
# States rows for the first months of the era, which a database naming error
# had silently dropped.
PATCH_DAYS = frozenset({
    dt.date(2023, 6, 19), dt.date(2023, 10, 25),
    dt.date(2023, 11, 13), dt.date(2023, 11, 17), dt.date(2023, 11, 19),
    dt.date(2023, 11, 23), dt.date(2023, 11, 27),
})
US_GAP_END = dt.date(2023, 9, 19)           # inclusive; Gaussian from the 20th


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
    if str(country_code).upper() == "US" and day <= US_GAP_END:
        return _LAPLACE_30
    return _GAUSSIAN[which]
