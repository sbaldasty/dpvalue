# Wikimedia pageview releases: departures from the documentation

**Status:** every claim here was established empirically against the released
files during the construction of `src/noisyvalue/wikimedia.py` (August 2026),
and the ones that affect posteriors are re-checked by the integration tests in
`test/test_wikimedia.py`. The releases' own documentation is the trio of
READMEs under `analytics.wikimedia.org/published/datasets/`, the [Tumult/WMF
paper (arXiv:2308.16298)](https://arxiv.org/abs/2308.16298), and the [Country
and Territory Protection
List](https://foundation.wikimedia.org/wiki/Legal:Country_and_Territory_Protection_List).

The workhorse validation method: within a large country's rows on one day, the
minimum released count hugs the suppression threshold that censored its
neighbours, so observing that minimum (90 / 450 / 550 / 1000 / 3500) reveals
which mechanism produced the rows. All of the routing below was pinned this
way.

| # | Documentation says | The files say |
| --- | --- | --- |
| 1 | files are `<year>-<month>-<day>.csv` (e.g. `2023-2-6.csv`) | tab-separated `.tsv`, zero-padded ISO dates |
| 2 | six data features (country, project, page_id, page_title, item_id, count) | seven columns — country *name* and ISO code are separate |
| 3 | (current-era schema implied throughout) | pre-2017 files have six columns: no `item_id`, and `page_id` is always empty |
| 4 | rows released only when the noisy count is **>** 90 / 450 / 3500 | observed minima are exactly 90 / 450 / 550 / 1000 / 3500 — the operative rule is **≥** |
| 5 | US data "was not published from 6 Feb 2023 to 19 Sep 2023" | intermittent: 48 of those 226 days carry ordinary Gaussian US rows; only 178 carry the geometric backfill |
| 6 | seven missing current-era days are enumerated (Update #2) | five *historical* days are also missing and nowhere mentioned: 2017-06-26, 2017-07-24, 2020-07-19, 2020-07-20, 2022-10-19 |
| 7 | Türkiye is "Higher risk" on every protection-list revision, which per the release rules means publication at ρ=1.546e-4 with threshold 1000 | no Turkish row exists in any file, in any era, through 2026 |
| 8 | counts are "pageviews" | historical-era counts include automated traffic: per-page country sums track the public API's *all-agents* totals, not its user totals |
| 9 | the Gaussian noise scale is not stated (only ρ per tier) | σ² = 10/(2ρ) — derived, then confirmed by reproducing all three README 95 % intervals |
| 10 | Thailand is "Lower risk" on every known list revision | absent from every pre-2024 file, appearing for the first time on exactly 2024-02-15 — the old protection list evidently excluded it |

Details and consequences follow, roughly in order of how much each matters for
posterior modeling.

## The US gap is intermittent (5)

The current-era README's Update #2 describes a solid gap. Scanning the US
minimum released count for every day in the documented range gives a cleanly
bimodal answer — 450 on 178 days (the geometric, scale-30 backfill produced
under the historical pipeline's rules) and 90–99 on 48 days (ordinary
lower-risk Gaussian rows, so the pipeline evidently succeeded those days). The
two never blur, and no day in the range lacks US rows entirely. There is no
published list of which days are which; the 48 exceptions are vendored as
`US_GAUSSIAN_DAYS` in `wikimedia.py`. Getting this wrong mislabels a US row's
noise standard deviation by a factor of ~2.3 (18.2 vs 42.4).

The seven whole-day patch files (19 Jun, 25 Oct, 13/17/19/23/27 Nov 2023) were
each confirmed to be wholly geometric: their overall minimum is 450 for every
country.

## Historical counts include automated traffic (8)

"Earth" on en.wikipedia, 2018-06-01: the DP country counts sum to 46,927. The
public pageview API reports 7,691 *user* views and 49,262 *all-agents* views
for that page-day. The DP sum tracks the all-agents figure, and the largest
single contributors are the Netherlands (8,433) and Singapore (8,374) —
data-center traffic, not readers. The current era, which counts via a
client-side browser cookie, sits *below* the API's user-only figure on the
day checked (9,691 vs 12,491), as deduplication and clipping predict.

Consequence: a historical-era count estimates "bounded pageview events from
anything with an IP address", and country-level comparisons there can be
dominated by bot placement. Any attempt at human-readership inference from the
historical era needs the public per-agent series as side information (they are
exact and free).

## Türkiye is excluded despite its listing (7)

Every revision of the protection list classifies Türkiye "Higher risk", the
tier that has been published (at ρ=1.546e-4, threshold 1000) since 15 Feb
2024. Pakistan, listed identically, appears with minima at its threshold;
Türkiye — a country of 85 million — has zero rows in any era through 2026. The
pipeline evidently excludes it outright for reasons not stated anywhere we
could find. `wikimedia.py` encodes this as `NEVER_PUBLISHED = {"TR"}`.

## Protection-list history is only in a git repo, and effective dates are not stated (adjunct to 7, 9)

The tier that determines a current-era row's σ and threshold is the tier *at
release time*, so the list's revision history matters. It lives in the git
history of Wikimedia's canonical
[`countries.tsv`](https://gitlab.wikimedia.org/repos/movement-insights/canonical-data/-/blob/main/country/countries.tsv);
the effective date of each revision in the data was pinned by probing files on
either side (tiering begins exactly 2024-02-15; Hong Kong is dropped
2024-05-19; the 2026 recalibration — 22 countries reshuffled — takes effect
2026-01-26, the TSV commit date, not the foundation wiki's "last updated 30
March 2026" stamp). A note: the *pre-2024* list differed materially in both
directions — Macao, Hong Kong, Palestine, and Tajikistan were published at
the ordinary threshold through 2024-02-14 even though the 2024 list bans or
restricts them, while Thailand (lower-risk on every list we can see) has no
row in any pre-2024 file and first appears on exactly 2024-02-15 — so today's
list must not be applied retroactively in either direction.
`_PRE_TIERS_EXCLUDED` in `wikimedia.py` vendors the empirically-absent set
(35 countries, deliberately conservative); it exists so that censored-cell
posteriors (`missing="censored"`), whose meaning is "the noisy count fell
below the threshold", are refused where absence actually meant policy.

## Small mechanical departures (1–4, 6, 9)

- The README's dataset-structure section says `.csv` with non-padded dates;
  the files are `.tsv` with ISO names. Its feature list omits that country
  name and ISO code are separate columns.
- The pre-2017 release predates page ids: its files drop the `item_id` column
  and never populate `page_id`, so a page's identity there is its title.
- The suppression rule as stated ("only rows with >90 pageviews") is off by
  one against the files, where minima land exactly on 90/450/550/1000/3500.
  Immaterial for posteriors; confusing when validating.
- Five days inside the historical range were never released at all (pipeline
  losses, unmentioned in any README): `MISSING_DAYS` in `wikimedia.py`.
- The Gaussian noise variance had to be derived: zCDP with L2 sensitivity
  √10 (one device touches ≤10 cells by 1 each) gives σ² = 10/(2ρ), i.e.
  σ ≈ 18.23 / 90.05 / 179.84 by tier. 1.96σ reproduces the README's own 95 %
  intervals (35.7 / 176.5 / 352.5) to the decimal, which is the confirmation.

## Maintenance

`test_live_codebook_matches_the_current_protection_list` diffs the vendored
tier sets against the live canonical `countries.tsv` and fails on the next
recalibration; the fix is a new dated entry in `_TIER_VERSIONS`, with its
effective day pinned by probing released files around the revision, as above.
`wikimedia.CODEBOOK_AS_OF` records how far the constants have been validated,
and the loaders warn when asked about later days.
