# Toward posteriors for actual pageviews

**Status:** not implemented — design notes for revisiting. The question:
starting from the Wikimedia DP release (`src/noisyvalue/wikimedia.py`), can we
model posteriors for the *actual* per-country pageview counts, undoing the
deduplication, the per-device clipping, and the country suppression? Short
answer: not without priors, but with three explicitly-stated beliefs the
posteriors become useful, and the pieces map onto machinery the library
already has.

## What separates the released number from the actual one

Write `V_c` for the actual pageviews of one page from country `c` on one day
(the public pageview API's definition), and `B_c` for the quantity the DP
release measures. The released `gbc` is `B_c` plus calibrated noise, and
`B_c` differs from `V_c` in three ways:

1. **Deduplication and clipping.** In the current era `B_c` counts each
   device's first 10 *unique* pageviews per day, so repeat views and views
   beyond the cap vanish. In the historical eras there is no per-device
   counting at all; the pipeline summed pre-aggregated counts and the
   privacy analysis merely *assumed* per-user daily bounds (30, then 300).
2. **Agent population.** The current era's cookie implies a real browser, and
   its sums sit below the API's user-only figures. The historical eras did
   not filter automated traffic — their sums track the API's *all-agents*
   totals, and data-center countries can dominate a page
   (see `docs/wikimedia-errata.md`).
3. **Suppression and exclusion.** Countries below the release threshold are
   absent (now recoverable as censored posteriors via
   `get_pageviews(missing="censored")`), and excluded countries — China,
   Russia, …, Türkiye, plus the wider pre-2024 list — are entirely latent.

## The exact side information

An unusual amount of *exact* truth is public, and any model should condition
on it:

- The pageview API gives every page's **daily global count**, split by agent
  (user / spider / automated) and access method, exactly. So `Σ_c V_c` over
  *all* countries is known, per agent class, per day.
- The ingestion threshold (150 global pageviews) is checkable exactly from
  the same API, which settles whether an absent row was even eligible.
- The API's per-agent split bounds how much of a page-day is automated,
  which is most of what makes the historical era hard.

## The model sketch

Three beliefs, each doing one job:

1. **A dedup/clip ratio prior.** Model `B_c ~ Binomial(V_c, r_c)` — binomial
   thinning is not literally what deduplication does, but it is the right
   variance model for "each view independently survives counting with
   probability r". The page-level average ratio r̄ is *nearly observable*:
   `Σ released gbc ÷ exact global user count` estimates it directly, biased
   down only by the suppressed remainder (≈0.78 for "Earth" on 2023-05-01,
   tight for popular pages). The belief added is only that per-country
   deviation from r̄ is modest — e.g. `r_c ~ Beta` centred at r̄ with
   dispersion reflecting how much browsing behaviour (repeats per session,
   device sharing) plausibly varies by country.
2. **A country-composition prior.** The exact global total constrains
   `Σ_c V_c` including the never-published countries, which are large and
   entirely latent; without a prior on country shares the constraint mostly
   feeds them. A project's country mix is stable day to day, so the released
   countries themselves pin most of the composition; the excluded countries
   need genuine prior input (internet population × language affinity, or
   shares transported from comparable projects).
3. **Temporal smoothness.** The noise is independent across days but
   readership and ratios are not; a state-space prior on `V_c(t)` and
   `r_c(t)` borrows strength across days. This both tightens per-day
   posteriors below the single-day noise sd and sharpens the censored-cell
   intervals dramatically — Croatia's missing World-Cup days are far more
   likely near 400 than near 0 once the neighbouring released days inform
   them.

For the historical era there is a fourth, harder belief: how automated
traffic distributes over countries (concentrated in data-center locations).
Targeting the all-agents total instead sidesteps it at the cost of answering
a different question.

## How it maps onto existing machinery

- **The thinning layer** is expressible today: `BinomialNode` accepts
  symbolic parameters, so `gbc = Binomial(V_c, r_c) + eps` is a node graph
  the sampler can already walk (hierarchical noise resolves in topological
  order). `V_c` and `r_c` become latent nodes.
- **The exact-global-total constraint** is the dataset framework's evidence
  vocabulary verbatim — an `ExactHistogram` whose block has many free cells.
  Today `solve` weakens such blocks to upper bounds; the missing piece is an
  MCMC/lattice `BlockStrategy` (the same gap noted for the census DHC
  blocks), which would serve both datasets at once.
- **Censored cells** (`missing="censored"`) are the likelihood terms for
  suppressed countries: their pmf `∝ F_noise(τ-1-t)` is exactly the factor a
  joint model needs, already packaged as samplable nodes.
- **The flat-prior cells** `get_pageviews` returns are the likelihood layer
  such a model consumes; nothing about them changes.

## Open questions

- Binomial thinning understates dependence: one device's views are not
  independent survivals. Does the extra-binomial variation matter at
  release-noise scales (σ ≥ 18)? Probably not for popular pages; worth a
  simulation.
- Identifiability at the tail: for a country–page pair seen only through
  censoring, the posterior is prior-dominated. The model should report that
  honestly (e.g. prior-vs-posterior overlap) rather than hide it.
- The composition prior for excluded countries is doing real work and cannot
  be validated against the release. Sensitivity analysis is mandatory there.
- Whether to target user views, all-agents views, or "counted" views
  (`B_c`) should be a caller choice, since each is defensible for different
  questions.
