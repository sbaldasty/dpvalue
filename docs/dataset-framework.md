# A general dataset framework

**Status:** implemented in `src/noisyvalue/dataset/` (framework) and
`src/noisyvalue/nmf.py` (the three 2020 Census products built on it), tested
in `test/test_dataset.py` and `test/test_nmf.py`. `census.py` is untouched and
still the shipping interface; `nmf.py` is a replacement candidate, verified
cell-for-cell against it. See "What was built" at the end for what the
implementation does and does not cover.

## Motivation

`census.py` is the only dataset interface in the library, and it is ~1500
lines. Most of what it does is not specific to the Census. If a second dataset
arrives — another statistical agency's release, a synthetic-data product, a
survey with published design effects — we would either fork most of that file
or grow a second one just as large.

Separately, `census.py` has a correctness gap that a per-dataset design cannot
fix on its own: nothing it returns is jointly consistent with anything else it
returns. That gap and the layering problem have the same root cause, so this
document treats them together.

## Where the seam already is

`census.py` stacks four layers with no interfaces between them:

1. **Codebook** (`census.py:97-505`) — declarative data: axis level names,
   binning-to-base-cell index groups (`_DHC_AXIS_GROUPS`), query names,
   geography levels. About 400 lines of tables.
2. **Physical access** (`census.py:970-999`, `fetch_dhc`) — parquet layout,
   Hive partition pruning, S3 mirroring, schema drift handling.
3. **Constraint interpretation** (`census.py:1155-1323`) — reading NMF
   constraint rows and deciding what they mean: structural zeros, category
   bounds, exact block totals.
4. **Posterior construction** (`census.py:1094-1150`, `_emit_cells`) — turning
   "this cell is pinned / bounded / free" into `Node` graphs.

Layer 2, and the *reading* half of layer 3, are genuinely Census-specific.
Everything else is not. `_conditioned_pair` (`census.py:1126`) is generic math:
two independent Gaussian measurements plus a known exact sum. `_merged_blocks`
(`census.py:1298`) is generic given a coarsening map. `_noisy_count` is generic
given a noise family.

The boundary test used throughout this document: **if the Census released
identical math in a different file format, would this code change?** If no, it
belongs in core.

## The dependency gap

Node identity comes from `Name.fresh()` (`graph.py:11-16`), a global counter,
with no memoization anywhere. Consequently:

- Two `get_dhc` calls for the same cell produce two independent posteriors for
  one physical unknown. Sampled jointly, they disagree.
- Even within a single call, two queries are disjoint node sets. `sex*hispanic`
  and `detailed` measure the same latent histogram, but today
  `sum(detailed cells) != sex*hispanic cell` in every draw, and the two are
  uncorrelated. That is not only a lost correlation; it discards information.
- The only sharing that exists is inside `_conditioned_pair`, within a single
  query row.

The structure the current code does not name: there is **one latent histogram
`theta`**, and every released number is a linear functional of it,
`y_k = A_k theta + eps_k`, subject to exact linear constraints
`C theta = b` and bounds. `census.py` goes measurement to answer per cell,
which is why nothing is shared.

A *view* is the object that says: here is `theta`, and queries are projections
of it.

## Core responsibilities

### A. Catalog

Metadata with no data attached — this is what answers "what variables exist,
what values can they take."

The central type is an `Axis` holding named `Binning`s over the axis's base
levels, plus the refinement relation between binnings (`age` refines
`age_10_groups` refines `age_18_64_116`). Today that lives in two
hand-maintained parallel dicts, `_DHC_AXIS_GROUPS` and `_DHC_AXIS_LEVELS`,
which must stay in sync by convention. Core owns the type and validates that
groups are disjoint and label count matches group count.

A binning is **not** required to cover the base set. Real ones do not:
`_HHGQ_GROUPS["gqNursingTotal"]` is `((3,),)` and
`_PL94_AXIS_LEVELS["votingage"]["nonvoting"]` is `("under_18",)`. These are
sub-selections, and the distinction is load-bearing — a constraint stated over
a partial cover does not pin a complete block total, so the block solver in
section C has to know which it is.

Geography is the same kind of object as an axis binning, but it is *not* a
coarsening chain, and the framework should not model it as one. See the next
section.

### A2. Partitions are a lattice, not a hierarchy

Five distinct layers of partitioning appear in `census.py`:

1. **Storage partitions** — Hive `State=` / `County=` directories. An I/O
   concern only.
2. **Geographic spine** — us, state, county, tract, block group, block, plus
   the AIAN / non-AIAN branch split of each state.
3. **Axis binnings** — `_DHC_AXIS_GROUPS`, index groups over base levels.
4. **Constraint blocks** — what `_merged_blocks` and `_pl94_blocks` derive by
   reconciling a query's binning against `pl94_con`'s binning.
5. **Sampling-independence blocks** — derived from 4, consumed by the
   construction ladder in `_dhc_pl94_blocks`.

Layers 3 and 4 are the same operation. `_merged_blocks` (`census.py:1298`)
takes a query binning and the PL94 category map and merges query positions
until each block covers whole PL94 categories: that is the **join** (finest
common coarsening) of two partitions in the refinement lattice.
`_pl94_blocks` takes the product of those joins across axes.

Geography does not fit a chain model, because the spine contains genuinely
incomparable partitions:

- **Spine block groups and tabulation block groups do not align.** `_geoid`
  returns `None` for `Block_Group` (`census.py:1048`), and
  `_aggregate_real_block_groups` recovers real ones only by descending to
  blocks. Two partitions at the same level, neither refining the other.
- **The AIAN branch cross-cuts the hierarchy.** It is not a level but a second
  partition of each state, which is why rows sharing a `geoid` must be summed
  to get whole-geography estimates.
- **Spine tracts correspond to tabulation tracts only partially.**
  `_tract_geoids` warns where a spine tract spans multiple tabulation tracts
  and leaves those `<NA>`.

So the core abstraction is **partitions over a set of atoms, with join, meet,
and an availability predicate**:

- **Join** reconciles a query stated in one partition against evidence stated
  in another. This is layer 4, and `_merged_blocks` already implements it
  generically.
- **Meet** is needed when two partitions are incomparable. `census.py` always
  answers it the same way: descend to the atoms and re-aggregate. That is what
  `real_block_groups=True` does, and it is also how `_tract_geoids` derives its
  crosswalk in the first place, by scanning block geocodes.
- **Availability** is the predicate the current code lacks. The meet always
  exists mathematically, but it is only usable if measurements exist at the
  atoms. PL94 has block-level data, so it does. DHC stops at county
  (`_DHC_GEO_LEVELS`), so no re-aggregation recovers a sub-county geography.
  Core must represent "computable in principle, unavailable in this product"
  and report it, rather than silently substituting something coarser.

Layer 1 falls out of the same machinery: pruning by `State=` is valid exactly
when the requested support is a union of storage-partition blocks — a
refinement test applied to I/O instead of semantics.

Modelling geography this way still collapses `pl94_queries()` /
`dhc_queries()` into one core function over a declared measurement catalog,
and still unifies `_aggregate_real_block_groups` (`census.py:1057`) with any
other axis rollup. It just does so through the lattice operations rather than
by pretending the spine is a chain.

### B. View

A view is `(universe, geography set, evidence, memo table)`. Its jobs:

- **Cell identity.** A stable key `(universe, geo, cell index)` maps to a node,
  memoized. Identical keys return the *same node object*. This alone closes the
  two-calls gap.
- **Query resolution.** A query is a projection; aggregate cells become
  `DerivedNode`s over base cells, also memoized. Returned frames hold roots
  pointing into the shared graph, so `sample_noisy_values` already does the
  right thing — the sampler needs no changes.
- **Laziness.** Never build all 1.2M cells; build what a query touches.

The contract, to be documented prominently: **two queries against the same view
are jointly consistent; two queries against different views are independent.**
This is honest about both the guarantee and its cost, since a view is stateful
and holds memory.

### C. Evidence ledger and block solver

Core accepts evidence in a dataset-agnostic vocabulary:

- these cells are pinned to zero
- these cells sum exactly to `b`
- this group of cells is bounded in `[lo, hi]`
- this cell was measured as `y` with variance `v` under noise family `F`

From that it partitions latent cells into independently sampleable blocks and
picks a construction per block. The ladder in `_dhc_pl94_blocks`
(`census.py:1233`) is already the right one; it is merely written for a single
dataset:

| free cells in block | construction |
| --- | --- |
| 0 | nothing to do |
| total is 0 | all pinned to zero |
| 1 | point mass at the block total |
| 2 | `_conditioned_pair` — closed form, stays anti-correlated |
| 3 or more | block total kept as an upper bound only |

Making the last row a **pluggable strategy** is the payoff: the exact
conditional there is a discrete Gaussian on a lattice hyperplane, and swapping
in a lattice sampler or MCMC later would improve every dataset at once. Today
the constraint is silently weakened to a bound; that should become a named,
reportable decision rather than an implementation detail.

The block partition itself comes from the join described in section A2, so the
solver is generic in both the query binning and the constraint binning. Two
things it must check that the current code gets right only implicitly:

- If either binning is a **partial cover** rather than a partition, the
  block total is not pinned and the block drops to the bound-only case.
- If a constraint's partition is **incomparable** with the query's and the
  atoms are unavailable, the constraint contributes nothing and should be
  reported as unused rather than silently dropped.

### D. Existing machinery

`core.py`, `graph.py`, `pandas.py`, and `io.py` need little change.
`io.py` already serializes node graphs, so views are serializable in principle
(see the caveat on node naming below).

## Per-dataset extension responsibilities

1. **Declare the schema** — axes, binnings (partitions or partial covers),
   labels, universes, and the geography partitions with their atom set. Data,
   not code. `census.py:97-505` becomes a spec that core validates.
2. **Physical access** — locate, scan, prune, fetch. Returns normalized
   measurement rows: `(geo key, universe, axis-spec tuple, values, variance)`.
   Core supplies the refinement test that decides when a storage partition can
   be pruned; the extension supplies the mapping from partition key to files.
3. **Translate native constraints into core's evidence vocabulary.**
   `_dhc_person_rules` and its helpers stay in the extension but shrink to
   interpretation: "`nurse_nva_0_con` equal to 0 means these cell indices are
   pinned to zero." The *consequence* becomes core's problem.
4. **Identity and labelling quirks** — spine geocode to GEOID, the AIAN branch,
   the tract crosswalk (`census.py:1002-1054`). Core needs only an opaque
   stable geography key plus display columns, and an **atom map**: given a
   partition block, which atoms it contains. That single hook is what makes the
   meet computable, and it is irreducibly dataset-specific — the reason the
   block-group crosswalk is derivable at all is that a block geocode embeds its
   tabulation GEOID in its last 16 characters, which core cannot know.
5. **Name the noise family** — "discrete Gaussian, scale from the `variance`
   column." Core supplies node types; another dataset might use Laplace, or a
   swap-based mechanism.

## Sketch of the API

```python
dhc = noisyvalue.datasets.dhc_2020            # catalog only, no I/O
dhc.axes                                      # ['relgq','sex','age','hispanic','cenrace']
dhc.axis('age').binnings                      # ['detailed','age_10_groups', ...]
dhc.axis('age').binning('age_10_groups').levels
dhc.measurements(geography='county')          # replaces dhc_queries()

view = dhc.view(geography='county', state='VT', county=['007', '009'])
a = view.query('sex*hispanic')
b = view.query('sex*age_38_groups')
# a and b share latent cells; sampling them together is consistent
```

`get_dhc(...)` survives as a thin wrapper: one call is one throwaway view.

## Open problems

**Memory is the binding constraint.** A view is a memo table that persists.
`detailed` nationally is 1,227,744 cells per geography, and `census.py:572`
already warns at 2M. One Python `Node` object per cell will not scale. The
change that matters most is a **vectorized node type** — a block of iid
truncated discrete Gaussians represented as one node backed by arrays, rather
than N objects. `NoisyIntArray` already stores roots in an object array, so the
storage side is half-built.

**Do not promise joint consistency across overlapping measurements in v1.**
Reconciling `sex*hispanic` against `detailed` properly is the DAS
post-processing problem: a large constrained optimization. The recommended
first step is to represent the evidence faithfully, resolve each cell using the
declared exact constraints plus one chosen measurement, and make the discarded
redundant evidence *visible* (a `view.unused_evidence()` report) instead of
silently ignoring it. Improving the resolver is then additive.

**Descending to atoms is correct but expensive.** Resolving incomparable
partitions via the meet means reading every atom in the requested area —
`real_block_groups=True` already carries that warning
(`census.py:636-641`). Core should treat it as an explicit, costed option
rather than something it does automatically to satisfy a query, and should
report when a query *could* be answered that way but was not.

**Node naming must become keyed rather than counter-based**, or views cannot be
serialized and reloaded reproducibly. The cheapest path keeps `Name.fresh()`
and has the view memo map keys to nodes, serializing the key map alongside the
graph.

**`consolidate()` will do less work.** Its eligibility rule is that a symbol
appears exactly once across the joint expression (`consolidate.py:128`). Views
increase sharing, so fewer symbols qualify. That is correct behaviour, but the
performance characteristics shift.

## Deferred

`analysis.py`'s `with_stratified_sampling_uncertainty` is the pattern for
stacking another uncertainty layer on top of dataset values. If views compose —
a measurement view feeding a sampling-uncertainty view — that generalizes
cleanly. Worth building only once the base view abstraction has earned its
keep.

## What was built

### Modules

| module | responsibility |
| --- | --- |
| `dataset/partition.py` | `Partition`, `join`, `meet`, refinement, cover vs. sub-selection |
| `dataset/catalog.py` | `Axis`, `Binning`, `Universe`, `Measurement`, `GeoLevel`, `Product`, `Source` |
| `dataset/evidence.py` | `ZeroRegion`, `ExactHistogram`, `BoundedHistogram`, `Region` |
| `dataset/solve.py` | `NoiseFamily`, the block ladder, `BlockStrategy`, node constructions |
| `dataset/view.py` | `View`: the memo table, query resolution, rollups, `unused_evidence()` |
| `nmf.py` | the Census schema, `NmfSource`, the constraint translation, `fetch_dhc` |

### Where the sketch changed under contact

- **The evidence vocabulary collapsed to three types.** "These cells sum
  exactly to `b`" generalized to `ExactHistogram`: a coarsening of the
  universe whose *every* cell total is known. That is what `pl94_con`
  actually is, and stating it that way made three separate code paths in
  `census.py` into one. The national person total, the housing-unit total,
  and `pl94_con` are now the same primitive at different coarsenesses, and
  `_emit_unit_h1`'s bespoke conditioned pair falls out of the ladder with no
  special case.
- **`exact_total_levels` disappeared.** It encoded "trust `total_con` only
  nationally"; the release simply does not carry the row anywhere else, so
  the evidence is absent rather than distrusted. Correspondingly, a
  measurement whose released query is missing at a level (the person total
  nationally) and one that was never released at all (the unit total) take
  the same path: resolve from evidence, and fail loudly if a cell is left
  free.
- **Bounds needed a "says nothing" state.** Summing the lower bounds of the
  coarse blocks a cell wholly contains gives 0 when it contains none — which
  is not the same fact as "at least zero", and conflating them silently
  reimposed nonnegativity on a caller who had declined it.
- **Generic evidence application needs an inertness rule.** Applying bounds
  by the lattice rather than by hand means a query that marginalizes an axis
  away picks up the sum of every category bound on it: real, and useless.
  `DiscreteGaussianFamily` drops a bound lying outside the sampler's own
  support window, where it is provably inert rather than merely improbable.
- **Geography stayed a lattice but did not need `meet` as a general
  operation.** The only incomparable case is tabulation block groups, and
  resolving it is `GeoLevel(derive_from=..., key=...)`: descend to the atoms
  the source measures and re-aggregate. `Product.plan` reports the three
  outcomes — direct, aggregated, unavailable — and `geography_table()`
  tabulates them.

### Verified against `census.py`

Every cell of every measurement, for PL94 person (36,442 cells across
Vermont's counties), PL94 units, and DHC person (252,083 cells for one
county), under `nonnegative` and `apply_constraints` both on and off: the
observation, the expression shape, and the exact posterior pmf all agree.
One cell of 288,525 differs, by a total-variation distance of 2e-56, from an
inert bound sitting exactly on the sampler's grid boundary.

Two deliberate behavioural differences:

- `apply_constraints=False` means the constraint rows are not read, so the
  housing-unit total (which has no released query) returns no rows and warns,
  where `census.py` served it anyway.
- Bounds are applied wherever the lattice says they bear, not only where a
  hand-written rule looked. This is strictly more evidence used; the inertness
  rule above keeps it from being strictly more expensive.

### Not built

- **The vectorized node type.** Still one `Node` per cell, so the memory
  ceiling in "Open problems" stands unchanged. `_CellKeys` computes keys for a
  whole measurement as an array specifically so a block of cells can become
  one array-backed node later without touching `View`.
- **Reconciliation of overlapping measurements.** As recommended: each cell is
  resolved from the exact evidence plus one chosen measurement, and
  `View.unused_evidence()` reports the rest. Worth noting that no two
  measurements in the three Census products name identical cells, so within
  these datasets the policy never actually has to choose; it is tested
  directly in `test_dataset.py`.
- **A real construction for blocks of three or more.** `BoundOnly` is still
  the default and still the honest answer; what changed is that it is now a
  named `BlockStrategy` a lattice sampler can replace, and the weakening is
  reported per block rather than passed over.
- **Keyed node naming.** `Name.fresh()` is untouched, so a view is not yet
  reproducibly serializable. Section D's claim that "`io.py` already
  serializes node graphs, so views are serializable in principle" is in any
  case wrong today for a nearer reason: `io.py` has no serializer for
  `TruncatedDiscreteGaussianNode`, so neither `census.py`'s output nor
  `nmf.py`'s can be saved at all. That gap predates this work and was left
  alone.

### One fix outside the framework

`core.py`'s `bin_op` lifted its right operand with the left operand's own
class, so once a running total had promoted to `NoisyFloat`, every further
`NoisyInt` was sympified down to its observation and its posterior discarded.
Any chain of three or more noisy additions silently lost everything past the
second term — which is exactly what geographic aggregation and `View.rollup`
do. Fixed by lifting with `accept=NoisyValue`; `census.py`'s own
`_aggregate_real_block_groups` was affected for any block group holding three
or more blocks.
