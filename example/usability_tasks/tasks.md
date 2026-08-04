# Task 1

The script `task1.R` prints the noisy released male and female counts for
Chittenden County, Vermont, from the actual 2020 DHC Noisy Measurement
File -- the real mechanism the Census Bureau used to satisfy differential
privacy for this release, not an assumed noise model.

Calculate 95% credible intervals for the true counts of male and female
residents.

# Task 2

The script `task2.R` prints the noisy released male and female counts for
Orange County, Vermont, from the same release.

Calculate 95% credible intervals for the true counts of male and female
residents.

# Task 3

Using the counts printed by `task1.R`, estimate the posterior probability
that Chittenden County's true male population exceeds its true female
population. If that probability is at least 0.95 or at most 0.05, state
which group has the larger population; otherwise, state that the data are
inconclusive.

(This replaces the frequentist hypothesis test of the original exercise:
`noisyvalue` tracks a full posterior over the true counts, including any
correlation the release mechanism induces between them, so "which is
bigger" has a direct probability rather than a p-value.)

A note on `>`: `male` and `female` need to be genuine scalar noisy values,
not noisy *columns*, for `male > female` to give you that posterior
probability. Comparing two noisy columns (or column slices) with `>`
instead gives an ordinary logical mask over their *observed* values --
correct for row-filtering a table, but not what task 3/4 are asking for.
`task1.R`/`task2.R` extract with `[[` for exactly this reason.

# Task 4

Using the counts printed by `task2.R`, estimate the posterior probability
that Orange County's true male population exceeds its true female
population. If that probability is at least 0.95 or at most 0.05, state
which group has the larger population; otherwise, state that the data are
inconclusive.

# Task 5

The script `task5.R` prints the noisy total number of male and female
residents across Chittenden and Orange counties, Vermont.

Calculate 95% credible intervals for the true totals of male and female
residents.
