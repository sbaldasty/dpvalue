# Estimates the posterior probability that Chittenden County's true male
# population exceeds its true female population, using the actual 2020 DHC
# Noisy Measurement File -- not an assumed noise model, the real
# differential-privacy mechanism.
#
# Analog of uvm-plaid/census_utility/tasks_census/task3.R, which instead
# prints two booleans over the *released* tidycensus counts (are they
# equal? is male larger?) and leaves the actual hypothesis test as written
# analysis. This script computes a posterior probability directly instead:
# `noisyvalue` tracks a full posterior over the true counts, including any
# correlation the release mechanism induces between them, so "which is
# bigger" has a direct probability rather than a p-value. See tasks.md's
# Task 3 for the full explanation.
#
# `[[` (not `[`) is what actually unboxes to a genuine scalar noisy value
# here -- male/female need to be that, not noisy *columns*, for `male >
# female` to give a posterior probability instead of an ordinary logical
# mask over observed values. See tasks.md's Task 3 for more.
#
# Setup (from the repo root):
#   uv sync --extra dev
#   Rscript example/usability_tasks/task3.R
#
# See tasks.md / solutions.md in this directory for the analysis exercises
# this script's output feeds into.

pkgload::load_all("r/noisyvalue", quiet = TRUE)
noisyvalue_init(venv = ".venv", src = "src")

population <- get_decennial(
  geography = "county",
  variables = c(male = "P12_002N", female = "P12_026N"),
  year = 2020,
  sumfile = "dhc",
  state = "VT")

chittenden_county <- population[population$GEOID == "50007", ]
male <- chittenden_county$value[chittenden_county$variable == "male"]
female <- chittenden_county$value[chittenden_county$variable == "female"]

cat("P(male > female):", round(noisy_prob(male > female), 4), "\n")
