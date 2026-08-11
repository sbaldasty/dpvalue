# Prints the noisy male and female resident counts for Orange County,
# Vermont, as released by the actual 2020 DHC Noisy Measurement File --
# not an assumed noise model, the real differential-privacy mechanism.
#
# Analog of uvm-plaid/census_utility/tasks_census/task2.R, which reads the
# same two counts from the published (already noise-free) tidycensus table.
# See task1.R's comments for how get_decennial() resolves tidycensus-style
# variable codes like "P12_002N" back to real NMF measurements.
#
# Setup (from the repo root):
#   uv sync --extra dev
#   Rscript example/usability_tasks/task2.R
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

# `[[` (not `[`) is what actually unboxes to a genuine scalar noisy value
# here -- see tasks.md's note on task 3 for why that distinction matters.
orange_county <- population[population$GEOID == "50017", ]
male <- orange_county$value[[which(orange_county$variable == "male")]]
female <- orange_county$value[[which(orange_county$variable == "female")]]

cat("Male residents in Orange County, Vermont:", format(male), "\n")
cat("Female residents in Orange County, Vermont:", format(female), "\n")

# Task 4: posterior probability that Orange County's true male population
# exceeds its true female population.
cat("P(male > female):", round(noisy_prob(male > female), 4), "\n")
