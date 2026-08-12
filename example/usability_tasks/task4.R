# Estimates the posterior probability that Orange County's true male
# population exceeds its true female population, using the actual 2020 DHC
# Noisy Measurement File.
#
# Analog of uvm-plaid/census_utility/tasks_census/task4.R. See task3.R's
# comments for why this script computes a posterior probability directly
# instead of the two released-value booleans the textbook version prints.
#
# Setup (from the repo root):
#   uv sync --extra dev
#   Rscript example/usability_tasks/task4.R
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

orange_county <- population[population$GEOID == "50017", ]
male <- orange_county$value[orange_county$variable == "male"]
female <- orange_county$value[orange_county$variable == "female"]

cat("P(male > female):", round(noisy_prob(male > female), 4), "\n")
