# Prints the noisy male and female resident counts for Orange County,
# Vermont, as released by the actual 2020 DHC Noisy Measurement File --
# not an assumed noise model, the real differential-privacy mechanism.
#
# Analog of uvm-plaid/census_utility/tasks_census/task2.R, which reads the
# same two counts from the published (already noise-free) tidycensus table.
# See task1.R's comments for why get_dhc_variables() spells "male"/"female"
# as a query plus an axis filter rather than a tidycensus-style variable
# code.
#
# Setup (from the repo root):
#   uv sync --extra dev
#   Rscript example/usability_tasks/task2.R
#
# See tasks.md / solutions.md in this directory for the analysis exercises
# this script's output feeds into.

pkgload::load_all("r/noisyvalue", quiet = TRUE)
noisyvalue_init(venv = ".venv", src = "src")

population <- get_dhc_variables(
  geography = "county",
  variables = list(
    male = list(query = "sex*hispanic", sex = "male"),
    female = list(query = "sex*hispanic", sex = "female")),
  state = "VT")

# `[[` (not `[`) is what actually unboxes to a genuine scalar noisy value
# here -- see tasks.md's note on task 3 for why that distinction matters.
orange_county <- population[population$geoid == "50017", ]
male <- orange_county$value[[which(orange_county$variable == "male")]]
female <- orange_county$value[[which(orange_county$variable == "female")]]

cat("Male residents in Orange County, Vermont:", format(male), "\n")
cat("Female residents in Orange County, Vermont:", format(female), "\n")
