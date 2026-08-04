# Prints the noisy male and female resident counts for Chittenden County,
# Vermont, as released by the actual 2020 DHC Noisy Measurement File --
# not an assumed noise model, the real differential-privacy mechanism.
#
# Analog of uvm-plaid/census_utility/tasks_census/task1.R, which reads the
# same two counts from the published (already noise-free) tidycensus table.
# get_dhc_variables() mirrors get_decennial()'s named-variables call shape;
# see its help (?get_dhc_variables) for why "male"/"female" have to be
# spelled as a query plus an axis filter rather than a variable code like
# tidycensus's "P12_002N" -- the NMF has no equivalent fixed catalog.
#
# Setup (from the repo root):
#   uv sync --extra dev
#   Rscript example/usability_tasks/task1.R
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
chittenden_county <- population[population$geoid == "50007", ]
male <- chittenden_county$value[[which(chittenden_county$variable == "male")]]
female <- chittenden_county$value[[which(chittenden_county$variable == "female")]]

cat("Male residents in Chittenden County, Vermont:", format(male), "\n")
cat("Female residents in Chittenden County, Vermont:", format(female), "\n")
