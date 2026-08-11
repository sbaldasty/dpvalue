# Prints the noisy male and female resident counts for Chittenden County,
# Vermont, as released by the actual 2020 DHC Noisy Measurement File --
# not an assumed noise model, the real differential-privacy mechanism.
#
# Analog of uvm-plaid/census_utility/tasks_census/task1.R, which reads the
# same two counts from the published (already noise-free) tidycensus table
# via the live Census API. get_decennial() takes the same literal Census
# variable codes tidycensus::get_decennial() does -- "P12_002N"/"P12_026N"
# are table P12's ("Sex by Age") Male/Female rows; see its help
# (?get_decennial) for how each code gets resolved back into the NMF query
# and axis filter(s) that reconstruct it from the actual differential-
# privacy measurements.
#
# Setup (from the repo root):
#   uv sync --extra dev
#   Rscript example/usability_tasks/task1.R
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

cat("Male residents in Chittenden County, Vermont:", format(male), "\n")
cat("Female residents in Chittenden County, Vermont:", format(female), "\n")