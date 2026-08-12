# Prints the total noisy male and female resident counts across Chittenden
# and Orange counties, Vermont, as released by the actual 2020 DHC Noisy
# Measurement File.
#
# Analog of uvm-plaid/census_utility/tasks_census/task5.R. See task1.R's
# comments for how get_decennial() resolves tidycensus-style variable codes
# like "P12_002N" back to real NMF measurements.
#
# Setup (from the repo root):
#   uv sync --extra dev
#   Rscript example/usability_tasks/task5.R
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
orange_county <- population[population$GEOID == "50017", ]
c_male <- chittenden_county$value[chittenden_county$variable == "male"]
c_female <- chittenden_county$value[chittenden_county$variable == "female"]
o_male <- orange_county$value[orange_county$variable == "male"]
o_female <- orange_county$value[orange_county$variable == "female"]
male <- c_male + o_male
male_interval <- noisy_credible_interval(male)
female <- c_female + o_female
female_interval <- noisy_credible_interval(female)

cat("Total male residents:", format(male), "\n")
cat("Male residents credible interval:", format(male_interval), "\n")
cat("Total female residents:", format(female), "\n")
cat("Female residents credible interval:", format(female_interval), "\n")
