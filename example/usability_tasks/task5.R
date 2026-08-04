# Prints the total noisy male and female resident counts across Chittenden
# and Orange counties, Vermont, as released by the actual 2020 DHC Noisy
# Measurement File.
#
# Analog of uvm-plaid/census_utility/tasks_census/task5.R. See task1.R's
# comments for why get_dhc_variables() spells "male"/"female" as a query
# plus an axis filter rather than a tidycensus-style variable code.
#
# Setup (from the repo root):
#   uv sync --extra dev
#   Rscript example/usability_tasks/task5.R
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
orange_county <- population[population$geoid == "50017", ]
c_male <- chittenden_county$value[[which(chittenden_county$variable == "male")]]
c_female <- chittenden_county$value[[which(chittenden_county$variable == "female")]]
o_male <- orange_county$value[[which(orange_county$variable == "male")]]
o_female <- orange_county$value[[which(orange_county$variable == "female")]]

cat("Total male residents:", format(c_male + o_male), "\n")
cat("Total female residents:", format(c_female + o_female), "\n")
