#install.packages(c("tidycensus", "dplyr", "tidyr", "purrr"))
library(tidycensus)
library(dplyr)
library(tidyr)
library(purrr)

# census_api_key("REDACTED_CENSUS_API_KEY", install = TRUE)

# Glaeser & Vigdor's main segregation statistic:
# D = 0.5 * sum_i | black_i / Black_total - nonblack_i / Nonblack_total |
dissimilarity <- function(black, nonblack) {
  B <- sum(black, na.rm = TRUE)
  N <- sum(nonblack, na.rm = TRUE)
  0.5 * sum(abs(black / B - nonblack / N), na.rm = TRUE)
}

# Approximate 2000 Detroit MSA counties.
# Adjust this list if you want the exact PMSA/CMSA definition used in the paper.
detroit_counties <- c("Wayne", "Oakland", "Macomb", "Lapeer", "Livingston", "St. Clair")

vars_2000 <- c(
    total = "P003001",  # Total population
    white = "P003003",
    black = "P003004"   # Black or African American alone
)

detroit_tracts <- map_dfr(detroit_counties, \(cty) {
  get_decennial(
    geography = "tract",
    variables = vars_2000,
    year = 2000,
    sumfile = "sf1",
    state = "MI",
    county = cty,
    geometry = FALSE
  )
})

detroit_counts <- detroit_tracts %>%
  select(GEOID, NAME, variable, value) %>%
  pivot_wider(names_from = variable, values_from = value) %>%
  mutate(nonblack = total - black) %>%
  filter(total > 0)

D <- dissimilarity(detroit_counts$black, detroit_counts$nonblack)

cat("Detroit 2000 Black/non-Black dissimilarity index:", round(D, 3), "\n")
cat("As percentage:", round(100 * D, 1), "\n")
cat("Numbers:", str(detroit_counts))
print(detroit_counts)

## load_variables(2000, "sf1") |>
##   dplyr::filter(grepl("^P003", name)) |>
##   print(n = 20)
