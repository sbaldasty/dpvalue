# Solutions

Each answer below was produced by loading `task1.R`/`task2.R`/`task5.R`'s
data and then calling `noisy_credible_interval()` / `noisy_prob()` from the
`noisyvalue` R package, e.g.:

```r
pkgload::load_all("r/noisyvalue", quiet = TRUE)
noisyvalue_init(venv = ".venv", src = "src")

population <- get_decennial(
  geography = "county",
  variables = c(male = "P12_002N", female = "P12_026N"),
  year = 2020, sumfile = "dhc", state = "VT", county = "007")
male <- population$value[[which(population$variable == "male")]]
female <- population$value[[which(population$variable == "female")]]

noisy_credible_interval(male, n = 20000, rng = 100)
noisy_prob(male > female, n = 20000, rng = 102)
```

`noisy_credible_interval()` and `noisy_prob()` both work by drawing `n`
samples from the posterior, so the bounds below are Monte Carlo estimates
(`~` throughout) that will vary slightly run to run, or with a different
`n`/`rng`; larger `n` narrows that variation.

The textbook version of these exercises (see
`uvm-plaid/census_utility/tasks_census`) assumes the added noise is
`N(0, 97)` for every count, independently. Neither half of that assumption
holds here: `get_dhc` reads the *actual* per-cell measurement variance
recorded in the NMF, which for this query is 32.26 -- about a third of the
textbook guess -- and it applies the invariant that the published PL94
table was exact in the DHC run, which correlates the male and female
releases within each county rather than leaving them independent. Both
effects make these intervals tighter and the group comparisons more
decisive than the textbook calculation would predict.

## Task 1

Chittenden County released counts: male ~82118, female ~86216.

- Male residents: 95% credible interval ~(82101, 82124)
- Female residents: 95% credible interval ~(86200, 86222)

## Task 2

Orange County released counts: male ~14619, female ~14629.

- Male residents: 95% credible interval ~(14623, 14645)
- Female residents: 95% credible interval ~(14632, 14654)

## Task 3

`noisy_prob(male > female)` for Chittenden County is ~0 (indistinguishable
from zero at 20,000 draws -- the released gap of about 4,098 residents is
enormous relative to each count's few-dozen-resident uncertainty). This
easily clears the 0.05/0.95 threshold: Chittenden County has more women
than men.

## Task 4

`noisy_prob(male > female)` for Orange County is ~0.19, so
`noisy_prob(female > male)` is ~0.81. Neither clears the 0.05/0.95
threshold, so the data are inconclusive about whether Orange County has
more men or more women -- though unlike the textbook version of this task
(whose naive z-test put the two sexes at essentially a coin flip, p =
0.28), the real release mechanism's correlation structure does give a
noticeable lean toward more women.

## Task 5

Combined Chittenden + Orange released counts: male ~96737, female
~100845.

- Male residents: 95% credible interval ~(96730, 96762)
- Female residents: 95% credible interval ~(100838, 100870)
