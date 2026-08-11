# .lookup_variable_spec()/.dhc_variable_rows() need the Python venv (the
# former calls into census.dhc_table_variables()) but not the bundled
# parquet data, so these run whenever the venv is available.
skip_if_not(getOption("noisyvalue.test_env_ready", FALSE), "noisyvalue Python venv not available")

test_that("looks up a P12 code with a single-level sex filter", {
  spec <- .lookup_variable_spec("P12_002N")
  expect_equal(spec$query, "sex*hispanic")
  expect_equal(spec$sex, "male")
})

test_that("looks up a P12 code with a multi-level age filter", {
  spec <- .lookup_variable_spec("P12_025N")
  expect_equal(spec$query, "sex*age_38_groups")
  expect_equal(spec$sex, "male")
  expect_equal(spec$age, c("85-89", "90-94", "95-99", "100-104", "105-109", "110-115"))
})

test_that("errors clearly on a P12 code with no reliable NMF reconstruction", {
  expect_error(.lookup_variable_spec("P12_010N"), "not supported")
})

test_that("looks up the P12 grand total, which has no filters at all", {
  spec <- .lookup_variable_spec("P12_001N")
  expect_equal(spec$query, "sex*hispanic")
  expect_null(spec$sex)
  expect_null(spec$age)
})

test_that("errors on an unknown code within a supported table", {
  expect_error(.lookup_variable_spec("P12_999N"), "Unknown variable code")
})

test_that("errors on a code from a table that isn't wrapped yet", {
  expect_error(.lookup_variable_spec("P1_003N"), "only supports table")
})

test_that("get_decennial rejects unsupported year, sumfile, and geometry", {
  expect_error(
    get_decennial("county", c(male = "P12_002N"), year = 2010),
    "only supports year"
  )
  expect_error(
    get_decennial("county", c(male = "P12_002N"), sumfile = "pl"),
    "only supports sumfile"
  )
  expect_error(
    get_decennial("county", c(male = "P12_002N"), geometry = TRUE),
    "does not support geometry"
  )
})

test_that(".dhc_variable_rows sums cells matching a multi-level filter", {
  df <- tibble::tibble(
    geoid = c("A", "A", "A"),
    geocode = c("A", "A", "A"),
    aian = c(FALSE, FALSE, FALSE),
    query = c("detailed", "detailed", "detailed"),
    sex = c("male", "male", "male"),
    age = c("22", "23", "24"),
    value = c(1, 2, 3)
  )
  spec <- list(query = "detailed", sex = "male", age = c("22", "23", "24"))
  out <- .dhc_variable_rows(df, "twenty_two_to_24", spec)
  expect_equal(out$geoid, "A")
  expect_equal(unname(out$value), 6)
})
