#' Read DHC person NMF measurements as a noisy tibble
#'
#' Thin wrapper around the Python `noisyvalue.census.get_dhc`; see its
#' docstring for the full parameter semantics. This is the DHC *person*
#' product -- the one whose histogram resolves sex and single-year age,
#' which PL94 has neither of.
#'
#' @param geography One of `"us"`, `"state"`, or `"county"` (`"county"`
#'   requires `state`).
#' @param queries Query name or character vector of names (e.g.
#'   `"sex*hispanic"`).
#' @param state State FIPS code, postal abbreviation, or name; may be a
#'   vector.
#' @param county County FIPS code(s), to narrow a county-level read.
#' @param root Directory holding the fetched DHC parquet partitions.
#' @param nonnegative Truncate every posterior at zero.
#' @param apply_constraints Condition posteriors on the NMF constraint rows
#'   (the exact PL94 invariant, relgq age rules, and GQ bounds).
#' @return A tibble with one row per histogram cell: plain columns `geoid`,
#'   `geocode`, `aian`, `query`, one label column per axis (`relgq`, `sex`,
#'   `age`, `hispanic`, `cenrace`), `variance`, and a noisy `value` column.
#' @export
get_dhc <- function(geography, queries, state = NULL, county = NULL,
                     root = "data/2020-dhc-nmf-parquets",
                     nonnegative = TRUE, apply_constraints = TRUE) {
  df_py <- .nv("census")$get_dhc(
    geography, queries,
    state = state, county = county, root = root,
    nonnegative = nonnegative, apply_constraints = apply_constraints)
  as_noisy_tibble(df_py)
}

#' Read named DHC variables as a tidy long table, tidycensus-style
#'
#' A drop-in replacement for
#' `tidycensus::get_decennial(variables = c(name = "P12_002N", ...))` when
#' `sumfile = "dhc"`: `variables` takes the same literal Census variable
#' codes tidycensus does. Each code is resolved via the Python
#' `census.dhc_table_variables()` crosswalk into the `get_dhc()` query and
#' axis filter(s) that reconstruct that published cell from the actual NMF
#' measurements -- see its docstring for why that reconstruction is
#' necessary (the NMF was measured against query workloads, not published
#' table cells) and how it picks the lowest-variance query available for
#' each cell.
#'
#' Only table P12 ("Sex by Age") is covered so far, and only 43 of its 49
#' codes: codes from other tables (`"P1_..."`, `"H1_..."`, etc.) error
#' clearly, as do 6 of P12's own codes -- Male/Female 20 years, 21 years,
#' and 22 to 24 years. Those six have no reliable NMF reconstruction: the
#' only query with single-year age resolution is `"detailed"`, and summing
#' the ~5,000 detailed cells one of these buckets marginalizes over
#' produces a posterior far from its own observed value (most of those
#' cells fall in PL94 constraint blocks with 3+ still-free cells, where the
#' block total is only an upper bound rather than a tight joint
#' constraint -- see Python's `census.dhc_table_variables()` docstring).
#' Every other P12 code checks out: its reconstruction's credible interval
#' sits tightly around its own observed value, as expected.
#'
#' Unlike `tidycensus::get_decennial()`, the returned tibble has no `NAME`
#' column: that requires a live FIPS-to-name geography lookup this package
#' doesn't have offline. Filter on `GEOID` instead.
#'
#' @param geography One of `"us"`, `"state"`, or `"county"` (`"county"`
#'   requires `state`).
#' @param variables A named character vector of literal Census variable
#'   codes, exactly as passed to `tidycensus::get_decennial()`, e.g.
#'   `c(male = "P12_002N", female = "P12_026N")`.
#' @param year Must be `2020` -- the only year this package has DHC data
#'   for. Accepted for call-shape parity with `tidycensus::get_decennial()`.
#' @param sumfile Must be `"dhc"`; other `tidycensus` sumfiles (e.g. `"pl"`)
#'   aren't wrapped by this function yet.
#' @param geometry Must be `FALSE`; this package doesn't return `sf`
#'   geometries. Accepted for call-shape parity with
#'   `tidycensus::get_decennial()`.
#' @param state,county,root,nonnegative,apply_constraints Passed through to
#'   [get_dhc()].
#' @return A tibble with columns `GEOID`, `geocode`, `aian`, `variable`, and
#'   a noisy `value` column -- one row per geography per requested
#'   variable, the same shape `tidycensus::get_decennial()` returns (plus
#'   `geocode`/`aian`, which carry DP-relevant provenance tidycensus has no
#'   analog for).
#' @examples
#' \dontrun{
#' get_decennial(
#'   "county", state = "VT", county = "007", year = 2020, sumfile = "dhc",
#'   variables = c(male = "P12_002N", female = "P12_026N"))
#' }
#' @export
get_decennial <- function(geography, variables, year = 2020, sumfile = "dhc",
                           state = NULL, county = NULL, geometry = FALSE,
                           root = "data/2020-dhc-nmf-parquets",
                           nonnegative = TRUE, apply_constraints = TRUE) {
  if (!identical(year, 2020)) {
    stop("get_decennial() only supports year = 2020", call. = FALSE)
  }
  if (!identical(sumfile, "dhc")) {
    stop("get_decennial() only supports sumfile = \"dhc\" (got \"", sumfile,
         "\"); PL94 (sumfile = \"pl\") isn't wrapped yet", call. = FALSE)
  }
  if (!identical(geometry, FALSE)) {
    stop("get_decennial() does not support geometry = TRUE", call. = FALSE)
  }
  if (is.null(names(variables)) || any(names(variables) == "")) {
    stop("`variables` must be a fully named character vector", call. = FALSE)
  }

  specs <- lapply(variables, .lookup_variable_spec)

  queries <- unique(vapply(specs, function(s) s$query, character(1)))
  df <- get_dhc(geography, queries, state = state, county = county, root = root,
                nonnegative = nonnegative, apply_constraints = apply_constraints)

  parts <- lapply(names(specs), function(name) {
    .dhc_variable_rows(df, name, specs[[name]])
  })

  tibble::tibble(
    GEOID = unlist(lapply(parts, `[[`, "geoid")),
    geocode = unlist(lapply(parts, `[[`, "geocode")),
    aian = unlist(lapply(parts, `[[`, "aian")),
    variable = unlist(lapply(parts, `[[`, "variable")),
    value = do.call(c, lapply(parts, `[[`, "value"))
  )
}

# Looks up a literal Census variable code (e.g. "P12_002N") via the Python
# crosswalk census.dhc_table_variables(), which knows the query and axis
# filters that reconstruct that published cell from the real NMF
# measurements. The code's table prefix (before the first "_") selects
# which crosswalk to consult -- there's no separate `table` argument,
# matching how tidycensus infers the table from the code itself.
.lookup_variable_spec <- function(code) {
  if (!is.character(code) || length(code) != 1 || is.na(code)) {
    stop("Each `variables` entry must be a single Census variable code string",
         call. = FALSE)
  }
  table <- sub("_.*$", "", code)
  catalog <- reticulate::py_to_r(.nv("census")$dhc_table_variables(table))
  if (!(code %in% names(catalog))) {
    unsupported <- reticulate::py_to_r(.nv("census")$unsupported_dhc_table_variables(table))
    if (code %in% names(unsupported)) {
      stop("Variable code '", code, "' is not supported: ", unsupported[[code]], call. = FALSE)
    }
    stop("Unknown variable code '", code, "' for table '", table, "'", call. = FALSE)
  }
  spec <- catalog[[code]]
  # reticulate converts a Python tuple-of-str nested in a dict to an R list,
  # not a character vector -- flatten each axis filter so `%in%` in
  # .dhc_variable_rows() sees a plain character vector.
  for (axis in setdiff(names(spec), c("query", "label"))) {
    spec[[axis]] <- unlist(spec[[axis]], use.names = FALSE)
  }
  spec
}

# Filter `df` (get_dhc() output) down to one variable's cells, then sum
# those cells within each geography. A filter axis may name more than one
# level (e.g. P12's "22 to 24 years" sums three single-year age cells) --
# `%in%` subsumes the single-level case, so this doesn't special-case it.
# Deliberately indexes the plain columns and the `value` noisy_vec directly
# with logical/integer vectors rather than subsetting `df` itself as a
# tibble/data.frame -- that keeps this on the noisy_vec indexing path
# that's actually implemented (`[.noisy_vec`), instead of tibble's
# row-subsetting machinery, which this package's vctrs extension points
# don't promise to support.
.dhc_variable_rows <- function(df, name, spec) {
  if (is.null(spec$query)) {
    stop("variables[[\"", name, "\"]] must include a `query` entry", call. = FALSE)
  }
  filters <- spec[setdiff(names(spec), c("query", "label"))]

  mask <- df$query == spec$query
  for (axis in names(filters)) {
    if (!(axis %in% names(df))) {
      stop("Unknown axis '", axis, "' for query '", spec$query, "'", call. = FALSE)
    }
    mask <- mask & (df[[axis]] %in% filters[[axis]])
  }
  if (!any(mask)) {
    stop("No cells matched variable '", name, "' (query '", spec$query, "')", call. = FALSE)
  }

  geoid <- df$geoid[mask]
  geocode <- df$geocode[mask]
  aian <- df$aian[mask]
  value <- df$value[mask]

  order <- unique(geoid)
  idx <- split(seq_along(geoid), geoid)[order]
  first <- vapply(idx, `[[`, integer(1), 1)

  list(
    geoid = order,
    geocode = geocode[first],
    aian = aian[first],
    variable = rep(name, length(order)),
    value = do.call(c, lapply(idx, function(i) sum(value[i])))
  )
}
