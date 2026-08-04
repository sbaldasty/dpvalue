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
#' A convenience layer over [get_dhc()] for callers coming from
#' `tidycensus::get_decennial(variables = c(name = "P12_002N", ...))`. The
#' NMF has no equivalent catalog of individually pre-tabulated variable
#' codes -- differential-privacy noise was measured against cross-tabulation
#' *query workloads*, not a fixed set of published cells -- so there is no
#' Python-layer analog of a `"P12_002N"`-style variable name to look up.
#' Each `variables` entry here instead names a query plus the axis label(s)
#' that pick out its cell(s); axes left unspecified are summed over. That
#' sum is exactly the same linear composition on the underlying posterior
#' that a `dplyr::filter() |> summarise(sum(value))` pipeline over
#' [get_dhc()]'s output would perform -- this just gives it a friendly name
#' and tidycensus's one-row-per-geography-per-variable shape, so the two
#' fetches read the same at the call site even though the DHC one is
#' visibly reducing real query cells rather than looking up a variable that
#' was already published.
#'
#' @param geography One of `"us"`, `"state"`, or `"county"` (`"county"`
#'   requires `state`).
#' @param variables A named list. Each element is itself a list with a
#'   `query` entry (a query name from Python's `dhc_queries()`, e.g.
#'   `"sex*hispanic"`) plus zero or more `axis = "level"` filters (e.g.
#'   `sex = "male"`) narrowing which cells of that query get summed into
#'   the variable. Axes of the query not named in the filter are summed
#'   over entirely.
#' @param state,county,root,nonnegative,apply_constraints Passed through to
#'   [get_dhc()].
#' @return A tibble with columns `geoid`, `geocode`, `aian`, `variable`, and
#'   a noisy `value` column -- one row per geography per requested
#'   variable, the same shape `tidycensus::get_decennial()` returns.
#' @examples
#' \dontrun{
#' get_dhc_variables(
#'   "county", state = "VT", county = "007",
#'   variables = list(
#'     male = list(query = "sex*hispanic", sex = "male"),
#'     female = list(query = "sex*hispanic", sex = "female")))
#' }
#' @export
get_dhc_variables <- function(geography, variables, state = NULL, county = NULL,
                               root = "data/2020-dhc-nmf-parquets",
                               nonnegative = TRUE, apply_constraints = TRUE) {
  if (is.null(names(variables)) || any(names(variables) == "")) {
    stop("`variables` must be a fully named list", call. = FALSE)
  }

  queries <- unique(vapply(variables, function(v) v$query, character(1)))
  df <- get_dhc(geography, queries, state = state, county = county, root = root,
                nonnegative = nonnegative, apply_constraints = apply_constraints)

  parts <- lapply(names(variables), function(name) {
    .dhc_variable_rows(df, name, variables[[name]])
  })

  tibble::tibble(
    geoid = unlist(lapply(parts, `[[`, "geoid")),
    geocode = unlist(lapply(parts, `[[`, "geocode")),
    aian = unlist(lapply(parts, `[[`, "aian")),
    variable = unlist(lapply(parts, `[[`, "variable")),
    value = do.call(c, lapply(parts, `[[`, "value"))
  )
}

# Filter `df` (get_dhc() output) down to one `variables[[name]]` entry's
# cells, then sum those cells within each geography. Deliberately indexes
# the plain columns and the `value` noisy_vec directly with logical/integer
# vectors rather than subsetting `df` itself as a tibble/data.frame -- that
# keeps this on the noisy_vec indexing path that's actually implemented
# (`[.noisy_vec`), instead of tibble's row-subsetting machinery, which this
# package's vctrs extension points don't promise to support.
.dhc_variable_rows <- function(df, name, spec) {
  if (is.null(spec$query)) {
    stop("variables[[\"", name, "\"]] must include a `query` entry", call. = FALSE)
  }
  filters <- spec[setdiff(names(spec), "query")]

  mask <- df$query == spec$query
  for (axis in names(filters)) {
    if (!(axis %in% names(df))) {
      stop("Unknown axis '", axis, "' for query '", spec$query, "'", call. = FALSE)
    }
    mask <- mask & (df[[axis]] == filters[[axis]])
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
