#' pandas dtype names reticulate's `pandas.Series` converter cannot handle
#' directly: pandas' nullable extension arrays (as opposed to plain numpy
#' `bool`/`object` dtypes). Confirmed against reticulate for `"boolean"`,
#' which fails with "names() applied to a non-vector" -- it appears to route
#' through a masked/categorical-array code path that a `BooleanDtype` array
#' doesn't support. `census.py`'s `aian` column carries this dtype (nullable
#' so a rolled-up geography whose contributors disagree can hold `<NA>`).
#' `"string"` is not listed here: it converts directly when a column has no
#' missing values, which is the common case for the columns this package
#' actually reads (`geoid`/`geocode`); a `<NA>` string still degrades to a
#' plain R list rather than erroring, a narrower gap left for later.
.UNSUPPORTED_EXTENSION_DTYPES <- c("boolean")

#' Convert a noisyvalue-backed pandas DataFrame into a tibble
#'
#' Plain columns convert through the normal reticulate pandas converter;
#' columns whose pandas dtype is `noisyfloat`/`noisyint`/`noisybool` are
#' wrapped with [noisy_vec()] instead, so R arithmetic/comparison operators
#' on them forward to the underlying Python objects.
#'
#' @param df_py A Python `pandas.DataFrame` object imported with
#'   `convert = FALSE` (e.g. as returned by `noisyvalue.io.load()`).
#' @export
as_noisy_tibble <- function(df_py) {
  if (!inherits(df_py, "pandas.DataFrame")) {
    stop("as_noisy_tibble() expects a Python pandas.DataFrame", call. = FALSE)
  }

  col_names <- reticulate::py_to_r(.nv("builtins")$list(df_py$columns))
  cols <- stats::setNames(vector("list", length(col_names)), col_names)

  for (name in col_names) {
    series <- df_py[[name]]
    dtype <- reticulate::py_to_r(series$dtype$name)
    cols[[name]] <- if (dtype %in% .NOISY_DTYPES) {
      new_noisy_vec(series, dtype)
    } else if (dtype %in% .UNSUPPORTED_EXTENSION_DTYPES) {
      # Route through plain object dtype, which reticulate maps to an R list
      # (one element per row, `pd.NA` mapped to `NA`) rather than erroring;
      # flatten that back to an atomic vector.
      unlist(reticulate::py_to_r(series$astype("object")), use.names = FALSE)
    } else {
      as.vector(reticulate::py_to_r(series))
    }
  }

  tibble::as_tibble(cols)
}

#' Load a container saved by `noisyvalue.io.save()` as a tibble
#'
#' @param path Path to the JSON file written by `noisyvalue.io.save()`. Must
#'   have been a `pandas.DataFrame` when saved.
#' @export
read_noisy_frame <- function(path) {
  df_py <- .nv("io")$load(normalizePath(path))
  as_noisy_tibble(df_py)
}
