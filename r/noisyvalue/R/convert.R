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
