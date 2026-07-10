.noisyvalue_env <- new.env(parent = emptyenv())

#' Point reticulate at the noisyvalue Python package
#'
#' Call this once, before anything else in this package, to select the
#' virtualenv created by `uv sync --extra dev` and import the Python modules
#' the rest of the package wraps. Mirrors the setup in
#' `example/reticulate_demo.R`.
#'
#' @param venv Path to the virtualenv (defaults to `.venv` in the current
#'   working directory, i.e. the repo root).
#' @param src Path to the `src` directory containing the `noisyvalue` Python
#'   package.
#' @export
noisyvalue_init <- function(venv = ".venv", src = "src") {
  reticulate::use_virtualenv(normalizePath(venv), required = TRUE)

  sys <- reticulate::import("sys")
  src <- normalizePath(src)
  if (!(src %in% sys$path)) {
    sys$path <- c(sys$path, src)
  }

  .noisyvalue_env$builtins <- reticulate::import_builtins(convert = FALSE)
  .noisyvalue_env$pd <- reticulate::import("pandas", convert = FALSE)
  .noisyvalue_env$np <- reticulate::import("numpy", convert = FALSE)
  .noisyvalue_env$core <- reticulate::import("noisyvalue.core", convert = FALSE)
  .noisyvalue_env$io <- reticulate::import("noisyvalue.io", convert = FALSE)
  .noisyvalue_env$pdext <- reticulate::import("noisyvalue.pandas_ext", convert = FALSE)

  invisible(TRUE)
}

.nv <- function(name) {
  val <- .noisyvalue_env[[name]]
  if (is.null(val)) {
    stop("noisyvalue_init() must be called before using this function", call. = FALSE)
  }
  val
}
