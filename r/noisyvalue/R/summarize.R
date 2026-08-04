# Both public functions below need a genuine scalar Python object to call
# .credible_interval()/.sample() on. `x` may already be one (e.g. the
# result of sum()/mean()), or it may be a length-1 *Series* -- e.g.
# `tbl$value[tbl$variable == "male"]`, which stays Series-backed under `[`
# even once filtered down to one row (`[[` is what actually unboxes to a
# scalar; see noisy_vec.R). Both are "a single noisy value" from the
# caller's point of view, so accept both and unbox the Series case here
# rather than making every caller know which flavor they're holding.
.scalar_py <- function(x, fname) {
  if (!inherits(x, "noisy_vec")) {
    stop(fname, "() expects a noisy_vec", call. = FALSE)
  }
  if (!.is_series(x)) {
    return(.py(x))
  }
  if (length(x) != 1) {
    stop(fname, "() expects a single noisy value, not a column of ", length(x),
         " -- reduce it first, e.g. with sum() or mean(), or select one row",
         call. = FALSE)
  }
  .py(x)$iloc[0L]
}

#' Credible interval for a scalar noisy value
#'
#' Draws `n` joint posterior samples of `x` (via the wrapped Python
#' `NoisyValue.credible_interval`) and returns the central `p` interval. If
#' `x` was built out of other noisy values via arithmetic (e.g. a sum or
#' difference), the samples correctly reflect any shared latent/noise
#' dependencies baked into its posterior.
#'
#' @param x A single noisy value -- e.g. one element of a noisy column
#'   (however it was selected down to one row), or the result of
#'   `sum()`/`mean()`/arithmetic on one.
#' @param p Central probability mass to cover (default 0.95).
#' @param n Number of joint posterior draws.
#' @param rng Optional integer seed, for reproducibility.
#' @return A length-2 numeric vector `c(lower, upper)`.
#' @export
noisy_credible_interval <- function(x, p = 0.95, n = 1000, rng = NULL) {
  py <- .scalar_py(x, "noisy_credible_interval")
  rng <- if (is.null(rng)) NULL else as.integer(rng)
  raw <- py$credible_interval(p = p, n = as.integer(n), rng = rng)
  as.vector(reticulate::py_to_r(raw))
}

#' Posterior probability that a noisy boolean holds
#'
#' `x` is typically the result of comparing two noisy values (e.g.
#' `male > female`), which is a `NoisyBool` rather than a plain logical --
#' comparing noisy values doesn't resolve to a definite answer, it has a
#' posterior probability. This draws `n` joint samples and returns the
#' fraction that were `TRUE`.
#'
#' @param x A single noisy value wrapping a `NoisyBool`.
#' @param n Number of joint posterior draws.
#' @param rng Optional integer seed, for reproducibility.
#' @return A single number between 0 and 1.
#' @export
noisy_prob <- function(x, n = 1000, rng = NULL) {
  if (inherits(x, "noisy_vec") && .dtype(x) != "noisybool") {
    stop("noisy_prob() expects a noisybool value, got: ", .dtype(x), call. = FALSE)
  }
  py <- .scalar_py(x, "noisy_prob")
  rng <- if (is.null(rng)) NULL else as.integer(rng)
  batch <- py$sample(n = as.integer(n), rng = rng)
  as.vector(reticulate::py_to_r(batch$mean()))
}
