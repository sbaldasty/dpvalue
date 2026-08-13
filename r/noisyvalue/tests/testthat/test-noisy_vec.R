skip_if_not(getOption("noisyvalue.test_env_ready", FALSE), "noisyvalue Python venv not available")

.mk_float_column <- function(obs) {
  core <- reticulate::import("noisyvalue.core", convert = FALSE)
  pd <- reticulate::import("pandas", convert = FALSE)
  pdext <- reticulate::import("noisyvalue.pandas", convert = FALSE)
  builtins <- reticulate::import_builtins(convert = FALSE)
  vals <- lapply(seq_along(obs), function(i) {
    core$NoisyFloat$gaussian(0, 1, obs = obs[i], rng = as.integer(i))
  })
  arr <- pdext$NoisyFloatArray$`_from_sequence`(builtins$list(vals))
  noisy_vec(pd$Series(arr))
}

test_that("a noisy column has the right length and class", {
  v <- .mk_float_column(c(1, 2, 3))
  expect_equal(length(v), 3L)
  expect_s3_class(v, "noisy_vec")
})

test_that("arithmetic between two noisy columns composes observed values and stays a noisy_vec", {
  a <- .mk_float_column(c(1, 2, 3))
  b <- .mk_float_column(c(10, 20, 30))
  s <- a + b
  expect_s3_class(s, "noisy_vec")
  obs <- as.vector(reticulate::py_to_r(.py(s)$array$`_obs`))
  expect_equal(obs, c(11, 22, 33))
})

test_that("comparing noisy columns for filtering gives a plain logical mask, not a noisy_vec", {
  a <- .mk_float_column(c(1, 2, 3))
  mask <- a > 1.5
  expect_type(mask, "logical")
  expect_equal(mask, c(FALSE, TRUE, TRUE))
})

test_that("comparing length-1 noisy columns preserves NoisyBool", {
  a <- .mk_float_column(c(1, 2, 3))[2]
  b <- .mk_float_column(c(10, 20, 30))[2]
  cmp <- a < b
  expect_s3_class(cmp, "noisy_vec")
  expect_false(is.logical(cmp))
  expect_equal(.dtype(cmp), "noisybool")
})

test_that("mean() of a noisy column is a length-1 noisy scalar", {
  a <- .mk_float_column(c(1, 2, 3))
  m <- mean(a)
  expect_s3_class(m, "noisy_vec")
  expect_equal(length(m), 1L)
})

test_that("comparing an aggregated scalar preserves NoisyBool instead of collapsing to logical", {
  a <- .mk_float_column(c(1, 2, 3))
  above <- mean(a) > 1
  expect_s3_class(above, "noisy_vec")
  expect_false(is.logical(above))
})

test_that("a noisy column round-trips through a tibble", {
  v <- .mk_float_column(c(1, 2, 3))
  tbl <- tibble::tibble(x = v)
  expect_equal(nrow(tbl), 3L)
  expect_s3_class(tbl$x, "noisy_vec")
})

test_that("filtering a tibble on a noisy comparison keeps matching rows with posterior intact", {
  skip_if_not_installed("dplyr")
  v <- .mk_float_column(c(1, 2, 3))
  tbl <- tibble::tibble(x = v)
  out <- dplyr::filter(tbl, x > 1.5)
  expect_equal(nrow(out), 2L)
  expect_s3_class(out$x, "noisy_vec")
})

test_that("c() concatenates noisy_vec objects", {
  a <- .mk_float_column(c(1, 2))
  b <- .mk_float_column(c(3, 4, 5))
  combined <- c(a, b)
  expect_equal(length(combined), 5L)
})

test_that("sum/mean/any/all are supported but other reductions are not", {
  a <- .mk_float_column(c(1, 2, 3))
  expect_s3_class(sum(a), "noisy_vec")
  expect_error(median(a))
})
