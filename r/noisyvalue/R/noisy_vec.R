.NOISY_DTYPES <- c("noisyfloat", "noisyint", "noisybool")

# A noisy_vec is an integer proxy (1:n, standing in for row position) with
# the actual payload attached as attributes: `py` (a pandas Series for a
# column, or a scalar NoisyFloat/NoisyInt/NoisyBool Python object for a
# single value) and `dtype` (one of .NOISY_DTYPES). Payload as *attributes*
# rather than named list elements matters: vctrs' generic fallback paths for
# foreign S3 vectors (used e.g. while building a tibble's print prototype)
# reliably preserve arbitrary attributes but drop list/vector *names*, so a
# `list(py = ..., dtype = ...)` representation silently loses its fields
# under those paths while this one survives.
new_noisy_vec <- function(py, dtype) {
  n <- if (inherits(py, "pandas.Series")) as.integer(reticulate::py_to_r(.nv("builtins")$len(py))) else 1L
  structure(seq_len(n), py = py, dtype = dtype, class = "noisy_vec")
}

.py <- function(x) attr(x, "py")
.dtype <- function(x) attr(x, "dtype")

#' Wrap a noisyvalue-backed Python object as an R vector
#'
#' `py` is either a pandas Series whose dtype is `noisyfloat`/`noisyint`/
#' `noisybool` (a *column*), or a scalar `NoisyFloat`/`NoisyInt`/`NoisyBool`
#' Python object (e.g. the result of indexing a single element, or of
#' `mean()`/`sum()` on a column). Both cases share one R class: arithmetic
#' and comparison forward straight through to the wrapped Python object
#' either way, so nothing but printing/subsetting needs to tell them apart.
#'
#' @param py A Python object imported with `convert = FALSE`.
#' @export
noisy_vec <- function(py) new_noisy_vec(py, .noisy_dtype(py))

.noisy_dtype <- function(py) {
  if (inherits(py, "pandas.Series")) {
    name <- reticulate::py_to_r(py$dtype$name)
    if (!(name %in% .NOISY_DTYPES)) {
      stop("Not a noisyvalue-backed Series (dtype '", name, "')", call. = FALSE)
    }
    return(name)
  }
  if (inherits(py, "noisyvalue.core.NoisyFloat")) return("noisyfloat")
  if (inherits(py, "noisyvalue.core.NoisyInt")) return("noisyint")
  if (inherits(py, "noisyvalue.core.NoisyBool")) return("noisybool")
  stop("Not a recognized noisyvalue Python object", call. = FALSE)
}

.is_series <- function(x) inherits(.py(x), "pandas.Series")

.array_cls_for_dtype <- function(dtype) {
  pdext <- .nv("pdext")
  switch(dtype,
    noisyfloat = pdext$NoisyFloatArray,
    noisyint = pdext$NoisyIntArray,
    noisybool = pdext$NoisyBoolArray,
    stop("Unknown noisyvalue dtype: ", dtype, call. = FALSE)
  )
}

# A scalar-backed noisy_vec promoted to a length-1 Series, for c()/concat.
.as_series <- function(x) {
  if (.is_series(x)) return(.py(x))
  arr_cls <- .array_cls_for_dtype(.dtype(x))
  one <- .nv("builtins")$list(list(.py(x)))
  .nv("pd")$Series(arr_cls$`_from_sequence`(one))
}

.empty_noisy_vec <- function(dtype) {
  arr_cls <- .array_cls_for_dtype(dtype)
  empty <- arr_cls$`_from_sequence`(.nv("builtins")$list(list()))
  new_noisy_vec(.nv("pd")$Series(empty), dtype)
}

# Turn a raw value coming back from a Python call into the right R shape:
# NA for pd.NA, a noisy_vec for a noisyvalue Series/scalar, a plain logical
# vector for the concrete row-mask Series numeric comparisons produce, and
# an untouched pass-through for anything already plain (R value or a Python
# object our operators never see, e.g. NotImplemented).
.wrap_py_result <- function(raw) {
  if (!inherits(raw, "python.builtin.object")) return(raw)
  if (inherits(raw, "pandas.api.typing.NAType")) return(NA)

  if (inherits(raw, "pandas.Series")) {
    dtype <- reticulate::py_to_r(raw$dtype$name)
    if (dtype %in% .NOISY_DTYPES) return(new_noisy_vec(raw, dtype))
    # A plain (nullable) boolean Series -- the output of comparing noisy
    # columns for row filtering, not a noisyvalue-backed column -- so drop
    # straight to a normal R logical vector instead of wrapping it.
    vals <- reticulate::py_to_r(raw$to_numpy(dtype = "float64", na_value = .nv("np")$nan))
    return(as.logical(vals))
  }

  if (inherits(raw, "noisyvalue.core.NoisyValue")) return(noisy_vec(raw))

  reticulate::py_to_r(raw)
}

#' @export
length.noisy_vec <- function(x) length(unclass(x))

#' @export
format.noisy_vec <- function(x, ...) {
  if (.is_series(x)) {
    arr <- .py(x)$array
    obs <- reticulate::py_to_r(arr$`_obs`)
    mask <- reticulate::py_to_r(arr$`_mask`)
    ifelse(mask, NA_character_, paste0("~", format(obs, trim = TRUE)))
  } else {
    paste0("~", format(reticulate::py_to_r(.py(x)$`_obs`), trim = TRUE))
  }
}

#' @export
print.noisy_vec <- function(x, ...) {
  cat(sprintf("<noisy_vec[%s]: %d>\n", .dtype(x), length(x)))
  print(format(x), quote = FALSE)
  invisible(x)
}

#' @export
as.character.noisy_vec <- function(x, ...) format(x)

#' @export
is.na.noisy_vec <- function(x) {
  if (.is_series(x)) reticulate::py_to_r(.py(x)$array$`_mask`) else FALSE
}

#' @export
`[.noisy_vec` <- function(x, i, ...) {
  if (!.is_series(x)) stop("Cannot subset a scalar noisy value", call. = FALSE)
  n <- length(x)
  pos <- seq_len(n)[i]
  if (anyNA(pos)) stop("Index out of bounds", call. = FALSE)
  idx0 <- reticulate::np_array(as.integer(pos - 1L), dtype = "int64")
  new_noisy_vec(.py(x)$iloc[idx0], .dtype(x))
}

#' @export
`[[.noisy_vec` <- function(x, i, ...) {
  if (!.is_series(x)) stop("Cannot index a scalar noisy value", call. = FALSE)
  stopifnot(length(i) == 1)
  # A bare length-1 R integer unboxes to a Python scalar index, so .iloc[]
  # returns a scalar NoisyValue object rather than a length-1 Series here.
  new_noisy_vec(.py(x)$iloc[as.integer(i - 1L)], .dtype(x))
}

#' @export
c.noisy_vec <- function(...) {
  # unname(): a *named* list handed to `.nv("builtins")$list()` below would
  # reticulate-convert to a Python dict instead of a list, and Python's
  # `list(dict)` yields the dict's keys, not its values -- silently
  # replacing every element with its argument name. Bites easily via
  # `do.call(c, x)` where `x` came from something like `split()`/`lapply()`
  # over named groups.
  parts <- unname(list(...))
  if (!all(vapply(parts, inherits, logical(1), "noisy_vec"))) {
    stop("c() on a noisy_vec only supports combining noisy_vec objects", call. = FALSE)
  }
  dtypes <- unique(vapply(parts, .dtype, character(1)))
  if (length(dtypes) > 1) {
    stop("Cannot combine noisy_vec columns of different dtypes: ",
         paste(dtypes, collapse = ", "), call. = FALSE)
  }
  series_list <- .nv("builtins")$list(lapply(parts, .as_series))
  combined <- .nv("pd")$concat(series_list, ignore_index = TRUE)
  new_noisy_vec(combined, dtypes)
}

.OPS_SUPPORTED <- c("+", "-", "*", "/", "^", "%%", "%/%",
                     "==", "!=", "<", "<=", ">", ">=", "&", "|")

.OPS_NOISY_BOOL <- c("==", "!=", "<", "<=", ">", ">=", "&", "|")

#' Arithmetic and comparison on noisy_vec objects
#'
#' Forwards straight through to the wrapped Python object's own operator
#' (`NoisyValue`/`NoisyFloatArray`/... already implement these), so the
#' symbolic posterior composition happens on the Python side exactly as it
#' would from Python.
#' @export
Ops.noisy_vec <- function(e1, e2) {
  op <- .Generic
  if (!(op %in% .OPS_SUPPORTED)) {
    stop("Unsupported operator for noisy_vec: ", op, call. = FALSE)
  }

  py1 <- if (inherits(e1, "noisy_vec")) .py(e1) else e1
  py2 <- if (missing(e2)) NULL else if (inherits(e2, "noisy_vec")) .py(e2) else e2

  # Preserve posterior semantics for comparisons on length-1 noisy columns.
  # For pandas Series, `>`/`<`/etc. normally return observed boolean masks;
  # when both sides are one row, unbox to noisy scalars so callers can pass
  # results to noisy_prob() without having to manually use [[ ]].
  if (!is.null(py2) && op %in% .OPS_NOISY_BOOL &&
      inherits(py1, "pandas.Series") && inherits(py2, "pandas.Series") &&
      length(e1) == 1L && length(e2) == 1L) {
    py1 <- py1$iloc[0L]
    py2 <- py2$iloc[0L]
  }

  # Two pandas Series align by index *label*, not position -- two noisy_vec
  # values sliced from different tibbles/filters can carry unrelated index
  # labels even at the same length, which silently produces NA at every
  # position instead of erroring. R vector arithmetic is positional, so
  # reset both to a shared RangeIndex before combining to match that.
  if (!is.null(py2) && inherits(py1, "pandas.Series") && inherits(py2, "pandas.Series")) {
    py1 <- py1$reset_index(drop = TRUE)
    py2 <- py2$reset_index(drop = TRUE)
  }

  fn <- get(op, envir = baseenv())
  raw <- if (is.null(py2)) fn(py1) else fn(py1, py2)
  .wrap_py_result(raw)
}

#' @export
`!.noisy_vec` <- function(x) .wrap_py_result(!.py(x))

.reduce <- function(x, op, skipna) {
  if (!.is_series(x)) stop("Cannot reduce a scalar noisy value", call. = FALSE)
  py <- .py(x)
  raw <- switch(op,
    sum = py$sum(skipna = skipna),
    mean = py$mean(skipna = skipna),
    any = py$any(skipna = skipna),
    all = py$all(skipna = skipna),
    stop("Unsupported reduction: ", op, call. = FALSE)
  )
  .wrap_py_result(raw)
}

.SUMMARY_SUPPORTED <- c("sum", "any", "all")

#' @export
Summary.noisy_vec <- function(..., na.rm = FALSE) {
  op <- .Generic
  if (!(op %in% .SUMMARY_SUPPORTED)) {
    stop("noisy_vec does not support ", op, "() -- there's no well-defined ",
         "posterior combination for it (only sum/mean/any/all are supported)",
         call. = FALSE)
  }
  parts <- list(...)
  x <- if (length(parts) == 1) parts[[1]] else do.call(c, parts)
  .reduce(x, op, skipna = isTRUE(na.rm))
}

#' @export
mean.noisy_vec <- function(x, na.rm = FALSE, ...) .reduce(x, "mean", skipna = isTRUE(na.rm))

# The vctrs proxy backing a noisy_vec is a bare 1:n position placeholder (see
# new_noisy_vec() above), not real numeric data -- vctrs-aware code (tibble,
# dplyr, our own Ops/Summary/mean methods) always goes through vec_proxy()/
# vec_restore()/Ops.noisy_vec and never sees it directly, but a handful of
# base R generics call their .default method straight on that raw data
# without going through vctrs at all. Left alone, those would silently
# compute a "result" from the row positions instead of erroring -- e.g.
# median(noisy_col) would quietly return the median row *position*. Guard
# the ones an analyst is most likely to reach for.

#' @export
Math.noisy_vec <- function(x, ...) {
  stop("noisy_vec does not support ", .Generic, "() in this R wrapper", call. = FALSE)
}

#' @export
median.noisy_vec <- function(x, na.rm = FALSE, ...) {
  stop("noisy_vec does not support median() -- there's no well-defined posterior ",
       "combination for it (only sum/mean/any/all are supported)", call. = FALSE)
}

#' @export
quantile.noisy_vec <- function(x, ...) {
  stop("noisy_vec does not support quantile()", call. = FALSE)
}

#' @export
sort.noisy_vec <- function(x, ...) {
  stop("noisy_vec does not support sort() -- there's no ordering on posteriors; ",
       "sort the tibble by a plain column instead", call. = FALSE)
}

#' @export
unique.noisy_vec <- function(x, ...) {
  stop("noisy_vec does not support unique() -- matches the Python library, which ",
       "leaves unique/factorize/isin unsupported for noisy columns by design",
       call. = FALSE)
}

# vctrs only needs a size-determining proxy and a way to rebuild from a
# sliced one to treat a foreign S3 class as a real vector -- this is what
# lets a noisy_vec live as a tibble column and survive dplyr::filter/
# arrange/mutate, without reimplementing vctrs' whole arithmetic protocol
# (Ops.noisy_vec above already covers arithmetic/comparison directly).

#' @importFrom vctrs vec_proxy
#' @export
vec_proxy.noisy_vec <- function(x, ...) unclass(x)

#' @importFrom vctrs vec_restore
#' @export
vec_restore.noisy_vec <- function(x, to, ...) {
  if (!.is_series(to)) return(to)  # slicing a scalar; nothing meaningful to restore
  if (length(x) > 0 && length(to) == 0) {
    # `to` is an empty ptype shell here, not real data -- this is vctrs
    # combining noisy_vec pieces from *different* source Series (e.g.
    # dplyr::bind_rows()/vctrs::vec_c() across two tibbles), which the
    # proxy/restore extension point can't support: there's no single
    # backing Series left to index into. c() on noisy_vec objects directly
    # (via pandas concat) still works fine for that case.
    stop("Combining noisy_vec columns from different source objects (e.g. via ",
         "dplyr::bind_rows() or vctrs::vec_c()) is not supported -- use c() to ",
         "concatenate noisy_vec objects directly instead.", call. = FALSE)
  }
  idx0 <- reticulate::np_array(as.integer(x - 1L), dtype = "int64")
  new_noisy_vec(.py(to)$iloc[idx0], .dtype(to))
}

#' @importFrom vctrs vec_ptype_abbr
#' @export
vec_ptype_abbr.noisy_vec <- function(x, ...) .dtype(x)

#' @importFrom vctrs vec_ptype_full
#' @export
vec_ptype_full.noisy_vec <- function(x, ...) paste0("noisy_vec<", .dtype(x), ">")

# Without these, vctrs falls back to a generic mechanism for combining two
# "foreign" S3 vectors when it needs a common type, e.g. while building a
# tibble's 0-row prototype for printing. Registering them explicitly avoids
# relying on that fallback path.

#' @importFrom vctrs vec_ptype2
#' @export
vec_ptype2.noisy_vec.noisy_vec <- function(x, y, ...) {
  if (!identical(.dtype(x), .dtype(y))) {
    stop("Cannot combine noisy_vec columns of different dtypes: ",
         .dtype(x), " vs ", .dtype(y), call. = FALSE)
  }
  .empty_noisy_vec(.dtype(x))
}

#' @importFrom vctrs vec_cast
#' @export
vec_cast.noisy_vec.noisy_vec <- function(x, to, ...) x
