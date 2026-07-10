# These tests exercise the real Python noisyvalue library through
# reticulate, so they need the repo's uv-managed virtualenv. Locate the repo
# root by walking up from the working directory (which differs between
# `devtools::test()`/`testthat::test_local()`, run from the repo root, and
# `R CMD check`, run from a copy of the package directory) and skip the
# whole suite if the venv isn't there instead of failing noisily.

.locate_repo_root <- function() {
  dir <- normalizePath(getwd())
  for (i in 1:8) {
    if (file.exists(file.path(dir, "pyproject.toml")) &&
        dir.exists(file.path(dir, "src", "noisyvalue"))) {
      return(dir)
    }
    parent <- dirname(dir)
    if (identical(parent, dir)) break
    dir <- parent
  }
  NULL
}

.repo_root <- .locate_repo_root()
options(noisyvalue.test_env_ready = FALSE)

if (!is.null(.repo_root) && dir.exists(file.path(.repo_root, ".venv"))) {
  ready <- tryCatch({
    noisyvalue_init(venv = file.path(.repo_root, ".venv"), src = file.path(.repo_root, "src"))
    TRUE
  }, error = function(e) FALSE)
  options(noisyvalue.test_env_ready = ready)
}
