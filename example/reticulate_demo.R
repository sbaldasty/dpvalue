# Minimal demo: call the Python noisyvalue library from R via reticulate.
#
# No R package, no wrapper classes -- just enough to see a real Python object
# (a NoisyInt) cross into R and back. Run with:
#   Rscript example/reticulate_demo.R
# (assumes you're running from the repo root, where `uv sync --extra dev`
# has already created .venv)

library(reticulate)

use_virtualenv(file.path(getwd(), ".venv"), required = TRUE)

# Python modules now live under `src/noisyvalue`, so add `src` to sys.path
# and import from the `noisyvalue` package.
sys <- import("sys")
sys$path <- c(sys$path, file.path(getwd(), "src"))

core <- import("noisyvalue.core", convert = FALSE)
graph <- import("noisyvalue.graph", convert = FALSE)

# The DP pattern: a true count of 87 gets a discrete-Gaussian noise mechanism
# applied (scale 4), producing an observed (published) value. `NoisyFloat.draw`
# builds the posterior over the *true* value given that observation -- this
# is the piece that actually matters for the differential-privacy use case.
# (Constructors like `NoisyInt$discrete_gaussian(obs=...)` build a bare noise
# node instead and ignore `obs` when resampling -- not what we want here.)
noise_mechanism <- graph$DiscreteGaussianNode$create(loc = 0L, scale = 4.0)
noisy_count <- core$NoisyFloat$draw(87.0, noise_mechanism, rng = 0L)

cat("Python repr:", py_str(noisy_count), "\n")
cat("Observed (published) value:", py_to_r(noisy_count$`__float__`()), "\n")

# Sample the posterior over the true value and pull draws back as a plain R
# numeric vector. `sample()` returns a Python SampleBatch wrapping a 1-D
# numpy array in `$draws`, which reticulate auto-converts.
batch <- noisy_count$sample(2000L)
draws <- py_to_r(batch$draws)

cat("Posterior mean (of the true value):", mean(draws), "\n")
cat("Posterior sd:", sd(draws), "\n")

interval <- py_to_r(noisy_count$credible_interval(0.95))
cat("95% credible interval for the true value:", interval[1], "-", interval[2], "\n")
