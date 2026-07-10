# Demo: load a NoisyValue-backed DataFrame as a tibble and work with it using
# ordinary R/dplyr syntax, with noisy columns keeping their posterior
# provenance through arithmetic and comparison operators.
#
# Setup (from the repo root):
#   uv sync --extra dev
#   uv run python example/make_demo_frame.py
#   Rscript example/tidy_demo.R
#
# This sources the noisyvalue R package straight from r/noisyvalue via
# pkgload, so no R package installation step is required.

library(dplyr)
pkgload::load_all("r/noisyvalue", quiet = TRUE)

noisyvalue_init(venv = ".venv", src = "src")

tbl <- read_noisy_frame("example/demo_frame.json")
cat("Loaded tibble:\n")
print(tbl)

cat("\nClass of a noisy column:", paste(class(tbl$black_pop), collapse = ", "), "\n")

# Arithmetic on noisy columns: composes the symbolic posterior on the Python
# side, same as it would from Python.
tbl <- tbl |> mutate(black_share = black_pop / total_pop)
cat("\nWith a derived noisy column (black_share):\n")
print(tbl$black_share)

# Row filtering: comparing a noisy column against a number produces a plain
# logical mask over the *observed* values, so this is ordinary dplyr::filter.
cat("\nCounties where the observed black_share exceeds 0.3:\n")
print(tbl |> filter(black_share > 0.3) |> select(county))

# Aggregation: mean() of a noisy column returns a length-1 noisy value
# wrapping a real Python NoisyFloat -- comparing *that* against a number
# preserves the posterior and returns a NoisyBool, not a plain logical.
avg_black <- mean(tbl$black_pop)
cat("\nMean black_pop (a noisy scalar):\n")
print(avg_black)

is_above_150 <- avg_black > 150
cat("\nIs the mean above 150? (a NoisyBool, not a plain logical):\n")
print(is_above_150)
cat("class:", paste(class(is_above_150), collapse = ", "), "\n")
