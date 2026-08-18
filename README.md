# Noisy values library for Python

This library supports

- The encapsulation of a nosiy observation and a symbolic representation of the posterior of its true value into a single object via `NoisyFloat`, `NoisyInt`, and `NoisyBool`
- Composing such noisy values with standard arithmetic operations, comparisons, boolean connectives, allowing the observed result and the posterior to propagate in tandem
- Custom operations such as `noisy_min` and `noisy_max`
- Noise sources that _depend on_ other noise sources; for instance `NoisyContingencyTable` can model **sampling uncertainty** over **differentially private** counts
- Sampling of the symbolic posteriors to estimate credible intervals, etc.
- Visualization of posteriors
- A custom file format so that tuples of noisy values and shaped arrays of noisy values can be written to persistent storage and shared
- Noisy types in `pandas` data frames
- An R compatibility layer via `reticulate`
- An API to download and use differentially private Census data as noisy values in Python, plus a `tidycensus`-like API for working with noisy Census data in R
- An API to download and use differentially private Wikimedia page view data as noisy values in Python

The roadmap for future development includes

- Better tooling for creating differentially private data releases
- Jupyter notebook tutorials
