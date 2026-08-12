"""A general framework for datasets of noisy measurements.

A dataset extension otherwise stacks four layers with no seam between them: a
codebook, physical file access, an interpretation of the release's own
constraint rows, and the construction of posterior node graphs.  Only the
middle two are specific to any one release.  This package owns the other two,
plus the reconciliation between them, so a second dataset declares a schema
and a reader instead of forking a thousand lines.

The layers, and where each lives:

- `partition` -- the one structural type.  Axis binnings, geographic levels,
  constraint coarsenings and independence blocks are all partitions of a
  finite atom set, and the operation that reconciles any two of them is the
  join.
- `catalog` -- metadata with no data attached: `Axis`, `Universe`,
  `Measurement`, `GeoLevel`, and the `Product` that ties them to a `Source`.
- `evidence` -- what a release knows besides its measurements, in a
  dataset-agnostic vocabulary: structural zeros, exact totals of a coarsening,
  bounded totals of a coarsening.
- `solve` -- the noise family, the block ladder, and the node constructions.
- `view` -- a set of areas whose latent cells are resolved once and shared, so
  two reads of one unknown stop disagreeing.

A dataset extension supplies: the schema (data, not code), a `Source` that
locates and normalizes files, a translation of its native constraint rows into
the evidence vocabulary, an atom map for any geography whose partition is
incomparable with the spine, and the name of its noise family.  See
`census.py` for the 2020 Census products built this way.
"""

from .catalog import Axis, Binning, GeoLevel, GeographyUnavailable
from .catalog import Measurement, MeasurementRow, Product, Source, Universe
from .evidence import BoundedHistogram, Evidence, ExactHistogram, Region
from .evidence import ZeroRegion, marginal_coarsening
from .partition import JoinBlock, Partition
from .solve import BlockStrategy, BoundOnly, DiscreteGaussianFamily
from .solve import DiscreteLaplaceFamily, NoiseFamily
from .solve import exact_count, solve
from .view import View

__all__ = [x.__name__ for x in [
    # structure
    Partition, JoinBlock,
    # catalog
    Axis, Binning, Universe, Measurement, GeoLevel, Product,
    Source, MeasurementRow, GeographyUnavailable,
    # evidence
    Evidence, Region, ZeroRegion, ExactHistogram, BoundedHistogram,
    marginal_coarsening,
    # solving
    NoiseFamily, DiscreteGaussianFamily, DiscreteLaplaceFamily,
    BlockStrategy, BoundOnly, solve, exact_count,
    # views
    View]]
