"""Turn a measurement plus evidence into posteriors, one area at a time.

Two things live here.  `NoiseFamily` names how a release's noise works and
owns the closed forms that go with it -- a discrete Gaussian measurement has a
conditional pair distribution that a Laplace one would not, so that formula
belongs to the family, not to the solver.  `solve` does the reconciliation:
it walks the evidence, decides for every cell whether it is pinned, bounded,
or tied to a neighbour, and reports what it had to weaken.

The ladder, for a block of cells whose total the evidence pins exactly:

| free cells | construction |
| --- | --- |
| 0 | nothing to do |
| total is 0 | all pinned to zero |
| 1 | point mass at the block total |
| 2 | the family's conditional pair -- closed form, stays anti-correlated |
| 3 or more | `strategy`, which by default keeps the total as an upper bound |

The last row is the one worth replacing.  The exact conditional there is a
discrete Gaussian restricted to a lattice hyperplane, which does not factor
into per-cell distributions the node graph can sample independently; a lattice
sampler or MCMC dropped in as a `BlockStrategy` would improve every dataset
that declares this family at once.  Until then the weakening is a named,
reported decision (`View.unused_evidence`) rather than a silent one.
"""

import math

import numpy as np
import sympy as sp

from ..core import NoisyInt
from ..graph import DerivedNode
from ..graph import DiscreteGaussianNode, DiscreteLaplaceNode
from .evidence import BoundedHistogram, ExactHistogram, ZeroRegion


# ── node constructions ───────────────────────────────────────────────────────

def exact_count(pinned):
    """A `NoisyInt` with no uncertainty left: evidence pins the true count.

    The measurement it replaces is dropped rather than kept as the observed
    value, since the posterior is a point mass and the noisy number is known
    to be wrong.
    """
    return NoisyInt(int(pinned), DerivedNode(sp.Integer(int(pinned))))


class NoiseFamily:
    """How a release noised its measurements, and what that buys."""

    name = "unknown"

    def cell(self, obs, variance, lo=None, hi=None):
        """Posterior for one measured cell, truncated to `[lo, hi]`."""
        raise NotImplementedError

    def pair(self, obs_a, obs_b, variance, total, nonnegative):
        """Two cells conditioned on their exact sum, or None if no closed form."""
        return None


class LatticeFamily(NoiseFamily):
    """Shared cell construction for noise living on the integer lattice.

    Under a flat prior a cell's posterior is the noise law re-centred at the
    observation, so it is encoded directly as `obs - eps` rather than through
    a latent-plus-constraint pair; building and sampling large collections of
    cells then avoids the symbolic solve step entirely.  A concrete family
    names its node class, how the release's variance maps to the node's scale
    parameter, and how far outside the centre a bound is provably inert.
    """

    node_cls = None

    # A bound this far outside the posterior's centre is not merely unlikely
    # to bind, it is *provably* inert: the node's `_draw` only ever puts
    # support on a grid of `grid_width(scale)` either side, so a bound beyond
    # that leaves the sampled distribution bit-for-bit unchanged. Dropping it
    # keeps the node's params free of symbols, which skips the (small) bound
    # substitution cost at sample time.  This matters because evidence is
    # applied generically -- a query that marginalizes an axis away picks up
    # the sum of every category bound on it, a real but useless number a
    # per-dataset rule would simply not have bothered to compute.
    INERT_SCALES = None
    INERT_FLOOR = 60.0

    def scale_from_variance(self, variance):
        """The node's scale parameter for a noise law of this variance."""
        raise NotImplementedError

    def cell(self, obs, variance, lo=None, hi=None):
        obs = int(obs)
        scale = self.scale_from_variance(float(variance))
        if lo is not None and hi is not None and lo == hi:
            return exact_count(lo)
        lo, hi = self._drop_inert(obs, scale, lo, hi)
        # theta = obs - eps lies in [lo, hi] iff eps lies in [obs-hi, obs-lo].
        low = -sp.oo if hi is None else sp.Integer(obs - int(hi))
        high = sp.oo if lo is None else sp.Integer(obs - int(lo))
        node = self.node_cls.create(
            loc=sp.Integer(0), scale=sp.Float(scale), low=low, high=high)
        return NoisyInt(obs, DerivedNode(sp.Integer(obs) - node.expr, deps=(node,)))

    def _drop_inert(self, obs, scale, lo, hi):
        """Discard bounds that lie outside the sampled support entirely."""
        centre = obs
        if lo is not None:
            centre = max(centre, lo)
        if hi is not None:
            centre = min(centre, hi)
        slack = max(self.INERT_FLOOR, self.INERT_SCALES * scale)
        if lo is not None and lo < centre - slack:
            lo = None
        if hi is not None and hi > centre + slack:
            hi = None
        return lo, hi


class DiscreteGaussianFamily(LatticeFamily):
    """The 2020 DAS mechanism: each query noised independently, discrete Gaussian."""

    name = "discrete gaussian"
    node_cls = DiscreteGaussianNode
    # the node's grid spans 6 sd either side, so 8 is safely beyond it
    INERT_SCALES = 8.0

    def scale_from_variance(self, variance):
        return math.sqrt(float(variance))

    def pair(self, obs_a, obs_b, variance, total, nonnegative):
        """Closed form for two cells with a known exact sum.

        With independent measurements obs_a = a + e1 and obs_b = b + e2 (equal
        variance v) and a known exact a + b = total, the posterior of a over
        the integers is a discrete Gaussian centred at (obs_a - obs_b + total)/2
        with variance v/2, and b = total - a exactly (perfect negative
        correlation).  This is the only block size for which the conditional is
        again a discrete Gaussian, which is why it is the only one built jointly.
        """
        obs_a, obs_b, total = int(obs_a), int(obs_b), int(total)
        scale = sp.sqrt(sp.Float(float(variance)) / 2)
        delta = sp.Rational(obs_a + obs_b - total, 2)
        low = sp.Integer(obs_a - total) if nonnegative else -sp.oo
        high = sp.Integer(obs_a) if nonnegative else sp.oo
        node = DiscreteGaussianNode.create(loc=delta, scale=scale, low=low, high=high)
        a_root = DerivedNode(sp.Integer(obs_a) - node.expr, deps=(node,))
        b_root = DerivedNode(sp.Integer(total) - a_root.expr, deps=(a_root,))
        return NoisyInt(obs_a, a_root), NoisyInt(obs_b, b_root)


class DiscreteLaplaceFamily(LatticeFamily):
    """The geometric mechanism of pure epsilon-DP releases, e.g. Wikimedia's
    pageview data: integer noise with pmf proportional to exp(-|k|/scale),
    where scale = sensitivity / epsilon.

    As for every family, `cell` takes the noise's true sampling variance;
    `variance_from_scale` converts a release's published scale, and the two
    conversions are exact inverses.  There is no `pair` closed form: two
    geometric-noised cells conditioned on their exact sum follow a
    piecewise-geometric law that is not again a lattice location-scale
    family, so exact totals over pairs fall through to the block strategy
    like any larger block.
    """

    name = "discrete laplace"
    node_cls = DiscreteLaplaceNode
    # the node's grid spans 30 scales either side, so 32 is safely beyond it
    INERT_SCALES = 32.0

    @staticmethod
    def variance_from_scale(scale):
        """Sampling variance of two-sided geometric noise with this scale."""
        p = math.exp(-1.0 / float(scale))
        return 2.0 * p / (1.0 - p) ** 2

    def scale_from_variance(self, variance):
        # invert 2p/(1-p)^2 = v for p in (0, 1), then scale = -1/log(p)
        v = float(variance)
        p = (v + 1.0 - math.sqrt(2.0 * v + 1.0)) / v
        return -1.0 / math.log(p)


# ── strategies for blocks of three or more ───────────────────────────────────

class BlockStrategy:
    """What to do with an exact block total the closed forms cannot use."""

    name = "abstract"

    def build(self, family, obs, variance, total, lo, hi, nonnegative):
        """Return {position within the block: NoisyInt}, or None to weaken."""
        raise NotImplementedError


class BoundOnly(BlockStrategy):
    """Keep the block total as an upper bound and say so.  The default."""

    name = "bound only"

    def build(self, family, obs, variance, total, lo, hi, nonnegative):
        return None


# ── the solver ───────────────────────────────────────────────────────────────

class Resolution:
    """Per-cell plan for one measurement in one area.

    `exact` holds a pinned true value or NaN, `lo`/`hi` the interval a free
    cell is truncated to, and `prebuilt` the cells a joint construction has
    already produced.  `notes` records every place the evidence was not used
    at full strength.
    """

    def __init__(self, shape, nonnegative):
        self.shape = tuple(shape)
        self.nonnegative = nonnegative
        self.exact = np.full(self.shape, np.nan)
        self.lo = np.full(self.shape, 0.0 if nonnegative else -np.inf)
        self.hi = np.full(self.shape, np.inf)
        self.prebuilt = {}
        self.notes = []

    def note(self, kind, source, cells, detail=""):
        self.notes.append({
            "kind": kind, "source": source, "cells": int(cells), "detail": detail})


def solve(measurement, values, variance, evidence=(), *, nonnegative=True,
          family=None, strategy=None):
    """Reconcile one measured array with the evidence bearing on it."""
    family = family or DiscreteGaussianFamily()
    strategy = strategy or BoundOnly()
    binnings = measurement.binnings
    shape = measurement.shape

    resolution = Resolution(shape, nonnegative)
    if values is not None and len(values) != math.prod(shape):
        raise ValueError(
            f"measurement {measurement.name!r}: {len(values)} cells do not "
            f"match its shape {shape}")

    for item in evidence:
        if isinstance(item, BoundedHistogram):
            _apply_bounds(resolution, item, binnings)
    for item in evidence:
        if isinstance(item, ZeroRegion):
            _apply_zeros(resolution, item, binnings)
    for item in evidence:
        if isinstance(item, ExactHistogram):
            _apply_totals(resolution, item, binnings, values, variance,
                          family, strategy)
    return resolution


def _apply_bounds(resolution, item, binnings):
    lo, hi = item.bounds(binnings)
    known_lo, known_hi = ~np.isnan(lo), ~np.isnan(hi)
    resolution.lo[known_lo] = np.maximum(resolution.lo[known_lo], lo[known_lo])
    resolution.hi[known_hi] = np.minimum(resolution.hi[known_hi], hi[known_hi])
    mute = np.count_nonzero(~known_hi & ~known_lo)
    if mute:
        resolution.note("bounds not applicable", item.describe(), mute,
                        "the coarsening does not account for every cell covered")


def _apply_zeros(resolution, item, binnings):
    mask = item.pinned(binnings)
    resolution.exact[mask] = 0.0


def _apply_totals(resolution, item, binnings, values, variance, family, strategy):
    exact = resolution.exact.reshape(-1)
    lo = resolution.lo.reshape(-1)
    hi = resolution.hi.reshape(-1)
    nonnegative = resolution.nonnegative
    tied = set(resolution.prebuilt)

    for block in item.blocks(binnings, resolution.shape):
        cells = block.cells
        pinned = ~np.isnan(exact[cells])
        free = cells[~pinned]
        if free.size == 0:
            continue
        if block.total is None:
            resolution.note("evidence unused", block.source, len(cells),
                            "the coarsening says nothing about these cells")
            continue

        remaining = block.total - int(np.nansum(exact[cells]))
        if remaining < 0:
            resolution.note("inconsistent total", block.source, len(cells),
                            "pinned cells already exceed the block total")
            continue

        if any(int(cell) in tied for cell in free):
            _weaken(resolution, block, free, hi, remaining,
                    "cells already tied by earlier evidence")
            continue
        if not block.exact:
            detail = ("the measurement covers only part of the block"
                      if block.contained else "partitions are incomparable here")
            _weaken(resolution, block, free, hi, remaining, detail)
            continue

        if remaining == 0 and nonnegative:
            exact[free] = 0.0
        elif free.size == 1:
            exact[free[0]] = remaining
        elif free.size == 2:
            first, second = (int(cell) for cell in free)
            built = family.pair(int(values[first]), int(values[second]),
                                variance, remaining, nonnegative)
            if built is None:
                _weaken(resolution, block, free, hi, remaining,
                        f"{family.name} has no closed form for a pair")
            else:
                resolution.prebuilt[first], resolution.prebuilt[second] = built
                tied.update((first, second))
        else:
            built = strategy.build(
                family, [int(values[int(c)]) for c in free], variance,
                remaining, lo[free], hi[free], nonnegative)
            if built is None:
                _weaken(resolution, block, free, hi, remaining,
                        f"block of {free.size} free cells; strategy is "
                        f"{strategy.name!r}")
            else:
                for offset, value in built.items():
                    resolution.prebuilt[int(free[offset])] = value
                    tied.add(int(free[offset]))


def _weaken(resolution, block, free, hi, remaining, detail):
    """Record a total we could not impose, keeping it as a bound if we can."""
    if resolution.nonnegative and block.contained:
        hi[free] = np.minimum(hi[free], remaining)
        kind = "exact total weakened to a bound"
    else:
        kind = "exact total unused"
    resolution.note(kind, block.source, len(free), detail)


# ── materialization ──────────────────────────────────────────────────────────

def build_values(resolution, values, variance, family=None):
    """Realize a resolution as parallel observation and node-root arrays."""
    family = family or DiscreteGaussianFamily()
    exact = resolution.exact.reshape(-1)
    lo = resolution.lo.reshape(-1)
    hi = resolution.hi.reshape(-1)
    n = exact.size

    obs_out = np.empty(n, dtype="int64")
    roots_out = np.empty(n, dtype=object)
    for flat in range(n):
        if flat in resolution.prebuilt:
            value = resolution.prebuilt[flat]
        elif not math.isnan(exact[flat]):
            value = exact_count(int(exact[flat]))
        elif values is None:
            raise ValueError(
                "this measurement has no released query, so every cell must be "
                f"pinned by evidence; cell {flat} was left free")
        else:
            value = family.cell(
                int(values[flat]), variance,
                lo=None if lo[flat] == -np.inf else int(lo[flat]),
                hi=None if hi[flat] == np.inf else int(hi[flat]))
        obs_out[flat] = value._obs
        roots_out[flat] = value._root
    return obs_out, roots_out
