"""The dataset-agnostic vocabulary for everything known besides a measurement.

An extension reads its native constraint rows and says only what they *mean*:

- `ZeroRegion` -- these base cells are structurally impossible.
- `ExactHistogram` -- the totals of this coarsening of the universe are known
  exactly (an invariant of the release, a published table it had to reproduce).
- `BoundedHistogram` -- the totals of this coarsening lie in `[lo, hi]`.

What follows from that -- which cells are pinned, which are tied to each other,
which merely bounded -- is core's problem, in `solve.py`.

Everything here is stated over the universe's *base* cells and translated onto
a measurement's cells on demand, because the base histogram is far too large to
enumerate (DHC is 1,227,744 cells per area).  The translation is what makes
that tractable: both a region and a coarsening are products over axes, so
containment and overlap factor axis by axis and never touch a base cell.
"""

import itertools

import numpy as np

from .partition import Partition


def _flat_indices(positions, shape):
    grids = np.meshgrid(*(np.asarray(p, dtype=np.intp) for p in positions),
                        indexing="ij")
    return np.ravel_multi_index(tuple(g.ravel() for g in grids), shape)


def _contract(values, matrices):
    """Sum `values` over each axis against a selection matrix.

    `matrices[a]` has shape (query positions, coarse positions) and marks
    which coarse blocks each query position draws from, so the result is the
    query-shaped array of per-cell sums.
    """
    out = np.asarray(values)
    for axis, matrix in enumerate(matrices):
        out = np.tensordot(matrix, out, axes=([1], [axis]))
        out = np.moveaxis(out, 0, axis)
    return out


class Region:
    """A rectangular subset of a universe's base cells.

    One selector per axis: a set of base level indices, or None for "every
    level of this axis".
    """

    def __init__(self, selectors, source=""):
        self.selectors = tuple(
            None if s is None else frozenset(int(x) for x in s) for s in selectors)
        self.source = source

    def __repr__(self):
        shown = ",".join("*" if s is None else f"{len(s)}" for s in self.selectors)
        return f"<Region {shown}>"

    def covers_axis(self, axis_index, atoms):
        selector = self.selectors[axis_index]
        return selector is None or frozenset(atoms) <= selector


class Evidence:
    """Base class; `source` names the native row this came from, for reports."""

    def __init__(self, source=""):
        self.source = source

    def describe(self):
        return self.source or type(self).__name__


class ZeroRegion(Evidence):
    """Every base cell in `region` is exactly zero."""

    def __init__(self, region, source=""):
        super().__init__(source or region.source)
        self.region = region

    def __repr__(self):
        return f"<ZeroRegion {self.source!r} {self.region}>"

    def pinned(self, binnings):
        """Boolean array over a measurement's cells: which are wholly inside.

        A measurement cell is the sum of the base cells in its region, so it
        is pinned to zero exactly when every base cell it covers is.
        """
        per_axis = [
            np.array([self.region.covers_axis(a, group)
                      for group in binning.partition.groups], dtype=bool)
            for a, binning in enumerate(binnings)
        ]
        mask = per_axis[0]
        for vector in per_axis[1:]:
            mask = mask[..., None] & vector
        return mask


class _Coarsened(Evidence):
    """Shared machinery for evidence stated over a coarsening of the universe.

    `coarsening` is one `Partition` per axis, over that axis's base levels; an
    axis the evidence does not resolve gets the trivial one-block partition,
    and the corresponding value axis has length 1.
    """

    def __init__(self, coarsening, source=""):
        super().__init__(source)
        self.coarsening = tuple(coarsening)

    @property
    def coarse_shape(self):
        return tuple(len(p) for p in self.coarsening)

    def _check(self, values, what):
        values = np.asarray(values)
        if values.shape != self.coarse_shape:
            raise ValueError(
                f"{self.source or type(self).__name__}: {what} has shape "
                f"{values.shape}, expected {self.coarse_shape}")
        return values

    def _joins(self, binnings):
        if len(binnings) != len(self.coarsening):
            raise ValueError("measurement and evidence disagree on axis count")
        return [b.partition.join(c) for b, c in zip(binnings, self.coarsening)]


class ExactHistogram(_Coarsened):
    """Totals of a coarsening of the universe, known exactly.

    Against a measurement, this splits the measurement's cells into blocks
    whose total is pinned: the join of the measurement's binning with this
    coarsening, taken axis by axis and then crossed.  A block only pins a
    total when both sides cover the same base cells in it -- otherwise the
    measurement is asking about a sub-slice, and (for nonnegative quantities)
    the block total survives as an upper bound instead.
    """

    def __init__(self, coarsening, values, source=""):
        super().__init__(coarsening, source)
        self.values = self._check(values, "values").astype(np.int64)

    def __repr__(self):
        return f"<ExactHistogram {self.source!r} {self.coarse_shape}>"

    def blocks(self, binnings, shape):
        """Yield `Block`s of a measurement's cells with a known block total."""
        joins = self._joins(binnings)
        for combination in itertools.product(*joins):
            positions = tuple(block.a_positions for block in combination)
            if any(len(p) == 0 for p in positions):
                continue                      # nothing of the measurement here
            selector = tuple(block.b_positions for block in combination)
            speaks = all(len(s) > 0 for s in selector)
            yield Block(
                cells=_flat_indices(positions, shape),
                total=int(self.values[np.ix_(*selector)].sum()) if speaks else None,
                exact=speaks and all(block.exact for block in combination),
                contained=speaks and all(block.contained for block in combination),
                source=self.source)


class BoundedHistogram(_Coarsened):
    """Totals of a coarsening of the universe, known to lie in `[lo, hi]`.

    The two directions transfer differently, and the asymmetry is not a
    convention -- it is what nonnegativity gives you.  A measurement cell
    sitting inside a coarse block is at most that block's upper bound, so
    upper bounds always transfer.  It is at least the lower bounds only of
    the blocks it wholly contains, so a lower bound reaches a cell only when
    the cell is the whole category.
    """

    def __init__(self, coarsening, lo=None, hi=None, source=""):
        super().__init__(coarsening, source)
        self.lo = None if lo is None else self._check(lo, "lo").astype(np.int64)
        self.hi = None if hi is None else self._check(hi, "hi").astype(np.int64)

    def __repr__(self):
        return f"<BoundedHistogram {self.source!r} {self.coarse_shape}>"

    def bounds(self, binnings):
        """Per-cell (lo, hi) arrays over a measurement's cells; NaN where mute."""
        touching, contained, complete = [], [], []
        for axis, binning in enumerate(binnings):
            coarse = self.coarsening[axis]
            groups = binning.partition.groups
            touching.append(np.array(
                [[bool(set(g) & set(c)) for c in coarse.groups] for g in groups],
                dtype=np.int64))
            contained.append(np.array(
                [[set(c) <= set(g) for c in coarse.groups] for g in groups],
                dtype=np.int64))
            complete.append(np.array(
                [frozenset(g) <= coarse.covered for g in groups], dtype=bool))

        # an upper bound is only sound if the coarsening accounts for every
        # base cell the measurement cell covers
        valid = complete[0]
        for vector in complete[1:]:
            valid = valid[..., None] & vector

        shape = tuple(len(b) for b in binnings)
        lo = np.full(shape, np.nan)
        hi = np.full(shape, np.nan)
        if self.lo is not None:
            # A cell containing no whole coarse block gets an empty sum, which
            # is "nothing is known" and not "at least zero" -- conflating the
            # two would quietly impose nonnegativity the caller may have
            # declined.
            reached = _contract(np.ones(self.coarse_shape, dtype=np.int64), contained)
            lo = np.where(reached > 0, _contract(self.lo, contained), np.nan)
        if self.hi is not None:
            hi = np.where(valid, _contract(self.hi, touching).astype(float), np.nan)
        return lo, hi


class Block:
    """A set of measurement cells whose total the evidence speaks to."""

    def __init__(self, cells, total, exact, contained, source=""):
        self.cells = cells
        self.total = total
        self.exact = exact
        self.contained = contained
        self.source = source

    def __repr__(self):
        kind = "exact" if self.exact else ("bound" if self.contained else "mute")
        return f"<Block {len(self.cells)} cells, {kind} total {self.total}>"


def marginal_coarsening(universe, resolved):
    """Coarsening partitions for a universe, given only the axes it resolves.

    `resolved` maps axis name to either a `Partition` or a per-atom label
    sequence; every axis left out is fully marginalized.  This is how an
    extension declares a constraint's shape without spelling out the axes the
    constraint says nothing about.
    """
    out = []
    for axis in universe.axes:
        spec = resolved.get(axis.name)
        if spec is None:
            out.append(Partition.trivial(axis.size, f"{axis.name}:*"))
        elif isinstance(spec, Partition):
            out.append(spec)
        else:
            out.append(Partition.from_labels(spec, axis.size, axis.name))
    return tuple(out)
