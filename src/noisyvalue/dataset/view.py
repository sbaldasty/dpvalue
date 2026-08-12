"""A view: a set of areas whose latent cells are resolved once and shared.

The contract, which is the whole point of the type:

**Two measurements read from the same view are jointly consistent wherever
they name the same latent cells; two views are independent.**

That is honest about both halves.  A view is stateful and holds memory -- it
memoizes a node per cell it has resolved -- and in exchange, asking for the
same measurement twice returns the *same node objects*, so sampling the two
together cannot disagree.  Without a view, `Name.fresh()` hands out a new
symbol every time and two reads of one physical unknown are two independent
posteriors.

What a view deliberately does *not* promise in this version: reconciling
overlapping measurements of the same cells.  `sex*hispanic` and `detailed`
measure the same latent histogram, and combining them properly is the DAS
post-processing problem, a large constrained optimization.  Instead each cell
is resolved from the declared exact evidence plus one chosen measurement --
the first that names it -- and everything that implies was discarded is
reported by `unused_evidence()` rather than dropped quietly.
"""

import math
import warnings

import numpy as np
import pandas as pd

from ..pandas import NoisyFloatArray, NoisyIntArray
from .solve import DiscreteGaussianFamily, build_values, solve

MATERIALIZE_WARN_CELLS = 2_000_000


class View:
    def __init__(self, product, geography, *, nonnegative=True,
                 apply_constraints=True, family=None, strategy=None,
                 **selection):
        self.product = product
        self.plan = product.plan(geography)
        self.selection = dict(selection)
        self.nonnegative = nonnegative
        self.apply_constraints = apply_constraints
        self.family = family or DiscreteGaussianFamily()
        self.strategy = strategy

        product.source.check_selection(self.plan.read_level, self.selection)

        self._keys = _CellKeys(product.universe)
        self._cells = {}          # geo key -> {cell key: (obs, root)}
        self._rows = {}           # measurement name -> list of MeasurementRow
        self._frames = {}         # measurement name -> built DataFrame
        self._evidence = None
        self._notes = []
        self._overlaps = 0

    def __repr__(self):
        return (f"<View of {self.product.name} at {self.plan.level.name!r}: "
                f"{len(self._frames)} measurements, {self.cells:,} cells held>")

    @property
    def cells(self):
        """Latent cells this view currently holds a node for."""
        return sum(len(cells) for cells in self._cells.values())

    # ── reading ─────────────────────────────────────────────────────────────

    def query(self, name):
        """Resolve one measurement over the view's areas as a tidy frame."""
        return self.queries([name])

    def queries(self, names):
        """Resolve several measurements in one pass over the source."""
        if isinstance(names, str):
            names = [names]
        measurements = [self.product.measurement(n) for n in names]

        missing = [m for m in measurements if m.name not in self._frames]
        if missing:
            self._fetch(missing)
            for m in missing:
                self._frames[m.name] = self._resolve(m)

        frames = [self._frames[m.name] for m in measurements]
        out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        return out.copy()

    def _fetch(self, measurements):
        wanted = [m for m in measurements
                  if m.native is not None and m.name not in self._rows]
        if wanted:
            fetched = {m.name: [] for m in wanted}
            for row in self.product.source.read_measurements(
                    self.plan.read_level, self.selection, wanted):
                fetched[row.measurement].append(row)
            self._rows.update(fetched)
        for m in measurements:
            self._rows.setdefault(m.name, [])
        if self.apply_constraints and self._evidence is None:
            self._evidence = self.product.source.read_evidence(
                self.plan.read_level, self.selection)
        elif self._evidence is None:
            self._evidence = {}

    # ── resolution ──────────────────────────────────────────────────────────

    def _resolve(self, measurement):
        rows = self._rows[measurement.name]
        geos = [row.geo for row in rows]
        if not rows:
            # No released query here.  Either the product never had one (the
            # housing-unit total is invariant, so it was never noised) or the
            # level does not carry it (the person total is a DP query
            # everywhere but nationally, where it is an invariant instead).
            # Both are the same thing to the solver: resolve from evidence
            # alone, and complain if that leaves a cell free.
            geos = sorted(self._evidence)
            rows = [None] * len(geos)
            if not geos:
                warnings.warn(
                    f"{measurement.name!r} has no released query at "
                    f"{self.plan.read_level!r} and no evidence to serve it "
                    "from; returning no rows."
                    + ("" if self.apply_constraints else
                       " Evidence is off (apply_constraints=False)."),
                    stacklevel=4)

        total_cells = len(geos) * measurement.cells
        if total_cells > MATERIALIZE_WARN_CELLS:
            warnings.warn(
                f"materializing {total_cells:,} noisy values; consider "
                "narrowing the geography, selection, or measurement list.",
                stacklevel=4)

        keys = self._keys.for_measurement(measurement)
        obs_blocks, root_blocks = [], []
        for geo, row in zip(geos, rows):
            obs, roots = self._resolve_area(measurement, geo, row, keys)
            obs_blocks.append(obs)
            root_blocks.append(roots)

        return self._frame(measurement, geos, rows, obs_blocks, root_blocks)

    def _resolve_area(self, measurement, geo, row, keys):
        owned = self._cells.setdefault(geo, {})
        held = [owned.get(int(k)) for k in keys]
        if all(cell is not None for cell in held):
            # every cell of this measurement is already resolved in this view
            return (np.array([c[0] for c in held], dtype="int64"),
                    np.array([c[1] for c in held], dtype=object))

        values = None if row is None else row.values
        variance = 0.0 if row is None else row.variance
        evidence = self._evidence.get(geo, ()) if self.apply_constraints else ()

        resolution = solve(
            measurement, values, variance, evidence,
            nonnegative=self.nonnegative, family=self.family,
            strategy=self.strategy)
        obs, roots = build_values(resolution, values, variance, self.family)

        for note in resolution.notes:
            self._notes.append({"geo": geo, "measurement": measurement.name, **note})
        for position, key in enumerate(keys):
            key = int(key)
            if key in owned:
                self._overlaps += 1
            else:
                owned[key] = (obs[position], roots[position])
        return obs, roots

    # ── frame assembly ──────────────────────────────────────────────────────

    def _frame(self, measurement, geos, rows, obs_blocks, root_blocks):
        axes = self.product.universe.axis_names
        n_cells = measurement.cells
        labels = _label_columns(measurement)

        obs = (np.concatenate(obs_blocks) if obs_blocks
               else np.empty(0, dtype="int64"))
        roots = (np.concatenate(root_blocks) if root_blocks
                 else np.empty(0, dtype=object))

        geo_column = np.repeat(np.asarray(geos, dtype=object), n_cells)
        frame = pd.DataFrame({
            **self.product.source.geo_columns(
                self.plan.read_level, self.selection, geo_column),
            "measurement": measurement.name,
            **{axis: np.tile(labels[a], len(geos)) for a, axis in enumerate(axes)},
            "value": NoisyIntArray(obs, roots),
            "variance": np.repeat(
                np.array([0.0 if r is None else r.variance for r in rows],
                         dtype="float64"),
                n_cells) if rows else np.empty(0, dtype="float64"),
        })
        if self.plan.aggregate:
            frame = self._aggregate(frame, axes, geo_column, n_cells)
        return frame

    def _aggregate(self, frame, axes, geo_column, n_cells):
        """Sum finer-level rows up to the requested partition.

        This is the meet of two incomparable geographic partitions, computed
        the only way it can be: descend to the atoms the source does measure
        and re-aggregate.  Summed with plain `+` rather than the extension
        array's own reduction, since pandas' groupby machinery collapses that
        to a constant and would silently discard every contributor's noise.

        The summed rows no longer correspond to one source area, so `geoid`
        becomes the aggregate key, `geocode` is dropped, and any flag that
        disagrees among the contributors (a block group straddling the AIAN
        branch, say) is left missing rather than picked arbitrarily.
        """
        frame = frame.copy()
        frame["geoid"] = [self.plan.key(g) for g in geo_column]
        flags = [c for c in ("aian",) if c in frame.columns]
        group_cols = ["geoid", "measurement", *axes]

        records = []
        for key, group in frame.groupby(group_cols, sort=False):
            values = list(group["value"])
            total = values[0]
            for value in values[1:]:
                total = total + value
            record = dict(zip(group_cols, _as_tuple(key)))
            for column in flags:
                distinct = pd.unique(group[column].dropna())
                record[column] = distinct[0] if len(distinct) == 1 else pd.NA
            record["variance"] = float(group["variance"].sum())
            record["value"] = total
            records.append(record)

        out = pd.DataFrame.from_records(records)
        out["geoid"] = out["geoid"].astype("string")
        for column in flags:
            out[column] = pd.array(out[column], dtype="boolean")
        out["geocode"] = pd.array([pd.NA] * len(out), dtype="string")
        out["value"] = _as_float_array(out["value"])
        return out[["geoid", "geocode", *flags, "measurement", *axes,
                    "value", "variance"]]

    # ── derived values ──────────────────────────────────────────────────────

    def rollup(self, name, **binnings):
        """Coarsen a resolved measurement by summing the cells it already holds.

        Unlike two separately measured queries, this *is* jointly consistent
        with its source: the coarse cells are derived nodes over the very
        nodes the fine measurement resolved.  Each target binning must be a
        coarsening of the one the measurement uses -- core checks that against
        the refinement lattice rather than trusting the caller.
        """
        measurement = self.product.measurement(name)
        universe = self.product.universe
        targets = []
        for axis, spec in zip(universe.axes, measurement.specs):
            target = binnings.get(axis.name, spec)
            if not axis.refines(spec, target):
                raise ValueError(
                    f"axis {axis.name!r}: {spec!r} does not refine {target!r}, "
                    "so the measurement cannot be summed up to it")
            targets.append(axis.binning(target))

        coarse = self.query(measurement.name)
        for a, (spec, target) in enumerate(zip(measurement.specs, targets)):
            axis = universe.axes[a]
            source = axis.binning(spec)
            relabel = {
                source.labels[position]: target.labels[target.partition.block_of(group[0])]
                for position, group in enumerate(source.partition.groups)}
            coarse[axis.name] = coarse[axis.name].map(relabel)

        group_cols = [c for c in coarse.columns
                      if c not in ("value", "variance", "measurement")]
        records = []
        for key, group in coarse.groupby(group_cols, sort=False, dropna=False):
            values = list(group["value"])
            total = values[0]
            for value in values[1:]:
                total = total + value
            records.append({
                **dict(zip(group_cols, _as_tuple(key))),
                "measurement": f"{measurement.name} rolled up",
                "variance": float(group["variance"].sum()),
                "value": total,
            })
        out = pd.DataFrame.from_records(records)
        out["value"] = _as_float_array(out["value"])
        return out[[*group_cols, "measurement", "value", "variance"]]

    # ── reporting ───────────────────────────────────────────────────────────

    def unused_evidence(self):
        """Everything the view knew but could not impose at full strength.

        One row per (measurement, evidence source, kind), with the number of
        cells affected.  An empty frame means every constraint that bears on
        what you asked for was used exactly.
        """
        rows = list(self._notes)
        if self._overlaps:
            rows.append({
                "geo": None, "measurement": None,
                "kind": "overlapping measurements not reconciled",
                "source": "", "cells": self._overlaps,
                "detail": "cells named by more than one measurement; each "
                          "measurement kept its own posterior",
            })
        if not rows:
            return pd.DataFrame(
                columns=["measurement", "kind", "source", "areas", "cells", "detail"])
        frame = pd.DataFrame(rows)
        grouped = frame.groupby(
            ["measurement", "kind", "source", "detail"], dropna=False, sort=False)
        out = grouped.agg(areas=("geo", "nunique"), cells=("cells", "sum"))
        out = out.reset_index()
        return out[["measurement", "kind", "source", "areas", "cells", "detail"]]


def _as_tuple(key):
    """pandas hands back a bare scalar when grouping on one column."""
    return key if isinstance(key, tuple) else (key,)


def _as_float_array(values):
    """Pack a column of summed NoisyFloats back into an extension array."""
    return NoisyFloatArray(
        np.array([float(v._obs) for v in values], dtype="float64"),
        np.array([v._root for v in values], dtype=object))


def _label_columns(measurement):
    """Per-axis label arrays laid out in the measurement's cell order."""
    shape = measurement.shape
    columns = []
    for a, labels in enumerate(measurement.labels):
        reps = math.prod(shape[a + 1:])
        tiles = math.prod(shape[:a])
        columns.append(np.tile(np.repeat(np.asarray(labels, dtype=object), reps), tiles))
    return columns


class _CellKeys:
    """Compact, stable identity for a latent cell of a universe.

    A cell is the set of base cells a measurement position covers, so its
    identity is the tuple of per-axis index groups.  Those groups are interned
    per axis once from the catalog -- there are only ever a few hundred across
    every binning an axis declares -- and packed into one integer, which keeps
    the memo a dict of ints rather than a dict of nested frozensets.

    This is also the seam for the vectorized node type the design calls for:
    keys are computed for a whole measurement at once, so a block of cells can
    become one array-backed node without changing anything above.
    """

    def __init__(self, universe):
        self._ids = []
        self._strides = []
        stride = 1
        for axis in universe.axes:
            table = {}
            for name in axis.binnings:
                for group in axis.binning(name).partition.groups:
                    table.setdefault(group, len(table))
            self._ids.append(table)
            self._strides.append(stride)
            stride *= max(len(table), 1)
        self._size = stride

    def for_measurement(self, measurement):
        """Cell keys for every position of a measurement, in cell order."""
        shape = measurement.shape
        key = np.zeros(shape, dtype=np.int64)
        for a, binning in enumerate(measurement.binnings):
            ids = np.array([self._ids[a][g] for g in binning.partition.groups],
                           dtype=np.int64) * self._strides[a]
            key += ids.reshape((-1,) + (1,) * (len(shape) - a - 1))
        return key.reshape(-1)
