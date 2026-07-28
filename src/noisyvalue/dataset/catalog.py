"""Catalog types: metadata with no data attached.

A catalog answers "what variables exist, what values can they take, what was
measured, and over which geographies" without touching a byte of the release.
It is declarative -- a dataset extension writes `Axis`/`Binning`/`Measurement`
literals, and this module validates them (groups disjoint, labels parallel to
groups, every measurement naming binnings that exist).

`Product` ties a catalog to a `Source` (the extension's physical access) and
an `EvidenceReader` (the extension's translation of native constraint rows
into the vocabulary of `evidence.py`).  `Product.view(...)` is the entry point
callers actually use.
"""

import math

import pandas as pd

from .partition import Partition


class Binning:
    """A named partition of an axis's base levels, plus a label per group."""

    def __init__(self, name, partition, labels):
        labels = tuple(labels)
        if len(labels) != len(partition):
            raise ValueError(
                f"binning {name!r}: {len(labels)} labels for "
                f"{len(partition)} groups")
        self.name = name
        self.partition = partition
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        kind = "" if self.partition.is_cover else " (partial cover)"
        return f"<Binning {self.name!r}: {len(self.labels)} levels{kind}>"

    @property
    def is_cover(self):
        return self.partition.is_cover


class Axis:
    """A histogram axis: base levels plus every binning declared over them."""

    def __init__(self, name, levels, binnings=()):
        self.name = name
        self.levels = tuple(levels)
        self._binnings = {}

        size = len(self.levels)
        self.add(Binning("*", Partition.trivial(size, f"{name}:*"), ("total",)))
        self.add(Binning(
            "detailed", Partition.discrete(size, f"{name}:detailed"), self.levels))
        for binning in binnings:
            self.add(binning)

    def __repr__(self):
        return f"<Axis {self.name!r}: {len(self.levels)} levels, {len(self._binnings)} binnings>"

    @property
    def size(self):
        return len(self.levels)

    @property
    def binnings(self):
        return tuple(self._binnings)

    def add(self, binning):
        if binning.partition.size != self.size:
            raise ValueError(
                f"axis {self.name!r}: binning {binning.name!r} is over "
                f"{binning.partition.size} atoms, not {self.size}")
        self._binnings[binning.name] = binning
        return binning

    def declare(self, name, groups, labels):
        """Declare a binning from raw index groups."""
        partition = Partition(groups, self.size, f"{self.name}:{name}")
        return self.add(Binning(name, partition, labels))

    def alias(self, name, existing):
        """Declare `name` as another spelling of an existing binning."""
        source = self.binning(existing)
        return self.add(Binning(name, source.partition, source.labels))

    def binning(self, name):
        if name not in self._binnings:
            raise KeyError(
                f"axis {self.name!r} has no binning {name!r}; "
                f"expected one of {sorted(self._binnings)}")
        return self._binnings[name]

    def has(self, name):
        return name in self._binnings

    def refines(self, finer, coarser):
        return self.binning(finer).partition.refines(self.binning(coarser).partition)


class Universe:
    """The latent histogram a product measures: a named product of axes."""

    def __init__(self, name, axes):
        self.name = name
        self.axes = tuple(axes)
        self._by_name = {axis.name: axis for axis in self.axes}
        if len(self._by_name) != len(self.axes):
            raise ValueError(f"universe {name!r}: duplicate axis names")

    def __repr__(self):
        return f"<Universe {self.name!r}: {' x '.join(self.axis_names)} = {self.cells:,} cells>"

    @property
    def axis_names(self):
        return tuple(axis.name for axis in self.axes)

    @property
    def shape(self):
        return tuple(axis.size for axis in self.axes)

    @property
    def cells(self):
        return math.prod(self.shape)

    def axis(self, name):
        if name not in self._by_name:
            raise KeyError(
                f"universe {self.name!r} has no axis {name!r}; "
                f"expected one of {self.axis_names}")
        return self._by_name[name]

    def index(self, name):
        return self.axis_names.index(name)

    def binnings(self, specs):
        """The `Binning` each axis uses, given one spec string per axis."""
        return tuple(axis.binning(spec) for axis, spec in zip(self.axes, specs))


class Measurement:
    """A released query: which binning each axis uses, and its native name."""

    def __init__(self, name, universe, specs, native=None, note=""):
        self.name = name
        self.universe = universe
        self.specs = tuple(specs)
        self.native = native
        self.note = note
        if len(self.specs) != len(universe.axes):
            raise ValueError(
                f"measurement {name!r}: {len(self.specs)} specs for "
                f"{len(universe.axes)} axes")
        for axis, spec in zip(universe.axes, self.specs):
            axis.binning(spec)          # validates

    def __repr__(self):
        return f"<Measurement {self.name!r}: {self.cells:,} cells>"

    @classmethod
    def parse(cls, name, universe, native=None, note=""):
        """Derive per-axis specs from a canonical name like `sex*age_38_groups`.

        A canonical name lists the binning each resolved axis uses, joined by
        "*", in universe axis order; every axis it does not name is
        marginalized.  `detailed` resolves every axis.
        """
        if name == "detailed":
            specs = tuple("detailed" for _ in universe.axes)
        else:
            parts = name.split("*")
            specs = tuple(
                next((p for p in parts if axis.has(p) and p != "*"), "*")
                for axis in universe.axes)
            unused = [p for p in parts if not any(
                p == s for s in specs) and p != "*"]
            if unused:
                raise ValueError(
                    f"measurement {name!r}: no axis has binning(s) {unused}")
        return cls(name, universe, specs, native, note)

    @property
    def binnings(self):
        return self.universe.binnings(self.specs)

    @property
    def shape(self):
        return tuple(len(b) for b in self.binnings)

    @property
    def cells(self):
        return math.prod(self.shape)

    @property
    def labels(self):
        return tuple(b.labels for b in self.binnings)


class GeoLevel:
    """One geographic partition of a product's areas.

    `native` names the level the source reads directly.  When a level is
    incomparable with the spine -- tabulation block groups against the DAS
    "optimized" ones, say -- it has no native level, and `derive_from` plus
    `key` say how to reach it instead: read the finer level and re-aggregate
    by an extension-supplied atom map.  That map is irreducibly
    dataset-specific, which is why it is a hook and not an inference.
    """

    def __init__(self, name, native=None, derive_from=None, key=None,
                 requires=(), aliases=(), note=""):
        if native is None and derive_from is None:
            raise ValueError(f"geography {name!r}: needs a native level or a source to derive from")
        if derive_from is not None and key is None:
            raise ValueError(f"geography {name!r}: deriving needs a key function")
        self.name = name
        self.native = native
        self.derive_from = derive_from
        self.key = key
        self.requires = tuple(requires)
        self.aliases = tuple(aliases)
        self.note = note

    def __repr__(self):
        return f"<GeoLevel {self.name!r}>"


class GeoPlan:
    """How a requested geography will actually be read."""

    def __init__(self, level, read_level, aggregate=False, key=None):
        self.level = level
        self.read_level = read_level
        self.aggregate = aggregate
        self.key = key

    def __repr__(self):
        how = f"aggregated from {self.read_level}" if self.aggregate else "direct"
        return f"<GeoPlan {self.level.name!r}: {how}>"


class GeographyUnavailable(LookupError):
    """A geography is well defined but this product cannot answer it.

    The meet of two incomparable partitions always exists mathematically; it
    is usable only when measurements exist at the atoms.  DHC stops at county,
    so no re-aggregation recovers a tract.
    """


class Source:
    """Physical access for a product.  Implemented by the dataset extension.

    Core supplies the refinement tests that decide what may be pruned; the
    extension supplies the mapping from a selection to files, and normalizes
    whatever it finds into measurement rows.
    """

    def levels(self):
        """Names of the native levels this source can read."""
        raise NotImplementedError

    def read_measurements(self, level, selection, measurements):
        """Yield `MeasurementRow`s for the requested measurements."""
        raise NotImplementedError

    def read_evidence(self, level, selection):
        """Return {geo key: tuple of Evidence} for the selected areas."""
        raise NotImplementedError

    def geo_columns(self, level, selection, geo_keys):
        """Display columns (geoid, ...) for a list of geo keys, in order."""
        return {}

    def check_selection(self, level, selection):
        """Raise if a selection makes no sense for a level."""


class MeasurementRow:
    """One released query for one area, normalized."""

    def __init__(self, geo, measurement, values, variance):
        self.geo = geo
        self.measurement = measurement
        self.values = values
        self.variance = float(variance)

    def __repr__(self):
        return f"<MeasurementRow {self.measurement!r} at {self.geo!r}>"


class Product:
    """A dataset: a universe, what was measured over it, and how to read it."""

    def __init__(self, name, universe, measurements, geographies, source,
                 note=""):
        self.name = name
        self.universe = universe
        self.source = source
        self.note = note
        self._measurements = {m.name: m for m in measurements}
        self._geographies = {}
        self._canonical_geographies = []
        for level in geographies:
            self._canonical_geographies.append(level)
            for key in (level.name, *level.aliases):
                self._geographies[key] = level

    def __repr__(self):
        return f"<Product {self.name!r}: {len(self._measurements)} measurements>"

    # ── catalog surface ─────────────────────────────────────────────────────

    @property
    def axes(self):
        return list(self.universe.axis_names)

    def axis(self, name):
        return self.universe.axis(name)

    @property
    def geographies(self):
        return [level.name for level in self._canonical_geographies]

    def geography_table(self):
        """Tabulate every declared geography and whether it can be answered."""
        rows = []
        for level in self._canonical_geographies:
            try:
                plan = self.plan(level.name)
                how = ("aggregated from " + plan.read_level if plan.aggregate
                       else "direct")
            except GeographyUnavailable:
                how = "unavailable"
            rows.append({"geography": level.name, "how": how,
                         "requires": ", ".join(level.requires),
                         "note": level.note})
        return pd.DataFrame(rows)

    def measurement(self, name):
        canonical = self._match(name)
        return self._measurements[canonical]

    def measurements(self, geography=None):
        """Tabulate the measurements available, optionally for one geography.

        Returns a plain DataFrame: name, native query name, cells per area,
        the binning each axis uses, and any note.
        """
        if geography is not None:
            self.plan(geography)
        rows = []
        for m in self._measurements.values():
            rows.append({
                "measurement": m.name,
                "native": m.native,
                "cells": m.cells,
                **{axis: spec for axis, spec in zip(self.universe.axis_names, m.specs)},
                "note": m.note,
            })
        return pd.DataFrame(rows)

    def _match(self, name):
        """Resolve a user-typed measurement name, case- and space-insensitively."""
        by_key = {k.lower().replace(" ", ""): k for k in self._measurements}
        key = str(name).strip().lower().replace(" ", "")
        if key.endswith("_dpq"):
            key = key[:-4]
        if key not in by_key:
            raise KeyError(
                f"{self.name}: unknown measurement {name!r}; expected one of "
                f"{sorted(self._measurements)}")
        return by_key[key]

    # ── geography resolution ────────────────────────────────────────────────

    def plan(self, geography):
        """Decide how a requested geography will be read, or say why it cannot.

        Direct if the spine measures it.  Otherwise re-aggregate from a finer
        level -- the meet, computed by descending to atoms -- but only if that
        finer level is itself available.  Anything else is reported rather
        than silently answered with something coarser.
        """
        key = str(geography).strip().lower()
        if key not in self._geographies:
            raise KeyError(
                f"{self.name}: unknown geography {geography!r}; expected one "
                f"of {sorted(self._geographies)}")
        level = self._geographies[key]
        available = set(self.source.levels())

        if level.native is not None and level.native in available:
            return GeoPlan(level, level.native)
        if level.derive_from is not None:
            source_level = self._geographies[level.derive_from]
            if source_level.native in available:
                return GeoPlan(level, source_level.native, aggregate=True,
                               key=level.key)
        raise GeographyUnavailable(
            f"{self.name} cannot answer geography {level.name!r}: "
            + (f"the spine level {level.native!r} is not in this release"
               if level.native is not None else
               f"it would have to be re-aggregated from {level.derive_from!r}, "
               "which this release does not carry"))

    # ── the entry point ─────────────────────────────────────────────────────

    def view(self, geography, **selection):
        """Open a view: a set of areas whose cells are resolved consistently.

        Two measurements read from one view share latent cells wherever they
        name the same ones; two views are independent.  See `View`.
        """
        from .view import View
        return View(self, geography, **selection)
