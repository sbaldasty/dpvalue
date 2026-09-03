"""OpenDP measurements that release `NoisyValue`s.

Drop-in wrappers for OpenDP's additive-noise mechanisms: `make_gaussian` and
`make_laplace` build the corresponding vetted OpenDP measurement internally,
let it perform the release (its sampler and privacy map are untouched), and
wrap each released number in a `NoisyValue` whose posterior encodes the
mechanism's public metadata -- distribution family and scale, discrete or
continuous per the input domain's carrier type.  The wrapper is a genuine
OpenDP `Measurement` (via `make_user_measurement`), so it chains with
transformations and composes like any other::

    import opendp.prelude as dp
    from noisyvalue import opendp as nvdp

    dp.enable_features("contrib", "honest-but-curious")

    meas = nvdp.make_gaussian(
        dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float),
        scale=2.0)
    value = meas(exact_aggregate)   # a NoisyFloat

`honest-but-curious` is genuinely required: the wrapping function receives the
exact aggregate (it hands it straight to the inner OpenDP measurement and
touches nothing else), and OpenDP rightly insists that running plugin code on
private data be an explicit opt-in.

`SharedLatentRegistry` gives statistics stable identities across releases.
Two measurements observed under one key are recognized as the same unknown
true value: their posteriors are combined in closed form (Gaussian family
only), every `NoisyValue` previously issued for that key tightens in place,
and all of them stay perfectly correlated under joint sampling.  This is the
construction-time analogue of `dataset/solve.py`'s closed forms -- the
sampler resolves latents by symbolic solve and cannot condition an
overdetermined system, so combination has to happen when the value is built.
"""

import hashlib
import math

import sympy as sp

try:
    import opendp.prelude as dp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "noisyvalue.opendp requires the 'opendp' package; install the "
        "project's [opendp] extra (e.g. `uv sync --extra opendp`)"
    ) from e

from .core import NoisyFloat, NoisyInt
from .graph import DiscreteGaussianNode, DiscreteLaplaceNode
from .graph import GaussianNode, LaplaceNode
from .pandas import NoisyFloatArray, NoisyIntArray


_INT_ATOMS = frozenset(
    {"i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128",
     "usize", "int"})
_FLOAT_ATOMS = frozenset({"f32", "f64", "float"})

_NODE_CLASSES = {
    ("gaussian", True): DiscreteGaussianNode,
    ("gaussian", False): GaussianNode,
    ("laplace", True): DiscreteLaplaceNode,
    ("laplace", False): LaplaceNode,
}


def _parse_carrier(input_domain):
    """(is_vector, is_discrete) from an OpenDP domain's carrier type."""
    t = str(input_domain.carrier_type)
    vector = t.startswith("Vec<") and t.endswith(">")
    atom = t[4:-1] if vector else t
    if atom in _INT_ATOMS:
        return vector, True
    if atom in _FLOAT_ATOMS:
        return vector, False
    raise ValueError(f"Unsupported input domain carrier type: {t}")


def _release_variance(mechanism, discrete, scale):
    """Sampling variance of the mechanism's noise, in parameter space.

    For the Gaussian family this is scale**2 for the continuous and discrete
    laws alike -- what matters downstream is that products of Gaussian kernels
    combine exactly through these parameters, which holds on the lattice too.
    The discrete Laplace formula matches
    `dataset.solve.DiscreteLaplaceFamily.variance_from_scale`.
    """
    scale = float(scale)
    if mechanism == "gaussian":
        return scale ** 2
    if discrete:
        p = math.exp(-1.0 / scale)
        return 2.0 * p / (1.0 - p) ** 2
    return 2.0 * scale ** 2


# A support bound this far outside the posterior's centre lies beyond the
# noise's sampled/informative range entirely, so it is dropped as provably
# inert (same reasoning as `dataset.solve.LatticeFamily._drop_inert`).
_INERT_FLOOR = 60.0
_INERT_SCALES = {"gaussian": 8.0, "laplace": 32.0}


def _posterior_node(mechanism, discrete, center, scale, low=None, high=None):
    """A noise node whose law *is* the statistic's posterior.

    Under a flat prior a single additive-noise measurement leaves the noise
    law re-centred at the observation, so the posterior is encoded directly
    as a node rather than through a latent-plus-constraint pair (the same
    choice `dataset/solve.py` makes for release cells).  `low`/`high` are
    prior knowledge that the true value cannot fall outside them (e.g. zero
    as a lower bound for counts, or a released statistic's own declared
    sensitivity bounds) -- both `GaussianNode`/`LaplaceNode` and their
    discrete-lattice counterparts accept truncation the same way.
    """
    node_cls = _NODE_CLASSES[(mechanism, discrete)]
    center = float(center)
    scale = float(scale)
    slack = max(_INERT_FLOOR, _INERT_SCALES[mechanism] * scale)
    if low is not None and low < center - slack:
        low = None
    if high is not None and high > center + slack:
        high = None
    to_bound = (lambda v: sp.Integer(int(v))) if discrete else (lambda v: sp.Float(float(v)))
    params = {
        "loc": sp.Float(center),
        "scale": sp.Float(scale),
        "low": -sp.oo if low is None else to_bound(low),
        "high": sp.oo if high is None else to_bound(high),
    }
    return node_cls.create(**params)


def _issue(obs, node, discrete):
    cls = NoisyInt if discrete else NoisyFloat
    return cls(obs, node)


def observe_release(obs, *, mechanism, scale, discrete=None, low=None, high=None):
    """Interpret one already-released number as a `NoisyValue`.

    `mechanism` is `"gaussian"` or `"laplace"`; `discrete` selects the
    integer-lattice law and defaults to whether `obs` is an integer;
    `low`/`high` are prior knowledge that the true value cannot fall outside
    them, e.g. zero as a lower bound for counts, or declared sensitivity
    bounds on a released statistic.  The value is unshared -- use a
    `SharedLatentRegistry` to give repeated measurements of one statistic a
    common identity.
    """
    if mechanism not in ("gaussian", "laplace"):
        raise ValueError(f"Unknown mechanism: {mechanism!r}")
    if discrete is None:
        discrete = isinstance(obs, int)
    node = _posterior_node(mechanism, discrete, obs, scale, low=low, high=high)
    return _issue(obs, node, discrete)


class _Statistic:
    __slots__ = ("mechanism", "discrete", "center", "variance", "node")

    def __init__(self, mechanism, discrete, center, variance, node):
        self.mechanism = mechanism
        self.discrete = discrete
        self.center = center
        self.variance = variance
        self.node = node


class SharedLatentRegistry:
    """Stable identities for released statistics, so repeated measurements
    of one statistic share a posterior.

    Every value issued for a key is backed by the *same* node object, so all
    of them are one random variable to the joint sampler, and a later
    measurement of the key tightens every previously issued value in place:
    a registry's values are live views of each statistic's current posterior,
    not snapshots of the release that produced them.

    Combination is closed-form and Gaussian-family only: the product of two
    Gaussian kernels (continuous or on the integer lattice) is again one, at
    the precision-weighted centre.  Laplace posteriors do not stay in family
    under a second measurement, so a repeated Laplace key is refused -- the
    same boundary `dataset/solve.py` draws for its pair closed form.
    """

    def __init__(self):
        self._stats = {}
        self._folded_sums = set()

    def observe(self, key, obs, *, mechanism, scale, discrete=None, low=None, high=None):
        """Record a release of the statistic `key` and return its value.

        `low`/`high` bound the support (see `observe_release`); they are
        fixed by the key's first observation and ignored afterwards.
        """
        if mechanism not in ("gaussian", "laplace"):
            raise ValueError(f"Unknown mechanism: {mechanism!r}")
        if discrete is None:
            discrete = isinstance(obs, int)
        stat = self._stats.get(key)
        if stat is None:
            node = _posterior_node(mechanism, discrete, obs, scale, low=low, high=high)
            self._stats[key] = _Statistic(
                mechanism, discrete, float(obs),
                _release_variance(mechanism, discrete, scale), node)
            return _issue(obs, node, discrete)
        if (stat.mechanism, stat.discrete) != (mechanism, discrete):
            raise ValueError(
                f"Key {key!r} was first observed as "
                f"{'discrete ' if stat.discrete else ''}{stat.mechanism}; "
                f"a {'discrete ' if discrete else ''}{mechanism} measurement "
                "cannot be the same statistic")
        self._combine(stat, float(obs),
                      _release_variance(mechanism, discrete, scale))
        return _issue(obs, stat.node, discrete)

    def add_estimate(self, key, center, variance):
        """Fold an independent unbiased estimate into `key`'s posterior.

        The estimate must be independent of every measurement already folded
        into the key.  The estimate enters as numbers, not nodes, so any
        correlation between the source of the estimate and this key's value
        is not represented in the graph -- joint samples of the two remain
        independent even though the marginals are right.
        """
        self._combine(self._require(key), float(center), float(variance))

    def add_sum_estimate(self, total_key, part_keys):
        """Fold the sum of some parts' posteriors into a total's posterior.

        The parts' current centres and variances are summed and folded into
        `total_key` as one independent estimate -- correct when the parts
        were measured independently of the total's own measurements.  For
        discrete parts the sum of lattice laws is treated as Gaussian with
        the summed variance, a close approximation.  As with `add_estimate`,
        the induced correlation between total and parts is not represented.
        """
        fold = (total_key, tuple(part_keys))
        if fold in self._folded_sums:
            raise ValueError(
                f"The sum of {list(part_keys)} was already folded into "
                f"{total_key!r}; folding the same evidence twice would "
                "double-count it")
        parts = [self._require(k) for k in part_keys]
        self.add_estimate(
            total_key,
            sum(p.center for p in parts),
            sum(p.variance for p in parts))
        self._folded_sums.add(fold)

    def posterior(self, key):
        """(center, variance) of the statistic's current posterior."""
        stat = self._require(key)
        return stat.center, stat.variance

    def current(self, key):
        """A fresh value viewing `key`'s current posterior.

        Its observation is the posterior centre (rounded for discrete
        statistics), which is generally *not* any single released number.
        """
        stat = self._require(key)
        obs = round(stat.center) if stat.discrete else stat.center
        return _issue(obs, stat.node, stat.discrete)

    def _require(self, key):
        stat = self._stats.get(key)
        if stat is None:
            raise KeyError(f"No statistic observed under key {key!r}")
        return stat

    @staticmethod
    def _combine(stat, center, variance):
        if stat.mechanism != "gaussian":
            raise NotImplementedError(
                "Combining repeated measurements is closed-form only for the "
                "Gaussian family; a second Laplace measurement of one "
                "statistic has a posterior outside the library's noise "
                "families. Use distinct keys, or a Gaussian mechanism.")
        if variance <= 0:
            raise ValueError(f"Estimate variance must be positive: {variance}")
        precision = 1.0 / stat.variance + 1.0 / variance
        combined = (stat.center / stat.variance + center / variance) / precision
        stat.center = combined
        stat.variance = 1.0 / precision
        # Node params are read at sample time, so updating them in place
        # retunes every value already issued for this statistic.
        stat.node.params[0] = sp.Float(combined)
        stat.node.params[1] = sp.Float(math.sqrt(stat.variance))


def make_gaussian(input_domain, input_metric, scale, *,
                  key=None, keys=None, registry=None, low=None, high=None,
                  **kwargs):
    """`dp.m.make_gaussian`, releasing `NoisyValue`s instead of numbers.

    Scalar domains yield one value (`key` names its statistic); vector
    domains yield a list (`keys` names each element's statistic, aligned by
    position).  Keys require a `registry`; without keys each release is an
    unshared value.  `low`/`high` are prior knowledge that the true value
    cannot fall outside them (see `observe_release`); passed to every
    released element alike.  Extra keyword arguments pass through to the
    OpenDP constructor.
    """
    inner = dp.m.make_gaussian(input_domain, input_metric, scale, **kwargs)
    return _wrap(inner, input_domain, input_metric, "gaussian", scale,
                 key, keys, registry, low, high)


def make_laplace(input_domain, input_metric, scale, *,
                 key=None, keys=None, registry=None, low=None, high=None,
                 **kwargs):
    """`dp.m.make_laplace`, releasing `NoisyValue`s instead of numbers.

    See `make_gaussian` for the `key`/`keys`/`registry`/`low`/`high`
    conventions.
    """
    inner = dp.m.make_laplace(input_domain, input_metric, scale, **kwargs)
    return _wrap(inner, input_domain, input_metric, "laplace", scale,
                 key, keys, registry, low, high)


def _wrap(inner, input_domain, input_metric, mechanism, scale,
          key, keys, registry, low=None, high=None):
    vector, discrete = _parse_carrier(input_domain)
    if vector and key is not None:
        raise ValueError("Vector domains take keys=, not key=")
    if not vector and keys is not None:
        raise ValueError("Scalar domains take key=, not keys=")
    if (key is not None or keys is not None) and registry is None:
        raise ValueError("Sharing statistics by key requires a registry")

    def build(released, k):
        if registry is not None and k is not None:
            return registry.observe(k, released, mechanism=mechanism,
                                    scale=scale, discrete=discrete,
                                    low=low, high=high)
        return observe_release(released, mechanism=mechanism, scale=scale,
                               discrete=discrete, low=low, high=high)

    def function(arg):
        released = inner(arg)
        if not vector:
            return build(released, key)
        ks = tuple(keys) if keys is not None else (None,) * len(released)
        if len(ks) != len(released):
            raise ValueError(
                f"Got {len(released)} released elements for {len(ks)} keys")
        return [build(x, k) for x, k in zip(released, ks)]

    return dp.m.make_user_measurement(
        input_domain, input_metric, inner.output_measure, function,
        privacy_map=inner.map)


# ── polars Context adapter ───────────────────────────────────────────────────
#
# OpenDP's polars integration is declarative enough to drive the whole
# wrapping automatically: `query.summarize()` reports, per noised output
# column, the aggregate, the noise distribution, and its scale -- exactly the
# public metadata a posterior needs -- and the released frame's remaining
# columns are the group-by keys, which give each cell a stable identity for
# the shared-latent registry.  `release_query` runs one query through that
# pipeline; `link_sum` folds a fine partition's sums into a coarser release
# of the same aggregate.

_DISTRIBUTIONS = {
    "Integer Gaussian": ("gaussian", True),
    "Float Gaussian": ("gaussian", False),
    "Integer Laplace": ("laplace", True),
    "Float Laplace": ("laplace", False),
}


def _query_namespace(query):
    """A stable identity for the statistic a query measures.

    Two queries with byte-identical plans measure the same statistics, so a
    hash of the serialized plan lets identical queries share latents without
    any declaration.  The plan does *not* embed the data -- a registry
    therefore describes one private dataset, and releases from different
    datasets must use different registries (or explicit namespaces).
    """
    try:
        blob = query.serialize()
    except Exception:
        blob = query.explain().encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


class QueryRelease:
    """One collected DP release, with noised columns wrapped as noisy values.

    - `frame`: tidy pandas DataFrame; group-by key columns are plain data,
      each noised column is a `NoisyIntArray`/`NoisyFloatArray` whose cells
      are live views of the registry's posteriors (they tighten in place as
      more evidence about the same statistics is folded in).
    - `key_columns`: the group-by columns, in released order.
    - `noised_columns`: `{name: (aggregate, mechanism, discrete, scale)}`.
    - `namespace`: the identity prefix under which cells were registered.
    """

    def __init__(self, frame, key_columns, noised_columns, namespace,
                 registry, cell_keys):
        self.frame = frame
        self.key_columns = key_columns
        self.noised_columns = noised_columns
        self.namespace = namespace
        self._registry = registry
        self._cell_keys = cell_keys

    def __repr__(self):
        return (f"QueryRelease(namespace={self.namespace!r}, "
                f"keys={list(self.key_columns)}, "
                f"noised={list(self.noised_columns)})\n{self.frame}")


def release_query(query, *, registry=None, namespace=None,
                  on_unsupported="raise"):
    """Release an OpenDP polars query as a frame of noisy values.

    `query` is a `LazyFrameQuery` from `dp.Context.query()`.  The query's
    own `summarize()` supplies each noised column's mechanism and scale;
    OpenDP performs the release untouched (`release().collect()` -- the
    query's entire budget is spent here, once), and each released cell is
    wrapped in a posterior over its true value.  Unsigned released columns
    get the prior bound true >= 0.

    With a `registry`, every cell is observed under the key
    `(namespace, column, ((key_col, value), ...))`, so releasing an
    identical query again -- or linking a related one via `link_sum` --
    combines evidence instead of creating parallel values.  `namespace`
    defaults to a hash of the query plan; pass it explicitly to share
    latents across queries whose plans differ only in noise parameters.

    Released columns the wrapper cannot model -- composite aggregates like
    `dp.mean` (released as a ratio of two noised numbers) or unknown
    distributions -- raise by default; `on_unsupported="keep"` passes them
    through as plain numbers instead.
    """
    if on_unsupported not in ("raise", "keep"):
        raise ValueError(f"Unknown on_unsupported mode: {on_unsupported!r}")
    if namespace is None:
        namespace = _query_namespace(query)

    summary = {}
    for row in query.summarize().to_dicts():
        summary.setdefault(row["column"], []).append(row)

    released = query.release().collect()

    noised = {}
    unsupported = {}
    for column, rows in summary.items():
        if len(rows) != 1:
            unsupported[column] = (
                f"composite aggregate ({' + '.join(r['aggregate'] for r in rows)}) "
                "is released as a ratio, not an additively noised number")
            continue
        row = rows[0]
        if row["distribution"] not in _DISTRIBUTIONS:
            unsupported[column] = f"unknown distribution {row['distribution']!r}"
            continue
        mechanism, discrete = _DISTRIBUTIONS[row["distribution"]]
        noised[column] = (row["aggregate"], mechanism, discrete, row["scale"])
    if unsupported and on_unsupported == "raise":
        problems = "; ".join(f"{c}: {why}" for c, why in unsupported.items())
        raise ValueError(
            f"Cannot wrap released column(s) as noisy values ({problems}). "
            'Pass on_unsupported="keep" to release them as plain numbers.')

    key_columns = tuple(c for c in released.columns if c not in summary)
    key_rows = released.select(key_columns).rows() if key_columns \
        else [()] * released.height

    import polars as pl
    unsigned = {pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}

    frame = released.to_pandas()
    cell_keys = {}
    for column, (aggregate, mechanism, discrete, scale) in noised.items():
        low = 0 if released.schema[column] in unsigned else None
        keys = []
        values = []
        for obs, key_row in zip(released[column].to_list(), key_rows):
            obs = int(obs) if discrete else float(obs)
            if registry is None:
                keys.append(None)
                values.append(observe_release(
                    obs, mechanism=mechanism, scale=scale,
                    discrete=discrete, low=low))
            else:
                key = (namespace, column, tuple(zip(key_columns, key_row)))
                keys.append(key)
                values.append(registry.observe(
                    key, obs, mechanism=mechanism, scale=scale,
                    discrete=discrete, low=low))
        cell_keys[column] = tuple(keys)
        array_cls = NoisyIntArray if discrete else NoisyFloatArray
        frame[column] = array_cls._from_sequence(values)

    return QueryRelease(frame, key_columns, noised, namespace, registry,
                        cell_keys)


def link_sum(fine, coarse, column=None):
    """Fold a fine release's group sums into a coarser release's posteriors.

    `fine` and `coarse` are `QueryRelease`s of the same additive aggregate
    over the same data, made through the same registry, where `coarse`
    groups by a proper subset of `fine`'s key columns (possibly none: a
    grand total).  Each coarse cell equals the sum of its matching fine
    cells, so the fine cells' summed posterior enters the coarse cell's as
    an independent estimate -- every noisy value already issued for the
    coarse cells tightens in place.

    Assumes the fine release covers each coarse group completely (group by
    a margin with `invariant="keys"`; a DP-thresholded key set silently
    drops small groups and would bias the fold) and that the two releases'
    noise is independent (distinct queries always are).  As in
    `SharedLatentRegistry.add_sum_estimate`, the induced coarse-fine
    correlation is not represented in the graph.
    """
    if fine._registry is None or fine._registry is not coarse._registry:
        raise ValueError(
            "link_sum needs both releases made through one shared registry")
    if not set(coarse.key_columns) < set(fine.key_columns):
        raise ValueError(
            f"Coarse keys {list(coarse.key_columns)} must be a proper "
            f"subset of fine keys {list(fine.key_columns)}")
    if column is None:
        shared = set(fine.noised_columns) & set(coarse.noised_columns)
        if len(shared) != 1:
            raise ValueError(
                f"Pass column= explicitly; the releases share "
                f"{sorted(shared) or 'no'} noised columns")
        (column,) = shared

    fine_pos = {c: i for i, c in enumerate(fine.key_columns)}
    project = tuple(fine_pos[c] for c in coarse.key_columns)
    parts_by_group = {}
    for key, cell_key in zip(
            fine.frame[list(fine.key_columns)].itertuples(index=False),
            fine._cell_keys[column]):
        group = tuple(key[i] for i in project)
        parts_by_group.setdefault(group, []).append(cell_key)

    registry = coarse._registry
    coarse_rows = coarse.frame[list(coarse.key_columns)].itertuples(index=False) \
        if coarse.key_columns else [()] * len(coarse.frame)
    for key, cell_key in zip(coarse_rows, coarse._cell_keys[column]):
        parts = parts_by_group.get(tuple(key))
        if not parts:
            raise ValueError(
                f"No fine cells found for coarse group {tuple(key)}; the "
                "fine release does not cover this group")
        registry.add_sum_estimate(cell_key, parts)
