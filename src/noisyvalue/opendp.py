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


def _posterior_node(mechanism, discrete, center, scale):
    """A noise node whose law *is* the statistic's posterior.

    Under a flat prior a single additive-noise measurement leaves the noise
    law re-centred at the observation, so the posterior is encoded directly
    as a node rather than through a latent-plus-constraint pair (the same
    choice `dataset/solve.py` makes for release cells).
    """
    node_cls = _NODE_CLASSES[(mechanism, discrete)]
    params = {"loc": sp.Float(float(center)), "scale": sp.Float(float(scale))}
    if discrete:
        params.update(low=-sp.oo, high=sp.oo)
    return node_cls.create(**params)


def _issue(obs, node, discrete):
    cls = NoisyInt if discrete else NoisyFloat
    return cls(obs, node)


def observe_release(obs, *, mechanism, scale, discrete=None):
    """Interpret one already-released number as a `NoisyValue`.

    `mechanism` is `"gaussian"` or `"laplace"`; `discrete` selects the
    integer-lattice law and defaults to whether `obs` is an integer.  The
    value is unshared -- use a `SharedLatentRegistry` to give repeated
    measurements of one statistic a common identity.
    """
    if mechanism not in ("gaussian", "laplace"):
        raise ValueError(f"Unknown mechanism: {mechanism!r}")
    if discrete is None:
        discrete = isinstance(obs, int)
    node = _posterior_node(mechanism, discrete, obs, scale)
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

    def observe(self, key, obs, *, mechanism, scale, discrete=None):
        """Record a release of the statistic `key` and return its value."""
        if mechanism not in ("gaussian", "laplace"):
            raise ValueError(f"Unknown mechanism: {mechanism!r}")
        if discrete is None:
            discrete = isinstance(obs, int)
        stat = self._stats.get(key)
        if stat is None:
            node = _posterior_node(mechanism, discrete, obs, scale)
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
        parts = [self._require(k) for k in part_keys]
        self.add_estimate(
            total_key,
            sum(p.center for p in parts),
            sum(p.variance for p in parts))

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
                  key=None, keys=None, registry=None, **kwargs):
    """`dp.m.make_gaussian`, releasing `NoisyValue`s instead of numbers.

    Scalar domains yield one value (`key` names its statistic); vector
    domains yield a list (`keys` names each element's statistic, aligned by
    position).  Keys require a `registry`; without keys each release is an
    unshared value.  Extra keyword arguments pass through to the OpenDP
    constructor.
    """
    inner = dp.m.make_gaussian(input_domain, input_metric, scale, **kwargs)
    return _wrap(inner, input_domain, input_metric, "gaussian", scale,
                 key, keys, registry)


def make_laplace(input_domain, input_metric, scale, *,
                 key=None, keys=None, registry=None, **kwargs):
    """`dp.m.make_laplace`, releasing `NoisyValue`s instead of numbers.

    See `make_gaussian` for the `key`/`keys`/`registry` conventions.
    """
    inner = dp.m.make_laplace(input_domain, input_metric, scale, **kwargs)
    return _wrap(inner, input_domain, input_metric, "laplace", scale,
                 key, keys, registry)


def _wrap(inner, input_domain, input_metric, mechanism, scale,
          key, keys, registry):
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
                                    scale=scale, discrete=discrete)
        return observe_release(released, mechanism=mechanism, scale=scale,
                               discrete=discrete)

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
