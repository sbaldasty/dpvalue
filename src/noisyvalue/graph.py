import numpy as np

from . import util
from itertools import count
from sympy import Symbol
from sympy import sympify
from sympy.stats import Laplace, Normal
from sympy.stats.frv_types import BinomialDistribution, rv


class Name:
    counter = count()

    @classmethod
    def fresh(cls):
        return f"id_{next(cls.counter)}"


class Parameter:
    def __init__(self, index=None):
        self.index = index
        self.name = None
        self.owner = None

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return obj.params[self.index]


class Node:
    def __init__(self, deps=()):
        self.expr = Symbol(Name.fresh())
        self.deps = util.as_tuple(deps, Node)

    def closure(self):
        seen = set()
        ordered = []

        def walk(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            ordered.append(node)
            for dep in node.deps:
                walk(dep)

        walk(self)
        return tuple(ordered)

    def latent_symbols(self):
        return {x.expr for x in self.closure() if isinstance(x, LatentNode)}

    def all_constraints(self):
        return frozenset(
            c
            for node in self.closure()
            if isinstance(node, DerivedNode)
            for c in node.constraints
        )


class LatentNode(Node):
    pass


class DerivedNode(Node):
    def __init__(self, expr, constraints=(), deps=()):
        super().__init__(deps)
        self.expr = sympify(expr)
        self.constraints = frozenset(sympify(x) for x in constraints)

    @classmethod
    def operational(cls, expr, deps=()):
        flat_deps = []
        flat_eqns = []
        for node in deps:
            if isinstance(node, DerivedNode):
                flat_eqns.extend(node.constraints)
                flat_deps.extend(node.deps)
            else:
                flat_deps.append(node)
        return cls(expr, frozenset(flat_eqns), frozenset(flat_deps))


class NoiseNode(Node):
    def __init__(self, params, deps=()):
        super().__init__(deps)
        self.params = list(params)

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Each concrete subclass declares its own full set of Parameters
        # directly in its own body -- deliberately not inherited from a base
        # or mixin, which would force coordinating indices across classes.
        params = [v for v in cls.__dict__.values() if isinstance(v, Parameter)]
        cls._parameters = tuple(sorted(params, key=lambda p: p.index))

    def param_symbols(self):
        return {s for p in self.params for s in p.free_symbols}

    def sympy_rv(self):
        raise NotImplementedError

    def sample(self, rng, size=None, resolved=None):
        raise NotImplementedError

    @classmethod
    def sample_arrays(cls, rng, *param_arrays):
        raise NotImplementedError

    @classmethod
    def create(cls, deps=(), **kwargs):
        params = []
        for p in cls._parameters:
            if p.name in kwargs:
                params.append(sympify(kwargs[p.name]))
            else:
                raise TypeError(f"Missing parameter: {p.name}")
        return cls(params, deps)


class NormalNode(NoiseNode):
    loc = Parameter(0)
    scale = Parameter(1)

    def sympy_rv(self):
        return Normal(Name.fresh(), self.loc, self.scale)

    def sample(self, rng, size=None, resolved=()):
        loc = float(self.loc.subs(resolved))
        scale = float(self.scale.subs(resolved))
        return rng.normal(loc, scale, size=size)

    @classmethod
    def sample_arrays(cls, rng, loc, scale):
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        valid = np.isfinite(loc) & (scale > 0)
        if np.all(valid):
            return rng.normal(loc, scale)
        result = rng.normal(np.where(valid, loc, 0.0), np.where(valid, scale, 1.0))
        return np.where(valid, result, np.nan)


class BinomialNode(NoiseNode):
    trials = Parameter(0)
    prob = Parameter(1)

    def sympy_rv(self):
        return rv(Name.fresh(), BinomialDistribution, self.trials, self.prob, check=False)

    def sample(self, rng, size=None, resolved=()):
        try:
            trials = int(self.trials.subs(resolved))
            prob = float(self.prob.subs(resolved))
        except (TypeError, ValueError):
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        if trials < 0 or not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        return rng.binomial(trials, prob, size=size)

    @classmethod
    def sample_arrays(cls, rng, n, p):
        n = np.asarray(n, dtype=float)
        p = np.asarray(p, dtype=float)
        valid = (n >= 0) & np.isfinite(p) & (p >= 0) & (p <= 1)
        result = np.asarray(rng.binomial(
            np.where(valid, n, 0).astype(int),
            np.where(valid, p, 0.5)), dtype=float)
        return np.where(valid, result, np.nan)


def _discretize_grid(log_density, center, width, low, high):
    """Integer support and normalized pmf for `log_density`, built on a grid
    of half-width `width` around `center` and truncated to `[low, high]`.

    The grid center is nudged inside `[low, high]` first (when finite) so
    that mass near the closer boundary is captured even when `center` itself
    lies outside the truncation window.
    """
    k_center = int(np.round(center))
    if np.isfinite(low):
        k_center = max(k_center, int(np.ceil(low)))
    if np.isfinite(high):
        k_center = min(k_center, int(np.floor(high)))
    lo, hi = k_center - width, k_center + width
    if np.isfinite(low):
        lo = max(lo, int(np.ceil(low)))
    if np.isfinite(high):
        hi = min(hi, int(np.floor(high)))
    k_vals = np.arange(lo, hi + 1)
    log_pmf = log_density(k_vals)
    log_pmf = log_pmf - log_pmf.max()
    pmf = np.exp(log_pmf)
    return k_vals, pmf / pmf.sum()


class LatticeNode(NoiseNode):
    """Shared sampling machinery for noise discretized onto the integer
    lattice, truncated to `[low, high]`. Pass `-oo`/`oo` for an unbounded
    side; callers needing that as a convenience default (rather than an
    explicit choice) should supply it at the `core.py` API layer.

    Subclasses declare their own `loc`/`scale`/`low`/`high` Parameters (so
    each class's index numbering is self-contained) and supply the
    distribution shape via `grid_width` (half-width of the untruncated
    grid, tuned to that distribution's tail) and `log_density`.
    """

    @classmethod
    def grid_width(cls, scale):
        raise NotImplementedError

    @classmethod
    def log_density(cls, k, loc, scale):
        raise NotImplementedError

    @staticmethod
    def resolve_bound(value, resolved):
        # Bounds are almost always already-concrete numbers (±oo included),
        # so skip the sympy substitution when there is nothing to resolve.
        if value.free_symbols:
            value = value.subs(resolved)
        return float(value)

    def sample(self, rng, size=None, resolved=()):
        loc = float(self.loc.subs(resolved))
        scale = float(self.scale.subs(resolved))
        low = self.resolve_bound(self.low, resolved)
        high = self.resolve_bound(self.high, resolved)
        if not np.isfinite(loc) or not np.isfinite(scale) or scale <= 0 or low > high:
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        return self._draw(rng, loc, scale, low, high, size)

    @classmethod
    def _draw(cls, rng, loc, scale, low, high, size=None):
        k_vals, pmf = _discretize_grid(
            lambda k: cls.log_density(k, loc, scale),
            loc, cls.grid_width(scale), low, high)
        return rng.choice(k_vals, size=size, p=pmf)

    @classmethod
    def sample_arrays(cls, rng, loc, scale, low, high):
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        n = loc.shape[0]
        if (np.all(loc == loc[0]) and np.all(scale == scale[0])
                and np.all(low == low[0]) and np.all(high == high[0])):
            l, s, lo, hi = float(loc[0]), float(scale[0]), float(low[0]), float(high[0])
            if not np.isfinite(l) or not np.isfinite(s) or s <= 0 or lo > hi:
                return np.full(n, np.nan, dtype=float)
            return cls._draw(rng, l, s, lo, hi, size=n).astype(float)
        result = np.empty(n, dtype=float)
        for i in range(n):
            l, s = float(loc[i]), float(scale[i])
            lo, hi = float(low[i]), float(high[i])
            if not np.isfinite(l) or not np.isfinite(s) or s <= 0 or lo > hi:
                result[i] = np.nan
            else:
                result[i] = float(cls._draw(rng, l, s, lo, hi))
        return result


class DiscreteGaussianNode(LatticeNode):
    loc = Parameter(0)
    scale = Parameter(1)
    low = Parameter(2)
    high = Parameter(3)

    @classmethod
    def grid_width(cls, scale):
        return int(np.ceil(max(50.0, 6.0 * abs(scale))))

    @classmethod
    def log_density(cls, k, loc, scale):
        return -0.5 * ((k - loc) ** 2) / (scale ** 2)

    def sympy_rv(self):
        # Continuous Normal as quantile-space approximation for visualization.
        # Truncation (if any) is not reflected here.
        return Normal(Name.fresh(), self.loc, self.scale)


class DiscreteLaplaceNode(LatticeNode):
    loc = Parameter(0)
    scale = Parameter(1)
    low = Parameter(2)
    high = Parameter(3)

    @classmethod
    def grid_width(cls, scale):
        # Exponential tails are heavier than Gaussian's, so this needs a
        # bigger multiple of scale for comparable truncation error.
        return int(np.ceil(max(50.0, 30.0 * abs(scale))))

    @classmethod
    def log_density(cls, k, loc, scale):
        return -np.abs(k - loc) / scale

    def sympy_rv(self):
        # Continuous Laplace as quantile-space approximation for visualization.
        # Truncation (if any) is not reflected here.
        return Laplace(Name.fresh(), self.loc, self.scale)


def topological_sort_law_nodes(law_nodes):
    law_symbols = {node.expr for node in law_nodes}
    by_symbol = {node.expr: node for node in law_nodes}
    predecessors = {
        node.expr: {
            dep.expr
            for dep in node.deps
            if not isinstance(dep, DerivedNode) and dep.expr in law_symbols
        }
        for node in law_nodes
    }
    ordered = []
    resolved = set()
    remaining = set(law_symbols)
    while remaining:
        ready = {sym for sym in remaining if predecessors[sym] <= resolved}
        for sym in sorted(ready, key=str):
            ordered.append(by_symbol[sym])
            resolved.add(sym)
            remaining.discard(sym)
    return tuple(ordered)
