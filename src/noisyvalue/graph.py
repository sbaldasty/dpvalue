import numpy as np
import scipy.stats
import sympy as sp

from . import util
from itertools import count
from sympy import Add, Integer, Mul, Symbol
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
        # A concrete subclass either declares its own full set of Parameters
        # directly in its own body, or inherits them from a shared base (e.g.
        # LocationScaleLatticeNode) -- search the MRO so both styles work.
        # A name defined at more than one level is ambiguous, not an
        # override, so that's a hard failure.
        seen_names = set()
        params = []
        for klass in cls.__mro__:
            for v in klass.__dict__.values():
                if isinstance(v, Parameter):
                    if v.name in seen_names:
                        raise TypeError(
                            f"{cls.__name__}: Parameter '{v.name}' is defined "
                            "in more than one class in its MRO"
                        )
                    seen_names.add(v.name)
                    params.append(v)
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


class GaussianNode(NoiseNode):
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


class LocationScaleLatticeNode(NoiseNode):
    """Shared sampling machinery for noise discretized onto the integer
    lattice, truncated to `[low, high]`. Pass `-oo`/`oo` for an unbounded
    side; callers needing that as a convenience default (rather than an
    explicit choice) should supply it at the `core.py` API layer.
    """

    loc = Parameter(0)
    scale = Parameter(1)
    low = Parameter(2)
    high = Parameter(3)

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
        width = cls.grid_width(scale)
        # Nudge center inside [low, high] to capture mass near the closer boundary
        k_center = int(np.round(loc))
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
        log_pmf = cls.log_density(k_vals, loc, scale)
        log_pmf = log_pmf - log_pmf.max()
        pmf = np.exp(log_pmf)
        pmf = pmf / pmf.sum()
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


class DiscreteGaussianNode(LocationScaleLatticeNode):
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


class DiscreteLaplaceNode(LocationScaleLatticeNode):
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

    def quantile(self, u):
        # sympy 1.14's LaplaceDistribution._quantile returns None, so
        # `sympy.stats.quantile` falls back to an unsolvable integral;
        # visualization prefers this closed form instead.  Continuous
        # approximation, like `sympy_rv`; truncation is not reflected.
        loc = float(self.loc)
        scale = float(self.scale)
        u = np.asarray(u, dtype=float)
        return loc - scale * np.sign(u - 0.5) * np.log1p(-2.0 * np.abs(u - 0.5))


class CensoredLatticeNode(NoiseNode):
    """A count observed only through the fact that its noisy measurement
    stayed at or below `cutoff`.

    Under a flat prior the posterior is pmf(k) proportional to
    F_noise(cutoff - k) on `[low, high]`: essentially uniform below the
    cutoff, rolling off through the noise's upper tail around it.  This is
    the posterior of a *suppressed* cell in a thresholded release, where a
    row's absence says its noisy count fell short.  `low` must be finite --
    the pmf tends to a constant as k -> -oo, so the flat prior is proper
    only on a bounded-below support.  Subclasses name the noise law whose
    CDF shapes the shoulder.
    """

    noise_cls = None
    cutoff = Parameter(0)
    scale = Parameter(1)
    low = Parameter(2)
    high = Parameter(3)

    @classmethod
    def noise_log_cdf(cls, x, scale):
        """log P(noise <= x), vectorized over x."""
        raise NotImplementedError

    def _resolved(self, resolved):
        cutoff = float(self.cutoff.subs(resolved))
        scale = float(self.scale.subs(resolved))
        low = LocationScaleLatticeNode.resolve_bound(self.low, resolved)
        high = LocationScaleLatticeNode.resolve_bound(self.high, resolved)
        return cutoff, scale, low, high

    @classmethod
    def _pmf(cls, cutoff, scale, low, high):
        # beyond the noise node's own grid width the shoulder carries no
        # sampled mass, so the support can stop there
        hi = int(np.floor(cutoff)) + cls.noise_cls.grid_width(scale)
        if np.isfinite(high):
            hi = min(hi, int(np.floor(high)))
        k = np.arange(int(np.ceil(low)), hi + 1)
        log_pmf = cls.noise_log_cdf(cutoff - k, scale)
        log_pmf = log_pmf - log_pmf.max()
        pmf = np.exp(log_pmf)
        return k, pmf / pmf.sum()

    @classmethod
    def _valid(cls, cutoff, scale, low, high):
        return (np.isfinite(cutoff) and np.isfinite(scale) and scale > 0
                and np.isfinite(low) and low <= high)

    def sample(self, rng, size=None, resolved=()):
        cutoff, scale, low, high = self._resolved(resolved)
        if not self._valid(cutoff, scale, low, high):
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        return self._draw(rng, cutoff, scale, low, high, size)

    @classmethod
    def _draw(cls, rng, cutoff, scale, low, high, size=None):
        k, pmf = cls._pmf(cutoff, scale, low, high)
        return rng.choice(k, size=size, p=pmf)

    @classmethod
    def sample_arrays(cls, rng, cutoff, scale, low, high):
        cutoff = np.asarray(cutoff, dtype=float)
        scale = np.asarray(scale, dtype=float)
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        n = cutoff.shape[0]
        result = np.empty(n, dtype=float)
        for i in range(n):
            if not cls._valid(cutoff[i], scale[i], low[i], high[i]):
                result[i] = np.nan
            else:
                result[i] = float(cls._draw(rng, cutoff[i], scale[i],
                                            low[i], high[i]))
        return result

    def quantile(self, u, resolved=()):
        cutoff, scale, low, high = self._resolved(resolved)
        if not self._valid(cutoff, scale, low, high):
            raise ValueError(
                "quantiles of a censored count need a finite lower bound "
                "and positive scale")
        k, pmf = self._pmf(cutoff, scale, low, high)
        cdf = np.cumsum(pmf)
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        idx = np.minimum(np.searchsorted(cdf, u), len(k) - 1)
        values = k[idx].astype(float)
        return values if values.ndim else float(values)


class GaussianCensoredNode(CensoredLatticeNode):
    noise_cls = DiscreteGaussianNode

    @classmethod
    def noise_log_cdf(cls, x, scale):
        # continuous-Gaussian approximation with continuity correction;
        # exact to ~1e-3 at the scales any release uses
        return scipy.stats.norm.logcdf((np.asarray(x, dtype=float) + 0.5) / scale)


class LaplaceCensoredNode(CensoredLatticeNode):
    noise_cls = DiscreteLaplaceNode

    @classmethod
    def noise_log_cdf(cls, x, scale):
        # exact two-sided geometric CDF with p = exp(-1/scale):
        # F(x) = p^(-x) / (1+p) for x < 0, and 1 - p^(x+1) / (1+p) above
        x = np.asarray(x, dtype=float)
        p = np.exp(-1.0 / scale)
        below = x / scale - np.log1p(p)
        # the branch is evaluated everywhere before `where` selects, so clamp
        # its argument into the region where it is defined
        above = np.log1p(-np.exp(-(np.maximum(x, 0.0) + 1.0) / scale)
                         / (1.0 + p))
        return np.where(x < 0, below, above)


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


def _extract_coeff_symbol(expr):
    """Return (coefficient, symbol) if expr is c*sym or sym, else (None, None)."""
    if isinstance(expr, Symbol):
        return Integer(1), expr
    if isinstance(expr, Mul):
        nums = [a for a in expr.args if a.is_number]
        syms = [a for a in expr.args if isinstance(a, Symbol)]
        rest = [a for a in expr.args if not a.is_number and not isinstance(a, Symbol)]
        if len(syms) == 1 and not rest:
            coeff = Mul(*nums) if nums else Integer(1)
            return coeff, syms[0]
    return None, None


class ConsolidationRule:
    """Base class for noise-combination rules used in consolidation.

    Concrete subclasses are auto-registered in `ConsolidationRule.registry`,
    which is the default rule set consolidate() applies.
    """

    registry = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ConsolidationRule.registry.append(cls())

    def matches(self, expr, symbol_to_node, eligible):
        raise NotImplementedError

    def apply(self, expr, symbol_to_node, eligible):
        raise NotImplementedError


class NormalSumRule(ConsolidationRule):
    """Collapse a linear combination of independent normal noise symbols into one."""

    def _parse(self, expr, symbol_to_node, eligible):
        if not isinstance(expr, Add):
            return None, None
        normal_terms = []
        other_args = []
        for arg in expr.args:
            coeff, sym = _extract_coeff_symbol(arg)
            if (
                sym is not None
                and sym in eligible
                and sym in symbol_to_node
                and isinstance(symbol_to_node[sym], NoiseNode)
                and isinstance(symbol_to_node[sym], GaussianNode)
                and not symbol_to_node[sym].deps
            ):
                normal_terms.append((coeff, symbol_to_node[sym]))
            else:
                other_args.append(arg)
        if len(normal_terms) < 2:
            return None, None
        return normal_terms, other_args

    def matches(self, expr, symbol_to_node, eligible):
        terms, _ = self._parse(expr, symbol_to_node, eligible)
        return terms is not None

    def apply(self, expr, symbol_to_node, eligible):
        normal_terms, other_args = self._parse(expr, symbol_to_node, eligible)
        combined_mu = sum(c * node.loc for c, node in normal_terms)
        combined_sigma = sp.sqrt(sum((c * node.scale) ** 2 for c, node in normal_terms))
        new_node = GaussianNode.create(loc=combined_mu, scale=combined_sigma)
        symbol_to_node[new_node.expr] = new_node
        for _, node in normal_terms:
            eligible.discard(node.expr)
        return Add(new_node.expr, *other_args)
