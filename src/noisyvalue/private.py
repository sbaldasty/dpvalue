"""Sensitivity tracking and DP releases for survey-shaped, per-unit data.

`noisyvalue.opendp` wraps OpenDP measurements whose sensitivity is already
known -- the caller (or an OpenDP transformation chain) supplies a `scale`
and OpenDP's own metric/transformation machinery is what established it.
This module is for the case that machinery doesn't reach: statistics built
directly from per-respondent contributions, such as a design-weighted
survey total, where the sensitivity comes from *this* library tracking
which `PrivacyUnit` each term traces back to.

`PrivateValue` mirrors `NoisyValue`'s tandem `(value, expr)` update, but for
a sensitivity bound instead of a posterior: `PrivateNumber` additionally
carries an interval `(lo, hi)` that propagates through arithmetic (interval
arithmetic -- always a sound enclosure, linear or not). `PrivateDataset`
implements exactly one sensitivity rule on top of that: for a statistic
expressed as a sum of terms, each traceable to exactly one `PrivacyUnit`,
the add/remove-one-unit global sensitivity is the worst-case magnitude any
single unit's total contribution can reach. `release()` refuses a term that
mixes more than one unit rather than silently guessing at its sensitivity.

That one rule covers more than it looks like:

- A design-weighted total/mean: each unit contributes `weight * value`,
  weight and value both bounded (trim weights the way survey-DP practice
  already does).
- The sufficient statistics of many parametric model fits (e.g. `sum(x*x)`,
  `sum(x*y)` for OLS) -- each term is still one unit's own product. Fit a
  model by releasing its sufficient statistics this way, then use the
  released (public, or `NoisyFloat`-uncertain) coefficients as ordinary
  constants downstream. Because DP is closed under post-processing, the
  matrix inverse/solve that turns statistics into coefficients needs no
  sensitivity analysis of its own.

What it does *not* cover, and cannot be made to cover by tracking more
expressions: a term's sensitivity is only well-defined here when it
depends on one unit's own bounded leaves. Calibrated/raked weights (a
unit's weight depends on how many other units share its stratum) and
model-based imputation fit directly against the private data (one unit's
value can move another unit's imputed value) both break that locality.
The sufficient-statistics route above is the intended workaround for the
imputation case: privatize the fitting inputs, then treat the fit as
post-processing rather than asking this module to bound the sensitivity of
an arbitrary fitting procedure.

Requires the `opendp` package (the project's `opendp` extra) -- actual
noise is added by `dp.m.make_gaussian`, calibrated to a target zCDP `rho`
via `dp.binary_search_param`; this module only computes the sensitivity
that calibration needs. Callers must call
`dp.enable_features("contrib", "honest-but-curious")` themselves, same as
`noisyvalue.opendp`.
"""

import math
import operator as op

import numpy as np
import sympy as sp
from sympy import sympify

try:
    import opendp.prelude as dp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "noisyvalue.private requires the 'opendp' package; install the "
        "project's [opendp] extra (e.g. `uv sync --extra opendp`)"
    ) from e

from . import util
from .graph import Name
from .opendp import SharedLatentRegistry, make_gaussian


def rho_for_epsilon_delta(epsilon, delta):
    """The rho such that rho-zCDP implies (epsilon, delta)-DP.

    Standard zCDP-to-approximate-DP conversion (Bun & Steinke 2016,
    Prop. 1.3): a rho-zCDP mechanism is
    `(rho + 2*sqrt(rho*log(1/delta)), delta)`-DP for every `delta > 0`.
    This inverts that bound for a target `epsilon`, i.e. it returns the
    rho at which the bound holds with equality -- a standard, slightly
    conservative conversion, not the tightest rho satisfying the target.
    """
    epsilon = float(epsilon)
    delta = float(delta)
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    t = math.sqrt(math.log(1.0 / delta))
    return (math.sqrt(t * t + epsilon) - t) ** 2


def _add_bounds(lo1, hi1, lo2, hi2):
    return lo1 + lo2, hi1 + hi2


def _sub_bounds(lo1, hi1, lo2, hi2):
    return lo1 - hi2, hi1 - lo2


def _mul_bounds(lo1, hi1, lo2, hi2):
    corners = (lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2)
    return min(corners), max(corners)


def _as_operand(x):
    """(value, lo, hi, units) for a PrivateNumber or a plain public number."""
    if isinstance(x, PrivateNumber):
        return x.value, x.lo, x.hi, x.units
    if isinstance(x, PrivateValue):
        raise TypeError(
            f"Expected a PrivateNumber or a plain number, got {type(x).__name__}")
    v = float(x)
    return x, v, v, frozenset()


def _is_int_like(x):
    if isinstance(x, PrivateInt):
        return True
    if isinstance(x, PrivateValue):
        return False
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _result_cls(a, b):
    return PrivateInt if _is_int_like(a) and _is_int_like(b) else PrivateFloat


class PrivateValue:
    """A value with sensitivity provenance: which PrivacyUnit(s) it traces
    back to, and (for `PrivateNumber`) a bound on how far it can range.

    `symbol` is this value's own fresh identity; `expr` is the full formula
    in terms of leaf symbols -- kept for inspection, not walked for
    sensitivity (see module docstring for why that's not attempted
    generically).
    """

    def __init__(self, value, symbol, units, expr=None):
        self.value = value
        self.symbol = symbol
        self.expr = symbol if expr is None else sympify(expr)
        self.units = frozenset(units)

    def __repr__(self):
        return f"{type(self).__name__}({self.value!r})"


class PrivateNumber(PrivateValue):
    def __init__(self, value, symbol, lo, hi, units, expr=None):
        super().__init__(value, symbol, units, expr=expr)
        lo, hi = float(lo), float(hi)
        if lo > hi:
            raise ValueError(f"Invalid bounds: lo={lo} > hi={hi}")
        self.lo = lo
        self.hi = hi

    def _bin_op(self, other, out_cls, value_op, expr_op, bound_op, rev=False):
        ov, olo, ohi, ounits = _as_operand(other)
        oexpr = other.expr if isinstance(other, PrivateValue) else sympify(other)
        lv, rv = (ov, self.value) if rev else (self.value, ov)
        llo, lhi, rlo, rhi = (
            (olo, ohi, self.lo, self.hi) if rev else (self.lo, self.hi, olo, ohi))
        le, re = (oexpr, self.expr) if rev else (self.expr, oexpr)
        value = value_op(lv, rv)
        lo, hi = bound_op(llo, lhi, rlo, rhi)
        expr = expr_op(le, re)
        return out_cls(value, sp.Symbol(Name.fresh()), lo, hi, self.units | ounits, expr=expr)

    def __add__(self, other):
        return self._bin_op(other, _result_cls(self, other), op.add, sp.Add, _add_bounds)

    def __radd__(self, other):
        return self._bin_op(other, _result_cls(self, other), op.add, sp.Add, _add_bounds, rev=True)

    def __sub__(self, other):
        return self._bin_op(other, _result_cls(self, other), op.sub, op.sub, _sub_bounds)

    def __rsub__(self, other):
        return self._bin_op(other, _result_cls(self, other), op.sub, op.sub, _sub_bounds, rev=True)

    def __mul__(self, other):
        return self._bin_op(other, _result_cls(self, other), op.mul, sp.Mul, _mul_bounds)

    def __rmul__(self, other):
        return self._bin_op(other, _result_cls(self, other), op.mul, sp.Mul, _mul_bounds, rev=True)

    def __neg__(self):
        return type(self)(-self.value, sp.Symbol(Name.fresh()), -self.hi, -self.lo,
                          self.units, expr=-self.expr)

    def clamp(self, lo, hi):
        """A copy re-bounded to the intersection of `[lo, hi]` and this
        value's own bounds, with the observed value clipped to match.

        Use this to assert a tighter public bound after a computation
        (e.g. after combining several leaves) -- the same role `dp.clamp`
        plays for an OpenDP transformation chain.
        """
        lo, hi = float(lo), float(hi)
        if lo > hi:
            raise ValueError(f"Invalid bounds: lo={lo} > hi={hi}")
        new_lo, new_hi = max(self.lo, lo), min(self.hi, hi)
        if new_lo > new_hi:
            raise ValueError(
                f"Clamp bounds [{lo}, {hi}] do not overlap this value's own "
                f"bounds [{self.lo}, {self.hi}]")
        value = min(max(self.value, new_lo), new_hi)
        expr = sp.Min(sp.Max(self.expr, sp.Float(lo)), sp.Float(hi))
        return type(self)(value, sp.Symbol(Name.fresh()), new_lo, new_hi, self.units, expr=expr)


class PrivateFloat(PrivateNumber):
    def __init__(self, value, symbol, lo, hi, units, expr=None):
        super().__init__(float(value), symbol, lo, hi, units, expr=expr)


class PrivateInt(PrivateNumber):
    def __init__(self, value, symbol, lo, hi, units, expr=None):
        super().__init__(int(value), symbol, lo, hi, units, expr=expr)


class PrivateBool(PrivateValue):
    def __init__(self, value, symbol, units, expr=None):
        super().__init__(bool(value), symbol, units, expr=expr)

    def __invert__(self):
        return PrivateBool(not self.value, sp.Symbol(Name.fresh()), self.units,
                           expr=sp.Not(self.expr))

    def as_int(self):
        """This flag as a 0/1 `PrivateInt`, e.g. to sum into a count.

        A boolean's sensitivity is the easy case: its contribution to a
        sum is bounded by 1 with no bound-tracking needed, which is why
        `PrivateBool` itself carries no `lo`/`hi` -- this is where that
        fixed [0, 1] bound enters the sensitivity machinery.
        """
        expr = sp.Piecewise((sp.Integer(1), self.expr), (sp.Integer(0), True))
        return PrivateInt(int(self.value), sp.Symbol(Name.fresh()), 0, 1, self.units, expr=expr)


class PrivacyUnit:
    """The thing DP protects, e.g. one survey respondent.

    Values contributed by a unit are created through this object so every
    leaf in a sensitivity computation is traceable to the unit whose
    removal it bounds. `lo`/`hi` are the declared bounds on the unit's
    *true* value -- the observed `value` is clipped into them, since a
    sensitivity claim about "any dataset within these bounds" is only
    honest if the released statistic was actually computed on a clipped
    value.
    """

    def __init__(self, uid):
        self.uid = uid
        self.values = []

    def private_float(self, value, lo, hi):
        return self._leaf(PrivateFloat, value, lo, hi)

    def private_int(self, value, lo, hi):
        return self._leaf(PrivateInt, value, lo, hi)

    def private_bool(self, value):
        v = PrivateBool(value, sp.Symbol(Name.fresh()), frozenset({self}))
        self.values.append(v)
        return v

    def _leaf(self, cls, value, lo, hi):
        lo, hi = float(lo), float(hi)
        if lo > hi:
            raise ValueError(f"Invalid bounds: lo={lo} > hi={hi}")
        clamped = min(max(float(value), lo), hi)
        v = cls(clamped, sp.Symbol(Name.fresh()), lo, hi, frozenset({self}))
        self.values.append(v)
        return v

    def __repr__(self):
        return f"PrivacyUnit({self.uid!r})"


def _domain_metric(discrete):
    T = int if discrete else float
    if discrete:
        return dp.atom_domain(T=T), dp.absolute_distance(T=T)
    return dp.atom_domain(T=T, nan=False), dp.absolute_distance(T=T)


class PrivateDataset:
    """A population of `PrivacyUnit`s and the DP budget spent releasing
    statistics about them.

    Budget is tracked as zCDP `rho` (composes additively across releases,
    and is what both the Gaussian and discrete-Gaussian mechanisms have
    simple, exact costs for); use `rho_for_epsilon_delta` to translate an
    `(epsilon, delta)` requirement. `release()` may be called repeatedly;
    pass the same `key` to two calls to have their releases of one
    statistic combine in closed form (see `noisyvalue.opendp
    .SharedLatentRegistry`) -- each call still spends its own `rho`.
    """

    def __init__(self, units, rho_budget=None):
        self.units = util.as_tuple(units, PrivacyUnit)
        self._unit_set = frozenset(self.units)
        self.rho_budget = None if rho_budget is None else float(rho_budget)
        self.rho_spent = 0.0
        self._registry = SharedLatentRegistry()

    def release(self, terms, *, rho, key=None):
        """Release `sum(t.value for t in terms)` under `rho`-zCDP noise.

        Each term must be a `PrivateNumber` whose `.units` is a single
        `PrivacyUnit` belonging to this dataset -- combine values within
        one unit's own contribution (via ordinary arithmetic) before
        passing them in, rather than mixing units into one term. Global
        sensitivity is the worst-case magnitude any single unit's terms
        can sum to; several terms from the same unit are added together
        for that purpose.

        Each term's `lo`/`hi` are also public bounds on that unit's own true
        contribution (see `PrivacyUnit`), so `sum(t.lo)`/`sum(t.hi)` across
        *all* terms bound the statistic's true value -- unlike sensitivity,
        which is driven by one worst-case unit. That sum truncates the
        released posterior the same way a known-nonnegative count already
        does elsewhere in this library, tightening it whenever the bound
        falls inside the noise's informative range.
        """
        terms = list(terms)
        if not terms:
            raise ValueError("release() needs at least one term")
        if rho <= 0.0:
            raise ValueError(f"rho must be positive, got {rho}")

        per_unit = {}
        discrete = True
        total_lo = total_hi = 0.0
        for t in terms:
            if not isinstance(t, PrivateNumber):
                raise TypeError(
                    f"release() terms must be PrivateNumber, got {type(t).__name__}")
            if len(t.units) != 1:
                raise ValueError(
                    f"{t!r} traces back to {len(t.units)} privacy units "
                    f"({sorted(u.uid for u in t.units)}); release() can only bound "
                    "a term's sensitivity when it belongs to exactly one unit -- "
                    "combine values within one unit's own contribution before "
                    "summing across units")
            (unit,) = t.units
            if unit not in self._unit_set:
                raise ValueError(f"{unit!r} is not a member of this PrivateDataset")
            discrete = discrete and isinstance(t, PrivateInt)
            lo, hi = per_unit.get(unit, (0.0, 0.0))
            per_unit[unit] = (lo + t.lo, hi + t.hi)
            total_lo += t.lo
            total_hi += t.hi

        sensitivity = max(max(abs(lo), abs(hi)) for lo, hi in per_unit.values())
        if sensitivity <= 0.0:
            raise ValueError("Sensitivity is zero; nothing to protect by adding noise")

        self._check_budget(rho)

        nominal = sum(t.value for t in terms)
        if not discrete:
            nominal = float(nominal)

        domain, metric = _domain_metric(discrete)
        # The metric's distance type follows the domain (an integer distance
        # for an integer domain); the search parameter (the noise scale) is
        # always a float regardless. Rounding the sensitivity up keeps the
        # integer distance a sound (if slightly conservative) bound.
        d_in = math.ceil(sensitivity) if discrete else sensitivity
        scale = dp.binary_search_param(
            lambda s: dp.m.make_gaussian(domain, metric, scale=s),
            d_in=d_in, d_out=rho, T=float)
        meas = make_gaussian(domain, metric, scale, key=key, registry=self._registry,
                             low=total_lo, high=total_hi)
        value = meas(nominal)

        self.rho_spent += rho
        return value

    def _check_budget(self, rho):
        if self.rho_budget is not None and self.rho_spent + rho > self.rho_budget + 1e-9:
            raise ValueError(
                f"Release would spend rho={rho:.6g}, but only "
                f"{self.rho_budget - self.rho_spent:.6g} of the "
                f"{self.rho_budget:.6g} rho budget remains")
