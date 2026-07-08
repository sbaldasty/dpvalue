"""Pandas ExtensionArray/ExtensionDtype for NoisyFloat/NoisyInt/NoisyBool columns.

Storing these values in a plain object-dtype column leaves pandas to loop
elementwise over scalar dunder methods for every operation, which is how
comparisons end up silently collapsing to `bool(obs)` and `.mean()` silently
collapses to a plain float. This module gives each type its own dtype so
comparisons/reductions are implemented deliberately instead of falling out of
pandas' generic object-dtype fallback.

Missing values are represented with a mask array (like pandas' own nullable
IntegerArray/BooleanArray), not with a sentinel-in-obs, so "no observation"
stays distinguishable from "observation happens to be NaN/0/False".

`unique`/`factorize`/`isin`/`groupby`-by-a-noisy-column are intentionally left
unsupported for all three types (none of them have a real equality/hash, only
a symbolic one).

Shared structure: `_NoisyArrayBase` holds the obs/roots/mask storage and all
the plumbing pandas requires (construction, indexing, take, concat, ...).
`_NumericOpsMixin` adds comparisons/arithmetic/reductions shared by
NoisyFloatArray and NoisyIntArray, since NoisyInt arithmetic already promotes
to NoisyFloat at the scalar level (see NoisyNumber.bin_op) — the mixin just
makes that promotion vectorized too. NoisyBoolArray isn't a NoisyNumber, so it
gets its own logical (`&`/`|`/`~`) and any/all reduction implementation.
"""

import numbers

import numpy as np
import pandas as pd
import sympy as sp
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype
from pandas.core.arraylike import OpsMixin
from sympy import And, Or, Not

from .core import NoisyBool, NoisyFloat, NoisyInt, NoisyNumber
from .graph import DerivedNode

_NUMERIC_REDUCTIONS = frozenset({"sum", "mean"})
_LOGICAL_REDUCTIONS = frozenset({"any", "all"})


def _is_na_scalar(v):
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))


# ── shared storage + pandas plumbing ────────────────────────────────────────

class _NoisyArrayBase(OpsMixin, ExtensionArray):
    """obs/roots/mask storage shared by all Noisy*Array types.

    Subclasses set `_scalar_cls` (the NoisyValue subclass this array holds),
    `_obs_dtype` (the numpy dtype backing `_obs`), and `_na_placeholder` (the
    obs value stored at masked-out positions; never read except as filler).
    """

    _scalar_cls = None
    _obs_dtype = None
    _na_placeholder = None

    def __init__(self, obs, roots, mask=None, copy=False):
        obs = np.array(obs, dtype=self._obs_dtype, copy=copy)
        roots = np.array(roots, dtype=object, copy=copy)
        mask = np.zeros(obs.shape, dtype=bool) if mask is None else np.array(mask, dtype=bool, copy=copy)
        if obs.shape != roots.shape or obs.shape != mask.shape:
            raise ValueError("obs, roots, and mask must have the same shape")
        self._obs = obs
        self._roots = roots
        self._mask = mask

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        n = len(scalars)
        obs = np.empty(n, dtype=cls._obs_dtype)
        roots = np.empty(n, dtype=object)
        mask = np.zeros(n, dtype=bool)
        for i, v in enumerate(scalars):
            if isinstance(v, cls._scalar_cls):
                obs[i] = v._obs
                roots[i] = v._root
            elif _is_na_scalar(v):
                obs[i] = cls._na_placeholder
                roots[i] = None
                mask[i] = True
            else:
                lifted = cls._scalar_cls.lift(v)
                obs[i] = lifted._obs
                roots[i] = lifted._root
        return cls(obs, roots, mask, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        raise NotImplementedError(
            f"{cls.__name__} does not support factorize/unique/isin/"
            "groupby-by-noisy-column (left unsupported by design)"
        )

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return len(self._obs)

    @property
    def nbytes(self):
        return self._obs.nbytes + self._roots.nbytes + self._mask.nbytes

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            if self._mask[item]:
                return pd.NA
            return self._scalar_cls(self._obs[item], self._roots[item])
        item = pd.api.indexers.check_array_indexer(self, item)
        return type(self)(self._obs[item], self._roots[item], self._mask[item])

    def isna(self):
        return self._mask.copy()

    def take(self, indices, allow_fill=False, fill_value=None):
        indices = np.asarray(indices, dtype=np.intp)
        if not allow_fill:
            return type(self)(self._obs[indices], self._roots[indices], self._mask[indices])

        missing_slot = indices == -1
        if fill_value is None or _is_na_scalar(fill_value):
            fill_is_na, fill_obs, fill_root = True, self._na_placeholder, None
        else:
            fill_value = self._scalar_cls.lift(fill_value)
            fill_is_na, fill_obs, fill_root = False, fill_value._obs, fill_value._root

        safe_indices = np.where(missing_slot, 0, indices)
        obs = np.where(missing_slot, fill_obs, self._obs[safe_indices])
        mask = np.where(missing_slot, fill_is_na, self._mask[safe_indices])
        roots = np.array([
            fill_root if slot_missing else self._roots[i]
            for i, slot_missing in zip(safe_indices, missing_slot)
        ], dtype=object)
        return type(self)(obs, roots, mask)

    def copy(self):
        return type(self)(self._obs.copy(), self._roots.copy(), self._mask.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        obs = np.concatenate([x._obs for x in to_concat])
        roots = np.concatenate([x._roots for x in to_concat])
        mask = np.concatenate([x._mask for x in to_concat])
        return cls(obs, roots, mask)

    def __repr__(self):
        shown = ["<NA>" if m else o for o, m in zip(self._obs, self._mask)]
        return f"{type(self).__name__}({shown!r})"


# ── shared numeric behavior (NoisyFloatArray, NoisyIntArray) ───────────────

class _NumericOpsMixin:
    """Comparisons/arithmetic/reductions shared by NoisyNumber-backed arrays.

    Arithmetic always produces a NoisyFloatArray, matching NoisyNumber.bin_op
    at the scalar level (NoisyInt + NoisyInt is a NoisyFloat there too).
    """

    def _cmp_method(self, other, op):
        other_obs, other_mask = self._coerce_cmp_operand(other)
        if other_obs is NotImplemented:
            return NotImplemented
        with np.errstate(invalid="ignore"):
            result = np.asarray(op(self._obs, other_obs), dtype=bool)
        combined_mask = self._mask | other_mask
        return pd.arrays.BooleanArray(result, combined_mask)

    def _coerce_cmp_operand(self, other):
        if isinstance(other, _NumericOpsMixin):
            return other._obs, other._mask
        if isinstance(other, NoisyNumber):
            return other._obs, np.zeros(len(self), dtype=bool)
        if _is_na_scalar(other):
            return np.full(len(self), np.nan), np.ones(len(self), dtype=bool)
        if isinstance(other, numbers.Real):
            return other, np.zeros(len(self), dtype=bool)
        return NotImplemented, None

    def _arith_method(self, other, op):
        other_obs, other_roots, other_mask = self._coerce_operand(other)
        if other_obs is NotImplemented:
            return NotImplemented

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            new_obs = op(self._obs.astype("float64"), np.asarray(other_obs, dtype="float64"))

        result_mask = self._mask | other_mask
        new_roots = np.array([
            None if is_missing else DerivedNode.operational(op(l.expr, r.expr), deps=[l, r])
            for l, r, is_missing in zip(self._roots, other_roots, result_mask)
        ], dtype=object)
        return NoisyFloatArray(new_obs, new_roots, result_mask)

    def _coerce_operand(self, other):
        if isinstance(other, _NumericOpsMixin):
            return other._obs, other._roots, other._mask
        if isinstance(other, NoisyNumber):
            return (
                np.full(len(self), other._obs),
                np.full(len(self), other._root, dtype=object),
                np.zeros(len(self), dtype=bool),
            )
        if _is_na_scalar(other):
            return (
                np.full(len(self), np.nan),
                np.full(len(self), None, dtype=object),
                np.ones(len(self), dtype=bool),
            )
        if isinstance(other, numbers.Real):
            literal_root = DerivedNode(sp.sympify(other))
            return (
                np.full(len(self), float(other)),
                np.full(len(self), literal_root, dtype=object),
                np.zeros(len(self), dtype=bool),
            )
        return NotImplemented, None, None

    def _reduce(self, name, *, skipna=True, keepdims=False, **kwargs):
        if name not in _NUMERIC_REDUCTIONS:
            raise TypeError(
                f"{type(self).__name__} does not support reduction {name!r} — "
                "there's no well-defined posterior combination for it "
                "(only 'sum' and 'mean' are supported)"
            )

        if not skipna and self._mask.any():
            return self._na_reduce_result(keepdims)

        roots = [r for r, m in zip(self._roots, self._mask) if not m]
        obs = self._obs[~self._mask]
        if len(roots) == 0:
            return self._na_reduce_result(keepdims)

        total_expr = sp.Add(*(r.expr for r in roots))
        root = DerivedNode.operational(total_expr, deps=roots)
        result = NoisyFloat(float(np.sum(obs)), root)
        if name == "mean":
            result = result / len(roots)

        if keepdims:
            return NoisyFloatArray._from_sequence([result])
        return result

    def _na_reduce_result(self, keepdims):
        if keepdims:
            return NoisyFloatArray._from_sequence([pd.NA])
        return pd.NA


# ── NoisyFloat ───────────────────────────────────────────────────────────────

@register_extension_dtype
class NoisyFloatDtype(ExtensionDtype):
    name = "noisyfloat"
    type = NoisyFloat
    kind = "f"
    na_value = pd.NA

    @classmethod
    def construct_array_type(cls):
        return NoisyFloatArray

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        raise TypeError(f"Cannot construct a 'NoisyFloatDtype' from '{string}'")


class NoisyFloatArray(_NumericOpsMixin, _NoisyArrayBase):
    _scalar_cls = NoisyFloat
    _obs_dtype = "float64"
    _na_placeholder = np.nan
    _dtype = NoisyFloatDtype()


# ── NoisyInt ─────────────────────────────────────────────────────────────────

@register_extension_dtype
class NoisyIntDtype(ExtensionDtype):
    name = "noisyint"
    type = NoisyInt
    kind = "i"
    na_value = pd.NA

    @classmethod
    def construct_array_type(cls):
        return NoisyIntArray

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        raise TypeError(f"Cannot construct a 'NoisyIntDtype' from '{string}'")


class NoisyIntArray(_NumericOpsMixin, _NoisyArrayBase):
    _scalar_cls = NoisyInt
    _obs_dtype = "int64"
    _na_placeholder = 0
    _dtype = NoisyIntDtype()


# ── NoisyBool ────────────────────────────────────────────────────────────────

@register_extension_dtype
class NoisyBoolDtype(ExtensionDtype):
    name = "noisybool"
    type = NoisyBool
    kind = "b"
    na_value = pd.NA

    @classmethod
    def construct_array_type(cls):
        return NoisyBoolArray

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        raise TypeError(f"Cannot construct a 'NoisyBoolDtype' from '{string}'")


_BOOL_LOGICAL_OPS = {
    # operator name -> (obs op, expr op), mirroring NoisyBool's own bin_op calls.
    # and/or are commutative, so the reflected variants (rand_/ror_, used for
    # e.g. `True & bool_series`) reuse the same forward computation.
    "and_": (np.logical_and, And),
    "rand_": (np.logical_and, And),
    "or_": (np.logical_or, Or),
    "ror_": (np.logical_or, Or),
}


class NoisyBoolArray(_NoisyArrayBase):
    """`&`/`|`/`~` and any()/all() for NoisyBool columns.

    NoisyBool doesn't define comparisons or `^` at the scalar level (core.py
    only gives it __and__/__or__/__invert__), so this array doesn't either —
    `_logical_method` only handles `and_`/`or_`; xor and comparisons raise.
    """

    _scalar_cls = NoisyBool
    _obs_dtype = "bool"
    _na_placeholder = False
    _dtype = NoisyBoolDtype()

    def _logical_method(self, other, op):
        key = getattr(op, "__name__", None)
        if key not in _BOOL_LOGICAL_OPS:
            raise TypeError(f"NoisyBoolArray does not support {key!r} (only 'and_'/'or_' are supported)")
        obs_op, expr_op = _BOOL_LOGICAL_OPS[key]

        other_obs, other_roots, other_mask = self._coerce_operand(other)
        if other_obs is NotImplemented:
            return NotImplemented

        new_obs = obs_op(self._obs, other_obs)
        result_mask = self._mask | other_mask
        new_roots = np.array([
            None if is_missing else DerivedNode.operational(expr_op(l.expr, r.expr), deps=[l, r])
            for l, r, is_missing in zip(self._roots, other_roots, result_mask)
        ], dtype=object)
        return NoisyBoolArray(new_obs, new_roots, result_mask)

    def _coerce_operand(self, other):
        if isinstance(other, NoisyBoolArray):
            return other._obs, other._roots, other._mask
        if isinstance(other, NoisyBool):
            return (
                np.full(len(self), other._obs),
                np.full(len(self), other._root, dtype=object),
                np.zeros(len(self), dtype=bool),
            )
        if _is_na_scalar(other):
            return (
                np.zeros(len(self), dtype=bool),
                np.full(len(self), None, dtype=object),
                np.ones(len(self), dtype=bool),
            )
        if isinstance(other, (bool, np.bool_)):
            literal_root = DerivedNode(sp.sympify(bool(other)))
            return (
                np.full(len(self), bool(other)),
                np.full(len(self), literal_root, dtype=object),
                np.zeros(len(self), dtype=bool),
            )
        return NotImplemented, None, None

    def __invert__(self):
        new_obs = ~self._obs
        new_roots = np.array([
            None if m else DerivedNode.operational(Not(r.expr), deps=[r])
            for r, m in zip(self._roots, self._mask)
        ], dtype=object)
        return NoisyBoolArray(new_obs, new_roots, self._mask.copy())

    def _reduce(self, name, *, skipna=True, keepdims=False, **kwargs):
        if name not in _LOGICAL_REDUCTIONS:
            raise TypeError(
                f"NoisyBoolArray does not support reduction {name!r} — "
                "there's no well-defined posterior combination for it "
                "(only 'any' and 'all' are supported)"
            )

        if not skipna and self._mask.any():
            return self._na_reduce_result(keepdims)

        roots = [r for r, m in zip(self._roots, self._mask) if not m]
        obs = self._obs[~self._mask]
        if len(roots) == 0:
            return self._na_reduce_result(keepdims)

        combine = Or if name == "any" else And
        obs_result = bool(np.any(obs)) if name == "any" else bool(np.all(obs))
        combined_expr = combine(*(r.expr for r in roots))
        root = DerivedNode.operational(combined_expr, deps=roots)
        result = NoisyBool(obs_result, root)

        if keepdims:
            return NoisyBoolArray._from_sequence([result])
        return result

    def _na_reduce_result(self, keepdims):
        if keepdims:
            return NoisyBoolArray._from_sequence([pd.NA])
        return pd.NA
