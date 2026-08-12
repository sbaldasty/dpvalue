import numpy as np
import pandas as pd
import pytest

import conftest as noise
from noisyvalue.core import NoisyBool, NoisyFloat, NoisyInt, sample_noisy_values
from noisyvalue.pandas import NoisyBoolArray, NoisyFloatArray, NoisyIntArray


def _series(values):
    return pd.Series(NoisyFloatArray._from_sequence(values))


def _int_series(values):
    return pd.Series(NoisyIntArray._from_sequence(values))


def _bool_series(values):
    return pd.Series(NoisyBoolArray._from_sequence(values))


# ── dtype plumbing ─────────────────────────────────────────────────────────

def test_series_reports_noisyfloat_dtype():
    s = _series([NoisyFloat.draw(1.0, noise.gaussian(0, 1), rng=0)])
    assert s.dtype.name == "noisyfloat"


# ── comparisons: real, vectorized bool, not object-dtype NoisyBool ─────────

def test_comparison_returns_real_bool_mask_not_object_dtype():
    vals = [NoisyFloat.lift(1.0), NoisyFloat.lift(5.0), NoisyFloat.lift(3.0)]
    s = _series(vals)
    mask = s < 3.0
    assert pd.api.types.is_bool_dtype(mask)
    assert mask.tolist() == [True, False, False]


def test_filtering_with_comparison_mask_works():
    vals = [NoisyFloat.lift(1.0), NoisyFloat.lift(5.0), NoisyFloat.lift(3.0)]
    s = _series(vals)
    filtered = s[s < 3.0]
    assert len(filtered) == 1
    assert float(filtered.iloc[0]) == 1.0


# ── arithmetic: preserves the symbolic posterior, matches scalar bin_op ────

def test_addition_matches_scalar_bin_op_and_preserves_joint_structure():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    b = a * 2.0  # shares a's latent/noise node

    s = _series([a])
    result_series = s + _series([b])
    result = result_series.array[0]

    assert isinstance(result, NoisyFloat)
    assert result._obs == pytest.approx(a._obs + b._obs)

    # Shared-latent correlation must survive: result == 3*a for every draw.
    draws_result, draws_a = sample_noisy_values(result, a, n=500, rng=2)
    assert np.allclose(draws_result.draws, 3.0 * draws_a.draws)


def test_addition_with_scalar_lifts_correctly():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    s = _series([a])
    result = (s + 10.0).array[0]
    assert result._obs == pytest.approx(a._obs + 10.0)


# ── reductions: sum/mean preserve posterior; others fail loudly ────────────

def test_sum_returns_noisyfloat_with_correlated_posterior():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    b = NoisyFloat.draw(3.0, noise.gaussian(0, 1), rng=2)
    s = _series([a, b])

    total = s.sum()
    assert isinstance(total, NoisyFloat)
    assert total._obs == pytest.approx(a._obs + b._obs)

    draws_total, draws_a, draws_b = sample_noisy_values(total, a, b, n=500, rng=3)
    assert np.allclose(draws_total.draws, draws_a.draws + draws_b.draws)


def test_mean_returns_noisyfloat_not_plain_float():
    a = NoisyFloat.draw(2.0, noise.gaussian(0, 1), rng=1)
    b = NoisyFloat.draw(4.0, noise.gaussian(0, 1), rng=2)
    s = _series([a, b])

    result = s.mean()
    assert isinstance(result, NoisyFloat)
    assert result._obs == pytest.approx((a._obs + b._obs) / 2)


@pytest.mark.parametrize("reduction", ["median", "std", "var"])
def test_unsupported_reductions_raise_clearly(reduction):
    a = NoisyFloat.draw(2.0, noise.gaussian(0, 1), rng=1)
    b = NoisyFloat.draw(4.0, noise.gaussian(0, 1), rng=2)
    s = _series([a, b])

    with pytest.raises(TypeError, match="does not support reduction"):
        getattr(s, reduction)()


# ── intentionally-left-broken: unique/groupby/isin/drop_duplicates ─────────

def test_unique_still_fails_by_design():
    s = _series([NoisyFloat.lift(1.0), NoisyFloat.lift(1.0)])
    with pytest.raises(Exception):
        s.unique()


# ── NA handling ─────────────────────────────────────────────────────────────

def test_none_in_sequence_is_missing():
    s = _series([NoisyFloat.lift(1.0), None, NoisyFloat.lift(3.0)])
    assert s.isna().tolist() == [False, True, False]
    assert s.iloc[1] is pd.NA


def test_nan_scalar_in_sequence_is_missing_but_wrapped_nan_obs_is_not():
    wrapped_nan = NoisyFloat.lift(float("nan"))
    s = _series([wrapped_nan, float("nan"), 2.0])
    assert s.isna().tolist() == [False, True, False]


def test_arithmetic_propagates_missing():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    s = _series([a, None])
    result = s + 1.0
    assert result.isna().tolist() == [False, True]
    assert result.iloc[1] is pd.NA
    assert float(result.iloc[0]) == pytest.approx(a._obs + 1.0)


def test_comparison_against_missing_is_na_not_true_or_false():
    s = _series([NoisyFloat.lift(1.0), None])
    mask = s < 5.0
    assert bool(mask.iloc[0]) is True
    assert pd.isna(mask.iloc[1])


def test_sum_skipna_true_ignores_missing():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    b = NoisyFloat.draw(3.0, noise.gaussian(0, 1), rng=2)
    s = _series([a, None, b])
    total = s.sum()  # skipna=True is the pandas default
    assert isinstance(total, NoisyFloat)
    assert total._obs == pytest.approx(a._obs + b._obs)


def test_sum_skipna_false_with_missing_returns_na():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    s = _series([a, None])
    assert s.sum(skipna=False) is pd.NA


def test_sum_of_all_missing_returns_na():
    s = _series([None, None])
    assert s.sum() is pd.NA


def test_take_with_fill_marks_missing():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    b = NoisyFloat.draw(3.0, noise.gaussian(0, 1), rng=2)
    arr = NoisyFloatArray._from_sequence([a, b])
    taken = arr.take([0, -1, 1], allow_fill=True, fill_value=None)
    assert taken.isna().tolist() == [False, True, False]


def test_concat_preserves_mask():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    left = NoisyFloatArray._from_sequence([a, None])
    right = NoisyFloatArray._from_sequence([None, a])
    combined = NoisyFloatArray._concat_same_type([left, right])
    assert combined.isna().tolist() == [False, True, True, False]


# ── NoisyInt: shares _NumericOpsMixin with NoisyFloat ──────────────────────

def test_int_series_reports_noisyint_dtype():
    s = _int_series([NoisyInt.lift(1)])
    assert s.dtype.name == "noisyint"


def test_int_comparison_returns_real_bool():
    s = _int_series([NoisyInt.lift(1), NoisyInt.lift(5), NoisyInt.lift(3)])
    mask = s < 3
    assert pd.api.types.is_bool_dtype(mask)
    assert mask.tolist() == [True, False, False]


def test_int_arithmetic_promotes_to_noisyfloat_like_the_scalar_does():
    a = NoisyInt.binomial(100, 0.3, obs=40)
    b = NoisyInt.binomial(100, 0.3, obs=25)
    s = _int_series([a, b])
    result = (s + s).array[0]
    # NoisyNumber.bin_op always promotes to NoisyFloat, even Int + Int.
    assert isinstance(result, NoisyFloat)
    assert result._obs == pytest.approx(a._obs + a._obs)


def test_int_plus_float_series_cross_type_works():
    a = NoisyInt.lift(2)
    b = NoisyFloat.draw(3.0, noise.gaussian(0, 1), rng=1)
    result = (_int_series([a]) + _series([b])).array[0]
    assert isinstance(result, NoisyFloat)
    assert result._obs == pytest.approx(2.0 + b._obs)


def test_int_sum_preserves_joint_posterior():
    a = NoisyInt.binomial(100, 0.3, obs=40)
    b = NoisyInt.binomial(100, 0.4, obs=45)
    s = _int_series([a, b])
    total = s.sum()
    assert isinstance(total, NoisyFloat)

    draws_total, draws_a, draws_b = sample_noisy_values(total, a, b, n=500, rng=3)
    assert np.allclose(draws_total.draws, draws_a.draws + draws_b.draws)


def test_int_none_is_missing_and_sum_skipna_ignores_it():
    a = NoisyInt.lift(2)
    b = NoisyInt.lift(5)
    s = _int_series([a, None, b])
    assert s.isna().tolist() == [False, True, False]
    total = s.sum()
    assert isinstance(total, NoisyFloat)
    assert total._obs == pytest.approx(7.0)


# ── NoisyBool: separate logical-ops implementation ─────────────────────────

def test_bool_series_reports_noisybool_dtype():
    s = _bool_series([NoisyBool.lift(True)])
    assert s.dtype.name == "noisybool"


def test_and_or_preserve_shared_latent_correlation():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    gt0 = a > 0.0
    lt10 = a < 10.0

    s_gt0 = _bool_series([gt0])
    s_lt10 = _bool_series([lt10])
    in_range = (s_gt0 & s_lt10).array[0]
    out_of_range = (~s_gt0 | ~s_lt10).array[0]

    assert isinstance(in_range, NoisyBool)
    draws_in_range, draws_out, draws_a = sample_noisy_values(
        in_range, out_of_range, a, n=500, rng=2
    )
    expected = (draws_a.draws > 0.0) & (draws_a.draws < 10.0)
    assert np.array_equal(draws_in_range.draws, expected)
    assert np.array_equal(draws_out.draws, ~expected)


def test_bool_and_with_plain_literal():
    s = _bool_series([NoisyBool.lift(True), NoisyBool.lift(False)])
    result = (s & True).array
    assert [bool(result[0]), bool(result[1])] == [True, False]


def test_bool_xor_is_unsupported():
    s = _bool_series([NoisyBool.lift(True)])
    with pytest.raises(TypeError, match="does not support"):
        s ^ True


def test_any_all_preserve_posterior():
    a = NoisyFloat.draw(5.0, noise.gaussian(0, 1), rng=1)
    gt0 = a > 0.0
    gt100 = a > 100.0
    s = _bool_series([gt0, gt100])

    any_result = s.any()
    all_result = s.all()
    assert isinstance(any_result, NoisyBool)
    assert isinstance(all_result, NoisyBool)

    draws_any, draws_all, draws_a = sample_noisy_values(any_result, all_result, a, n=500, rng=2)
    assert np.array_equal(draws_any.draws, (draws_a.draws > 0.0) | (draws_a.draws > 100.0))
    assert np.array_equal(draws_all.draws, (draws_a.draws > 0.0) & (draws_a.draws > 100.0))


def test_bool_none_is_missing():
    s = _bool_series([NoisyBool.lift(True), None])
    assert s.isna().tolist() == [False, True]
    assert s.iloc[1] is pd.NA


def test_bool_any_skipna_true_ignores_missing():
    a = NoisyBool.lift(True)
    s = _bool_series([a, None])
    result = s.any()
    assert isinstance(result, NoisyBool)
    assert bool(result) is True


def test_bool_any_skipna_false_with_missing_returns_na():
    a = NoisyBool.lift(True)
    s = _bool_series([a, None])
    assert s.any(skipna=False) is pd.NA
