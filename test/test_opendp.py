import math

import numpy as np
import pytest

dp = pytest.importorskip("opendp.prelude")
dp.enable_features("contrib", "honest-but-curious")

from noisyvalue import opendp as nvdp
from noisyvalue.core import NoisyFloat, NoisyInt, sample_noisy_values


def float_space():
    return dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)


def int_space():
    return dp.atom_domain(T=int), dp.absolute_distance(T=int)


def int_vector_space(metric=dp.l2_distance):
    return dp.vector_domain(dp.atom_domain(T=int)), metric(T=int)


def test_scalar_gaussian_release_matches_posterior():
    meas = nvdp.make_gaussian(*float_space(), scale=2.0)
    value = meas(10.0)

    assert isinstance(value, NoisyFloat)
    draws = value.sample(n=8000, rng=np.random.default_rng(0)).draws
    assert abs(draws.mean() - float(value)) < 0.1
    assert abs(draws.std() - 2.0) < 0.2


def test_wrapped_measurement_preserves_privacy_map():
    wrapped = nvdp.make_gaussian(*float_space(), scale=2.0)
    inner = dp.m.make_gaussian(*float_space(), scale=2.0)
    assert wrapped.map(1.0) == inner.map(1.0)


def test_vector_integer_laplace_releases_noisy_ints():
    meas = nvdp.make_laplace(*int_vector_space(dp.l1_distance), scale=3.0)
    values = meas([10, 20, 30])

    assert len(values) == 3
    assert all(isinstance(v, NoisyInt) for v in values)
    sd = math.sqrt(nvdp._release_variance("laplace", True, 3.0))
    draws = values[0].sample(n=8000, rng=np.random.default_rng(0)).draws
    assert abs(draws.mean() - int(values[0])) < 0.2 * sd
    assert abs(draws.std() - sd) < 0.15 * sd


def test_chains_with_transformation():
    trans = dp.t.make_sum(
        dp.vector_domain(dp.atom_domain(bounds=(0.0, 1.0), nan=False)),
        dp.symmetric_distance())
    meas = trans >> nvdp.make_gaussian(*float_space(), scale=1.0)

    value = meas([0.5, 0.25, 1.0])
    assert isinstance(value, NoisyFloat)
    assert abs(float(value) - 1.75) < 10.0


def test_shared_key_combines_gaussian_measurements():
    registry = nvdp.SharedLatentRegistry()
    meas = nvdp.make_gaussian(*float_space(), scale=2.0,
                              key="stat", registry=registry)
    first = meas(10.0)
    second = meas(10.0)

    center, variance = registry.posterior("stat")
    assert center == pytest.approx((float(first) + float(second)) / 2)
    assert variance == pytest.approx(2.0)

    # The first value tightened in place, and both values are one random
    # variable to the joint sampler.
    a, b = sample_noisy_values(first, second, n=4000,
                               rng=np.random.default_rng(0))
    np.testing.assert_array_equal(a.draws, b.draws)
    assert abs(a.draws.mean() - center) < 0.1
    assert abs(a.draws.std() - math.sqrt(2.0)) < 0.15


def test_histogram_total_roundtrip_tightens_total():
    registry = nvdp.SharedLatentRegistry()
    cells_meas = nvdp.make_gaussian(*int_vector_space(), scale=5.0,
                                    keys=("a", "b", "c"), registry=registry)
    total_meas = nvdp.make_gaussian(*int_space(), scale=5.0,
                                    key="total", registry=registry)

    cells = cells_meas([40, 50, 60])
    total = total_meas(150)
    direct_center, direct_variance = registry.posterior("total")
    assert direct_variance == pytest.approx(25.0)

    registry.add_sum_estimate("total", ("a", "b", "c"))

    center, variance = registry.posterior("total")
    expected_variance = 1.0 / (1.0 / 25.0 + 1.0 / 75.0)
    expected_center = expected_variance * (
        float(total) / 25.0 + sum(float(c) for c in cells) / 75.0)
    assert variance == pytest.approx(expected_variance)
    assert center == pytest.approx(expected_center)
    assert variance < direct_variance

    # The value released before the fold reflects the tightened posterior.
    draws = total.sample(n=8000, rng=np.random.default_rng(0)).draws
    assert abs(draws.mean() - center) < 0.2
    assert abs(draws.std() - math.sqrt(variance)) < 0.15 * math.sqrt(variance)


def test_repeated_laplace_key_is_refused():
    registry = nvdp.SharedLatentRegistry()
    meas = nvdp.make_laplace(*int_space(), scale=3.0,
                             key="stat", registry=registry)
    meas(10)
    # OpenDP surfaces exceptions from user-defined functions as its own type.
    from opendp.mod import OpenDPException
    with pytest.raises(OpenDPException, match="Gaussian family"):
        meas(10)


def test_key_reuse_across_mechanisms_is_refused():
    registry = nvdp.SharedLatentRegistry()
    registry.observe("stat", 10.0, mechanism="gaussian", scale=2.0)
    with pytest.raises(ValueError):
        registry.observe("stat", 10.0, mechanism="laplace", scale=2.0)


def test_keys_without_registry_are_refused():
    with pytest.raises(ValueError):
        nvdp.make_gaussian(*float_space(), scale=2.0, key="stat")


def test_current_view_of_unmeasured_key_is_refused():
    with pytest.raises(KeyError):
        nvdp.SharedLatentRegistry().current("stat")


def test_observe_release_infers_discreteness():
    v_int = nvdp.observe_release(7, mechanism="gaussian", scale=2.0)
    v_float = nvdp.observe_release(7.0, mechanism="gaussian", scale=2.0)
    assert isinstance(v_int, NoisyInt)
    assert isinstance(v_float, NoisyFloat)
