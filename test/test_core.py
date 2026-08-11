import numpy as np
import pytest
import conftest as noise
import sympy as sp

from numpy.random import default_rng
from conftest import rooted_float
from conftest import from_node
from noisyvalue.core import NoisyFloat
from noisyvalue.core import NoisyInt
from noisyvalue.core import consolidate
from noisyvalue.core import noisy_value_sampler
from noisyvalue.core import sample_noisy_values
from noisyvalue.graph import DerivedNode
from noisyvalue.graph import LatentNode
from noisyvalue.graph import Node
from noisyvalue.graph import NormalNode
from noisyvalue.graph import NoiseNode
from noisyvalue.graph import DiscreteGaussianNode
from noisyvalue.graph import DiscreteLaplaceNode


_rng_factory = lambda: default_rng(42)


def test_joint_sampling_preserves_shared_latent_dependency():
    theta_node = LatentNode()
    theta = theta_node.expr
    eps_obs_node = noise.gaussian(0, 1)
    eps_obs = eps_obs_node.expr

    constraints = [theta + eps_obs - 1.0]
    noisy_a = rooted_float(obs=0.0, expr=theta, eqns=constraints, deps=(theta_node, eps_obs_node))
    noisy_b = rooted_float(obs=0.0, expr=2.0 * theta, eqns=constraints, deps=(theta_node, eps_obs_node))

    batch_a, batch_b = sample_noisy_values(noisy_a, noisy_b, n=2000, rng=123)
    draws_a = batch_a.draws
    draws_b = batch_b.draws

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_literal_conversions_use_native_root_model():
    converted = NoisyFloat.lift(7.0)

    assert isinstance(converted._root, DerivedNode)
    assert converted._root.constraints == frozenset()
    assert float(converted) == 7.0


def test_integer_noisy_values_sample_as_integers():
    count = NoisyInt.lift(3)
    draws = sample_noisy_values(count, n=5, rng=123)[0].draws

    assert isinstance(draws, np.ndarray)
    assert draws.dtype == int
    assert np.all(draws == 3)


def test_prepared_sampler_preserves_shared_latent_dependency():
    theta_node = LatentNode()
    theta = theta_node.expr
    eps_obs_node = noise.gaussian(0, 1)
    eps_obs = eps_obs_node.expr

    constraints = [theta + eps_obs - 1.0]
    noisy_a = rooted_float(obs=0.0, expr=theta, eqns=constraints, deps=(theta_node, eps_obs_node))
    noisy_b = rooted_float(obs=0.0, expr=2.0 * theta, eqns=constraints, deps=(theta_node, eps_obs_node))

    prepared = noisy_value_sampler(noisy_a, noisy_b)
    batch_a, batch_b = prepared.sample(n=2000, rng=123)
    draws_a = batch_a.draws
    draws_b = batch_b.draws

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_prepared_sampler_matches_direct_sampling_for_same_seed():
    theta_node = LatentNode()
    theta = theta_node.expr
    eps_obs_node = noise.gaussian(0, 1)
    eps_obs = eps_obs_node.expr

    constraints = [theta + eps_obs - 1.0]
    noisy_a = rooted_float(obs=0.0, expr=theta, eqns=constraints, deps=(theta_node, eps_obs_node))
    noisy_b = rooted_float(obs=0.0, expr=theta + 3.0, eqns=constraints, deps=(theta_node, eps_obs_node))

    direct_a, direct_b = sample_noisy_values(noisy_a, noisy_b, n=250, rng=777)
    prepared = noisy_value_sampler(noisy_a, noisy_b)
    prepared_a, prepared_b = prepared.sample(n=250, rng=777)

    assert np.allclose(prepared_a.draws, direct_a.draws)
    assert np.allclose(prepared_b.draws, direct_b.draws)


def test_node_constructors_reject_unknown_kwargs():
    with pytest.raises(TypeError):
        LatentNode(symbol=sp.Symbol("theta_node"))


def test_sampler_uses_root_constraints():
    theta_node = LatentNode()
    theta = theta_node.expr
    root = DerivedNode(
        expr=theta,
        constraints=(theta - 3.5,),
        deps=(theta_node,),
    )

    value = from_node(obs=3.5, root=root, expr=theta)
    draws = value.sample(n=6, rng=123).draws

    assert draws.shape == (6,)
    assert np.all(draws == 3.5)


def test_noisyfloat_round_nearest_for_deterministic_value():
    value = NoisyFloat.lift(2.6)
    rounded = value.round_nearest()

    assert isinstance(rounded, NoisyInt)
    assert int(rounded) == 3
    draws = rounded.sample(n=5, rng=123).draws
    assert np.all(draws == 3)


def test_noisyfloat_round_nearest_tie_uses_floor_plus_half_rule():
    theta_node = LatentNode()
    theta = theta_node.expr
    root = DerivedNode(
        expr=theta,
        constraints=(theta - 2.5,),
        deps=(theta_node,),
    )

    value = from_node(obs=2.5, root=root, expr=theta)
    rounded = value.round_nearest()

    assert int(rounded) == 3
    draws = rounded.sample(n=4, rng=123).draws
    assert np.all(draws == 3)


def test_noisyfloat_divide_by_zero_returns_inf_observation():
    symbol = sp.Symbol("symbol")
    node = DerivedNode(expr=symbol)
    x = NoisyFloat(1.0, node)
    y = NoisyFloat(0.0, node)

    z = x / y

    assert isinstance(z, NoisyFloat)
    assert np.isinf(float(z))


def test_noisyfloat_zero_divide_zero_returns_nan_observation():
    symbol = sp.Symbol("symbol")
    node = DerivedNode(expr=symbol)
    x = NoisyFloat(0.0, node)
    y = NoisyFloat(0.0, node)

    z = x / y

    assert isinstance(z, NoisyFloat)
    assert np.isnan(float(z))


def test_noisyfloat_guarded_returns_value_when_guard_true():
    x = NoisyFloat.lift(3.5)

    guarded = x.guarded(True)

    assert isinstance(guarded, NoisyFloat)
    assert float(guarded) == pytest.approx(3.5)
    draws = guarded.sample(n=5, rng=123).draws
    assert np.all(draws == 3.5)


def test_noisyfloat_guarded_returns_nan_when_guard_false():
    x = NoisyFloat.lift(3.5)

    guarded = x.guarded(False)

    assert isinstance(guarded, NoisyFloat)
    assert np.isnan(float(guarded))
    draws = guarded.sample(n=5, rng=123).draws
    assert np.all(np.isnan(draws))


def test_noisyfloat_guarded_preserves_uncertainty_when_guard_is_noisy_bool():
    theta_node = LatentNode()
    theta = theta_node.expr
    eps_node = noise.gaussian(0, 1)
    eps = eps_node.expr
    constraints = [theta + eps - 4.0]

    value = rooted_float(obs=4.0, expr=theta, eqns=constraints, deps=(theta_node, eps_node))
    guarded = value.guarded(value > 0)

    assert isinstance(guarded, NoisyFloat)
    draws = guarded.sample(n=128, rng=123).draws
    finite = draws[np.isfinite(draws)]
    assert draws.shape == (128,)
    assert finite.size > 0


def test_noisyfloat_power_supports_plain_exponent():
    x = NoisyFloat.lift(3.0)

    z = x ** 2

    assert isinstance(z, NoisyFloat)
    assert float(z) == pytest.approx(9.0)


def test_noisyfloat_reverse_power_supports_plain_base():
    x = NoisyFloat.lift(3.0)

    z = 2.0 ** x

    assert isinstance(z, NoisyFloat)
    assert float(z) == pytest.approx(8.0)


def test_noisyfloat_invalid_real_power_returns_nan_observation():
    x = NoisyFloat.lift(-1.0)
    y = NoisyFloat.lift(0.5)

    z = x ** y

    assert isinstance(z, NoisyFloat)
    assert np.isnan(float(z))


def test_noisyint_binomial_explicit_obs_replaces_resample_pattern():
    resampled = NoisyInt.binomial(2, 0.5, obs=5)

    assert isinstance(resampled, NoisyInt)
    assert int(resampled) == 5
    assert any(isinstance(node, NoiseNode) for node in resampled._root.closure())

    draws = resampled.sample(n=128, rng=123).draws
    assert draws.dtype == int
    assert np.all(draws >= 0)
    assert np.all(draws <= 2)


def test_noisyint_binomial_preserves_upstream_dependency_from_n():
    theta_node = LatentNode()
    theta = theta_node.expr
    root = DerivedNode(
        expr=theta,
        constraints=(theta - 3.0,),
        deps=(theta_node,),
    )
    count = NoisyInt(3, root)
    resampled = NoisyInt.binomial(count, 0.5, obs=3)

    assert theta in resampled._root.latent_symbols()


def test_noisyint_binomial_invalid_binomial_parameter_yields_nan_draws():
    theta_node = LatentNode()
    theta = theta_node.expr
    root = DerivedNode(
        expr=theta,
        constraints=(theta - 1.5,),
        deps=(theta_node,),
    )
    prob = from_node(obs=1.5, root=root, expr=theta)
    resampled = NoisyInt.binomial(10, prob, obs=3)

    noisy_float = from_node(obs=float(int(resampled)), root=resampled._root, expr=resampled._root.expr)
    draws = noisy_float.sample(n=16, rng=123).draws

    assert draws.shape == (16,)
    assert np.all(np.isnan(draws))


def test_noisyint_binomial_invalid_parameter_raises_instead_of_corrupting_draws():
    theta_node = LatentNode()
    theta = theta_node.expr
    root = DerivedNode(
        expr=theta,
        constraints=(theta - 1.5,),
        deps=(theta_node,),
    )
    prob = from_node(obs=1.5, root=root, expr=theta)
    resampled = NoisyInt.binomial(10, prob, obs=3)

    with pytest.raises(ValueError, match="NaN"):
        resampled.sample(n=16, rng=123)


def test_sampler_resolves_multilayer_law_dependencies():
    z1 = noise.gaussian(0, 1)
    z1_symbol = z1.expr
    z2 = NormalNode.create(deps=(z1,), loc=z1_symbol, scale=1)
    root = DerivedNode(
        expr=z2.expr,
        deps=(z2,),
    )

    value = from_node(obs=0.0, root=root, expr=z2.expr)
    draws = value.sample(n=4000, rng=123).draws

    assert draws.shape == (4000,)
    assert np.all(np.isfinite(draws))
    assert np.var(draws) == pytest.approx(2.0, rel=0.25)


def test_sampler_uses_root_output_definition():
    theta_node = LatentNode()
    theta = theta_node.expr
    root = DerivedNode(
        expr=theta,
        constraints=(theta - 4.0,),
        deps=(theta_node,),
    )

    value = from_node(obs=4.0, root=root, expr=theta + 9.0)
    assert value._root.expr == theta + 9.0
    assert value._root.constraints == frozenset({theta - 4.0})
    draws = value.sample(n=8, rng=123).draws

    assert draws.shape == (8,)
    assert np.all(draws == 13.0)


def test_node_derived_uses_explicit_dependencies():
    theta = LatentNode()
    eps = noise.gaussian(0, 1)

    value = DerivedNode(expr=theta.expr + eps.expr, deps=(theta, eps))

    assert {dep.expr for dep in value.deps} == {theta.expr, eps.expr}


def test_node_noise_uses_explicit_dependencies():
    theta = LatentNode()

    z = NormalNode.create(deps=(theta,), loc=theta.expr, scale=1)

    assert {dep.expr for dep in z.deps} == {theta.expr}


def test_node_uses_fresh_symbols_for_each_node():
    first = LatentNode()
    second = LatentNode()

    assert first.expr != second.expr


def test_node_derived_wires_wrapper_and_symbol_nodes_explicitly():
    theta_node = LatentNode()
    theta = theta_node.expr
    base_root = DerivedNode(expr=theta, constraints=(theta - 2.0,), deps=(theta_node,))
    wrapped = from_node(obs=3.0, root=base_root, expr=theta + 1.0)

    out = DerivedNode(expr=theta + 2.0, deps=(wrapped._root, theta_node))

    dep_symbols = {dep.expr for dep in out.deps}
    assert wrapped._root.expr in dep_symbols
    assert theta in dep_symbols


def test_draw_obs():
    '''
    The observed value of a NoisyFloat is its true value plus noise.
    '''
    noise_source = noise.gaussian(loc=0, scale=1)
    noisy_float = NoisyFloat.draw(5.0, noise_source, rng=_rng_factory())
    expected_noise = _rng_factory().normal(loc=0, scale=1)
    assert float(noisy_float) == expected_noise + 5.0


def test_draw_uses_root_node():
    noise_source = noise.gaussian(loc=0, scale=1)
    noisy_float = NoisyFloat.draw(5.0, noise_source, rng=_rng_factory())

    assert isinstance(noisy_float._root, DerivedNode)
    assert noisy_float._root.latent_symbols()


# ── NoisyFloat.normal ──────────────────────────────────────────────────────────

def test_noisyfloat_normal_returns_noisyfloat_with_float_obs():
    x = NoisyFloat.normal(0, 1, rng=42)

    assert isinstance(x, NoisyFloat)
    assert isinstance(x._obs, float)


def test_noisyfloat_normal_root_is_noise_node():
    x = NoisyFloat.normal(0, 1, rng=42)

    assert isinstance(x._root, NoiseNode)


def test_noisyfloat_normal_explicit_obs_is_used():
    x = NoisyFloat.normal(0, 1, obs=7.5)

    assert x._obs == 7.5


def test_noisyfloat_normal_same_rng_gives_same_obs():
    assert NoisyFloat.normal(0, 1, rng=42)._obs == NoisyFloat.normal(0, 1, rng=42)._obs


def test_noisyfloat_normal_plain_params_yield_independent_noise_node():
    x = NoisyFloat.normal(3, 2, rng=42)

    noise_nodes = [n for n in x._root.closure() if isinstance(n, NoiseNode)]
    assert all(len(n.deps) == 0 for n in noise_nodes)


def test_noisyfloat_normal_samples_from_correct_distribution():
    x = NoisyFloat.normal(3.0, 2.0, rng=42)
    draws = x.sample(n=4000, rng=99).draws

    assert draws.mean() == pytest.approx(3.0, abs=0.15)
    assert draws.std() == pytest.approx(2.0, abs=0.15)


def test_noisyfloat_normal_noisy_loc_obs_uses_observed_value_of_loc():
    mu = NoisyFloat.normal(5.0, 0.0001, rng=1)
    x = NoisyFloat.normal(mu, 0.0001, rng=2)

    assert x._obs == pytest.approx(5.0, abs=0.05)


def test_noisyfloat_normal_noisy_loc_propagates_uncertainty_in_sampling():
    mu = NoisyFloat.normal(5.0, 1.0, rng=1)
    x = NoisyFloat.normal(mu, 0.1, rng=2)
    draws = x.sample(n=4000, rng=99).draws

    assert draws.mean() == pytest.approx(5.0, abs=0.2)
    assert draws.std() == pytest.approx(1.0, abs=0.15)


def test_noisyfloat_normal_shared_noisy_loc_induces_correlation():
    mu = NoisyFloat.normal(0, 1, rng=1)
    x = NoisyFloat.normal(mu, 0.01, rng=2)
    y = NoisyFloat.normal(mu, 0.01, rng=3)

    batch_x, batch_y = sample_noisy_values(x, y, n=2000, rng=42)
    corr = np.corrcoef(batch_x.draws, batch_y.draws)[0, 1]
    assert corr > 0.99


# ── NoisyInt.binomial ──────────────────────────────────────────────────────────

def test_noisyint_binomial_returns_noisyint_with_int_obs():
    k = NoisyInt.binomial(10, 0.3, rng=42)

    assert isinstance(k, NoisyInt)
    assert isinstance(k._obs, int)


def test_noisyint_binomial_root_is_noise_node():
    k = NoisyInt.binomial(10, 0.3, rng=42)

    assert isinstance(k._root, NoiseNode)


def test_noisyint_binomial_explicit_obs_is_used():
    k = NoisyInt.binomial(10, 0.3, obs=4)

    assert k._obs == 4


def test_noisyint_binomial_same_rng_gives_same_obs():
    assert NoisyInt.binomial(10, 0.3, rng=42)._obs == NoisyInt.binomial(10, 0.3, rng=42)._obs


def test_noisyint_binomial_plain_params_yield_independent_noise_node():
    k = NoisyInt.binomial(10, 0.3, rng=42)

    noise_nodes = [n for n in k._root.closure() if isinstance(n, NoiseNode)]
    assert all(len(n.deps) == 0 for n in noise_nodes)


def test_noisyint_binomial_samples_from_correct_distribution():
    k = NoisyInt.binomial(20, 0.4, rng=42)
    draws = k.sample(n=4000, rng=99).draws

    assert draws.mean() == pytest.approx(8.0, abs=0.3)


def test_noisyint_binomial_noisy_p_obs_uses_observed_value_of_p():
    p = NoisyFloat.normal(0.5, 0.0001, rng=1)
    k = NoisyInt.binomial(10, p, rng=2)

    assert k._obs == pytest.approx(5.0, abs=1.0)


def test_noisyint_binomial_noisy_p_propagates_uncertainty_in_sampling():
    p = NoisyFloat.normal(0.5, 0.05, rng=1)
    k = NoisyInt.binomial(10, p, rng=2)
    draws = k.sample(n=4000, rng=99).draws

    assert draws.mean() == pytest.approx(5.0, abs=0.3)
    assert draws.std() > np.sqrt(10 * 0.5 * 0.5)  # wider than pure binomial variance


# ── NoisyInt.discrete_gaussian ────────────────────────────────────────────────

def test_noisyint_discrete_gaussian_returns_noisyint_with_int_obs():
    k = NoisyInt.discrete_gaussian(10, rng=42)

    assert isinstance(k, NoisyInt)
    assert isinstance(k._obs, int)


def test_noisyint_discrete_gaussian_root_is_noise_node():
    k = NoisyInt.discrete_gaussian(10, rng=42)

    assert isinstance(k._root, DiscreteGaussianNode)


def test_noisyint_discrete_gaussian_explicit_obs_is_used():
    k = NoisyInt.discrete_gaussian(10, obs=-3)

    assert k._obs == -3


def test_noisyint_discrete_gaussian_same_rng_gives_same_obs():
    assert NoisyInt.discrete_gaussian(10, rng=42)._obs == NoisyInt.discrete_gaussian(10, rng=42)._obs


def test_noisyint_discrete_gaussian_plain_scale_yields_independent_noise_node():
    k = NoisyInt.discrete_gaussian(10, rng=42)

    noise_nodes = [n for n in k._root.closure() if isinstance(n, NoiseNode)]
    assert all(len(n.deps) == 0 for n in noise_nodes)


def test_noisyint_discrete_gaussian_samples_are_integers():
    k = NoisyInt.discrete_gaussian(10, rng=42)
    draws = k.sample(n=200, rng=99).draws

    assert draws.dtype == int


def test_noisyint_discrete_gaussian_samples_have_correct_mean_and_variance():
    k = NoisyInt.discrete_gaussian(20, rng=42)
    draws = k.sample(n=8000, rng=99).draws

    assert draws.mean() == pytest.approx(0.0, abs=1.0)
    assert draws.var() == pytest.approx(400.0, rel=0.1)


def test_noisyint_discrete_gaussian_noisy_scale_propagates_uncertainty():
    sigma = NoisyFloat.normal(10.0, 1.0, rng=1)
    k = NoisyInt.discrete_gaussian(sigma, rng=2)
    draws = k.sample(n=4000, rng=99).draws

    assert draws.mean() == pytest.approx(0.0, abs=0.5)
    assert draws.std() == pytest.approx(10.0, abs=1.5)


# ── NoisyInt.discrete_laplace ─────────────────────────────────────────────────

def test_noisyint_discrete_laplace_returns_noisyint_with_int_obs():
    k = NoisyInt.discrete_laplace(10, rng=42)

    assert isinstance(k, NoisyInt)
    assert isinstance(k._obs, int)


def test_noisyint_discrete_laplace_root_is_noise_node():
    k = NoisyInt.discrete_laplace(10, rng=42)

    assert isinstance(k._root, DiscreteLaplaceNode)


def test_noisyint_discrete_laplace_same_rng_gives_same_obs():
    assert NoisyInt.discrete_laplace(10, rng=42)._obs == NoisyInt.discrete_laplace(10, rng=42)._obs


def test_noisyint_discrete_laplace_samples_have_correct_mean_and_variance():
    k = NoisyInt.discrete_laplace(20, rng=42)
    draws = k.sample(n=8000, rng=99).draws

    # two-sided geometric with scale 20: variance = 2p/(1-p)^2, p = e^(-1/20)
    p = np.exp(-1 / 20)
    assert draws.mean() == pytest.approx(0.0, abs=1.5)
    assert draws.var() == pytest.approx(2 * p / (1 - p) ** 2, rel=0.1)


def test_noisyint_discrete_laplace_tails_are_heavier_than_gaussian():
    # P(|X| > 4 sd) is ~3.5e-3 for Laplace noise but ~6e-5 for Gaussian;
    # this is what distinguishes the log-density from the Gaussian node's
    k = NoisyInt.discrete_laplace(30, rng=42)
    draws = k.sample(n=20000, rng=99).draws

    assert (np.abs(draws) > 4 * draws.std()).mean() > 1e-3


def test_noisyint_discrete_laplace_respects_truncation_bounds():
    k = NoisyInt.discrete_laplace(10, low=0, high=5, rng=1)
    draws = k.sample(n=500, rng=99).draws

    assert draws.min() >= 0
    assert draws.max() <= 5


def test_mixing_noisy_types_in_one_expression_keeps_every_operand_noisy():
    """A chain of three or more additions used to silently drop operands.

    `bin_op` lifted the right operand with the left's own class, so once the
    running total had promoted to NoisyFloat, each further NoisyInt was
    treated as a plain scalar and sympified down to its observation. The sum
    still looked right; it just had no uncertainty from anything past the
    second term.
    """
    parts = [NoisyInt.discrete_gaussian(5, obs=obs) for obs in (10, 20, 30, 40)]
    total = parts[0]
    for part in parts[1:]:
        total = total + part

    assert len(total.expr.free_symbols) == len(parts)

    batches = sample_noisy_values(total, *parts, n=300, rng=4)
    assert np.array_equal(batches[0].draws, sum(b.draws for b in batches[1:]))
    assert batches[0].draws.std() > 5


def test_a_noisy_float_times_a_noisy_int_keeps_both_posteriors():
    scale = NoisyFloat.normal(2.0, 0.5, rng=1)
    count = NoisyInt.discrete_gaussian(5, obs=100)
    product = scale * count

    assert len(product.expr.free_symbols) == 2
    a, b, c = sample_noisy_values(product, scale, count, n=300, rng=6)
    assert np.allclose(a.draws, b.draws * c.draws)


# --- consolidate() -----------------------------------------------------------


def _draw(true_val, scale=1):
    return NoisyFloat.draw(true_val, noise.gaussian(0, scale))


def _noise_node_count(value):
    return sum(1 for node in value._root.closure() if isinstance(node, NoiseNode))


def test_consolidate_combines_two_normals_into_one_node():
    x = _draw(1.0)
    y = _draw(2.0)
    (z_cons,) = consolidate(x + y)
    assert _noise_node_count(z_cons) == 1


def test_consolidate_combines_three_normals_into_one_node():
    (z_cons,) = consolidate(_draw(1.0) + _draw(2.0) + _draw(3.0))
    assert _noise_node_count(z_cons) == 1


def test_consolidate_preserves_observation():
    x = _draw(1.0)
    y = _draw(2.0)
    z = x + y
    (z_cons,) = consolidate(z)
    assert float(z_cons) == float(z)


def test_consolidate_returns_same_type():
    x = _draw(1.0)
    y = _draw(2.0)
    (z_cons,) = consolidate(x + y)
    assert type(z_cons) is NoisyFloat


def test_consolidate_shared_noise_symbol_not_combined():
    # eps_x appears in both x and z=x+y, so it must remain in both outputs.
    x = _draw(1.0)
    y = _draw(2.0)
    z = x + y
    x_cons, z_cons = consolidate(x, z)
    assert _noise_node_count(x_cons) == 1
    assert _noise_node_count(z_cons) == 2


def test_consolidate_non_normal_noise_not_combined():
    # The binomial source has no combination rule, so nothing is merged.
    x = _draw(1.0)
    count = NoisyInt.binomial(10, 0.3, obs=3)
    z = x + count
    (z_cons,) = consolidate(z)
    assert _noise_node_count(z_cons) == _noise_node_count(z)


def test_consolidate_single_value_nothing_to_combine():
    x = _draw(1.0)
    (x_cons,) = consolidate(x)
    assert _noise_node_count(x_cons) == 1


def test_consolidate_samples_have_correct_mean():
    x = _draw(1.0)
    y = _draw(2.0)
    z = x + y
    (z_cons,) = consolidate(z)
    samples = z_cons.sample(n=5000, rng=42)
    assert abs(samples.mean() - float(z)) < 0.1


def test_consolidate_samples_match_original_std():
    x = _draw(1.0)
    y = _draw(2.0)
    z = x + y
    (z_cons,) = consolidate(z)
    orig_std = np.std(z.sample(n=5000, rng=42).draws)
    cons_std = np.std(z_cons.sample(n=5000, rng=42).draws)
    assert abs(orig_std - cons_std) < 0.05


def test_consolidate_combined_sigma_scales_with_coefficients():
    # 2*x + 3*y: combined sigma = sqrt(4*s^2 + 9*s^2) = sqrt(13)*s
    x = _draw(0.0, scale=1.0)
    y = _draw(0.0, scale=1.0)
    z = 2.0 * x + 3.0 * y
    (z_cons,) = consolidate(z)
    assert _noise_node_count(z_cons) == 1
    expected_std = float(np.sqrt(13))
    actual_std = np.std(z_cons.sample(n=10000, rng=42).draws)
    assert abs(actual_std - expected_std) < 0.1
