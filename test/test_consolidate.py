import numpy as np
import conftest as noise

from noisyvalue.consolidate import consolidate
from noisyvalue.core import NoisyFloat, NoisyInt
from noisyvalue.graph import NoiseNode


def _draw(true_val, scale=1):
    return NoisyFloat.draw(true_val, noise.gaussian(0, scale))


def _noise_node_count(value):
    return sum(1 for node in value._root.closure() if isinstance(node, NoiseNode))


# --- structural tests -------------------------------------------------------


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


def test_consolidate_empty_rules_leaves_nodes_unchanged():
    x = _draw(1.0)
    y = _draw(2.0)
    (z_cons,) = consolidate(x + y, rules=[])
    assert _noise_node_count(z_cons) == 2


# --- statistical tests ------------------------------------------------------


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
