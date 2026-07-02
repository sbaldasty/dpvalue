import noisyvalue as nv
import numpy as np
import pytest
from scipy.stats import chi2_contingency

import conftest as noise

from conftest import rooted_float
from noisyvalue import NoisyContingencyTable
from noisyvalue import NoisyFloat
from noisyvalue.graph import LatentNode
from noisyvalue.graph import Node
from noisyvalue.graph import NoiseNode


def test_noisy_min_and_noisy_max_for_plain_floats_match_python_min_max():
    lo = nv.noisy_min(3.0, -1.5, 8.0)
    hi = nv.noisy_max(3.0, -1.5, 8.0)

    assert isinstance(lo, NoisyFloat)
    assert isinstance(hi, NoisyFloat)
    assert float(lo) == -1.5
    assert float(hi) == 8.0


def test_noisy_min_raises_for_empty_input():
    with pytest.raises(ValueError, match="Requires at least one value"):
        nv.noisy_min()


def test_noisy_max_combines_noisy_value_metadata():
    theta_node = LatentNode()
    theta = theta_node.expr
    constraints = [theta - 1.0]
    a = rooted_float(obs=1.0, expr=theta, eqns=constraints, deps=(theta_node,))
    b = rooted_float(obs=2.0, expr=2.0 * theta, eqns=constraints, deps=(theta_node,))

    out = nv.noisy_max(a, b)

    assert isinstance(out, NoisyFloat)
    assert float(out) == 2.0
    assert out._root.latent_symbols() == a._root.latent_symbols()
    assert (theta - 1.0) in out._root.all_constraints()


def test_as_contingency_table_returns_noisy_contingency_table():
    table = NoisyContingencyTable([[1.0, 2.0], [3.0, 4.0]])

    assert isinstance(table, NoisyContingencyTable)
    assert table.tbl.shape == (2, 2)


def test_odds_ratio_enforces_2x2_shape():
    with pytest.raises(AssertionError):
        NoisyContingencyTable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).odds_ratio()


def test_noisy_contingency_table_odds_ratio_enforces_2x2_shape():
    with pytest.raises(AssertionError):
        NoisyContingencyTable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).odds_ratio()


def test_chi_squared_enforces_2d_shape():
    with pytest.raises(AssertionError):
        NoisyContingencyTable([1.0, 2.0, 3.0]).chi_squared()


def test_chi_squared_matches_scipy_for_plain_floats():
    table = [[65.0, 109.0], [243.0, 1348.0]]
    result = NoisyContingencyTable(table).chi_squared()
    expected = chi2_contingency(table, correction=False).statistic

    assert isinstance(result, NoisyFloat)
    assert float(result) == pytest.approx(expected)


def test_noisy_contingency_table_repeatability_on_same_input():
    table_data = [[65.0, 109.0], [243.0, 1348.0]]

    table = NoisyContingencyTable(table_data)
    chi_a = table.chi_squared()
    or_a = table.odds_ratio()
    chi_b = table.chi_squared()
    or_b = table.odds_ratio()

    assert float(chi_a) == pytest.approx(float(chi_b))
    assert float(or_a) == pytest.approx(float(or_b))


def test_chi_squared_returns_nan_when_a_row_has_no_mass():
    result = NoisyContingencyTable([[1.0, 2.0], [0.0, 0.0]]).chi_squared()

    assert isinstance(result, NoisyFloat)
    assert np.isnan(float(result))


def test_chi_squared_builds_single_noisy_float_with_propagated_uncertainty():
    theta_node = LatentNode()
    theta = theta_node.expr
    noisy_a = rooted_float(obs=5.0, expr=theta, eqns=[theta - 5.0], deps=(theta_node,))

    result = NoisyContingencyTable([[noisy_a, 7.0], [11.0, 13.0]]).chi_squared()

    assert isinstance(result, NoisyFloat)
    assert result._root.latent_symbols() == noisy_a._root.latent_symbols()

    draws = result.sample(n=128, rng=123).draws
    assert draws.shape == (128,)
    assert np.all(np.isfinite(draws))


def test_odds_ratio_matches_closed_form_for_plain_floats():
    ratio = NoisyContingencyTable([[65.0, 109.0], [243.0, 1348.0]]).odds_ratio()
    expected = (65.0 * 1348.0) / (109.0 * 243.0)

    assert isinstance(ratio, NoisyFloat)
    assert float(ratio) == pytest.approx(expected)


def test_odds_ratio_sample_keeps_only_valid_draws():
    ratio = NoisyContingencyTable([[5.0, 7.0], [11.0, 13.0]]).odds_ratio()
    expected = (5.0 * 13.0) / (7.0 * 11.0)

    draws = ratio.sample(n=400, rng=123).draws

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (400,)
    assert np.all(np.isfinite(draws))
    assert np.allclose(draws, expected)


def test_odds_ratio_sample_with_zero_n_returns_empty_array():
    ratio = NoisyContingencyTable([[1.0, 2.0], [3.0, 4.0]]).odds_ratio()

    draws = ratio.sample(n=0, rng=123).draws

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (0,)

def test_odds_ratio_builds_single_noisy_float_with_propagated_uncertainty():
    theta_node = LatentNode()
    theta = theta_node.expr
    eps_node = noise.gaussian(0, 1)
    eps = eps_node.expr

    noisy_a = rooted_float(obs=5.0, expr=theta, eqns=[theta + eps - 5.0], deps=(theta_node, eps_node))

    ratio = NoisyContingencyTable([[noisy_a, 7.0], [11.0, 13.0]]).odds_ratio()

    assert isinstance(ratio, NoisyFloat)
    assert ratio._root.latent_symbols() == noisy_a._root.latent_symbols()
    assert any(isinstance(node, NoiseNode) for node in ratio._root.closure())

    draws = ratio.sample(n=128, rng=123).draws
    assert draws.shape == (128,)
    assert np.all(np.isfinite(draws))


def test_odds_ratio_returns_nan_observation_when_observed_ratio_is_invalid():
    ratio = NoisyContingencyTable([[1.0, 0.0], [2.0, 3.0]]).odds_ratio()

    assert isinstance(ratio, NoisyFloat)
    assert np.isnan(float(ratio))
    assert isinstance(ratio._root, Node)


def test_odds_ratio_sampling_handles_out_of_range_noisy_probabilities():
    # Invalid observed counts produce NaNs through the validity gate.
    ratio = NoisyContingencyTable([[5.0, -1.0], [11.0, 13.0]]).odds_ratio()

    draws = ratio.sample(n=64, rng=123).draws

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (64,)
    assert np.all(np.isnan(draws))


def test_contingency_table_predictive_supports_general_2d_shape():
    table = [[10.0, 20.0, 30.0], [9.0, 3.0, 8.0]]

    predictive = NoisyContingencyTable.stratified(table).tbl

    assert predictive.shape == (2, 3)
    assert np.isfinite(np.asarray([float(value) for value in predictive.ravel()], dtype=float)).all()


def test_stratified_returns_noisy_contingency_table():
    table = [[10.0, 20.0, 30.0], [9.0, 3.0, 8.0]]

    predictive = NoisyContingencyTable.stratified(table)

    assert isinstance(predictive, NoisyContingencyTable)
    assert predictive.tbl.shape == (2, 3)
    assert np.isfinite(np.asarray([float(value) for value in predictive.tbl.ravel()], dtype=float)).all()


def test_chi_squared_accepts_predictive_contingency_table():
    table = [[65.0, 109.0], [243.0, 1348.0]]

    stat = NoisyContingencyTable.stratified(table).chi_squared()
    draws = stat.sample(n=256, rng=123).draws

    assert isinstance(stat, NoisyFloat)
    assert draws.shape == (256,)
    assert np.isfinite(draws).all()
    assert np.std(draws) > 0.0


def test_odds_ratio_accepts_predictive_contingency_table():
    table = [[65.0, 109.0], [243.0, 1348.0]]

    ratio = NoisyContingencyTable.stratified(table).odds_ratio()
    draws = ratio.sample(n=256, rng=123).draws
    finite = draws[np.isfinite(draws)]

    assert isinstance(ratio, NoisyFloat)
    assert draws.shape == (256,)
    assert finite.size > 0
    assert np.std(finite) > 0.0
