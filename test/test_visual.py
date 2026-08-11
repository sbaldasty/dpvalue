import matplotlib
import numpy as np
import pytest

from conftest import rooted_float

import conftest as noise
from noisyvalue.graph import LatentNode
from noisyvalue.visual import plot_posterior


matplotlib.use("Agg")

def test_plot_posteriors_for_composed_expression_returns_density_curve():
    theta_0_node = LatentNode()
    theta_0 = theta_0_node.expr
    theta_1_node = LatentNode()
    theta_1 = theta_1_node.expr
    eps_0_node = noise.gaussian(0, 1)
    eps_0 = eps_0_node.expr
    eps_1_node = noise.gaussian(0, 2)
    eps_1 = eps_1_node.expr
    eps_pred_node = noise.gaussian(0, 0.5)
    eps_pred = eps_pred_node.expr

    observed_0 = 3.0
    observed_1 = -1.0

    # Posterior of a composed expression in two latent dimensions.
    noisy = rooted_float(
        obs=0.0,
        expr=theta_0 * theta_1 + eps_pred,
        eqns=[theta_0 + eps_0 - observed_0, theta_1 + eps_1 - observed_1],
        deps=(theta_0_node, theta_1_node, eps_0_node, eps_1_node, eps_pred_node),
    )

    result = plot_posterior(noisy, quadrature_points=7, grid_size=220)

    assert "ax" in result
    assert "curves" in result
    assert len(result["curves"]) == 1

    curve = result["curves"][0]
    x = curve["x"]
    density = curve["density"]

    assert isinstance(x, np.ndarray)
    assert isinstance(density, np.ndarray)
    assert x.shape == density.shape
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)

    area = np.trapezoid(density, x)
    assert area == pytest.approx(1.0, rel=0.08)


def test_plot_posteriors_supports_multiple_values():
    theta_node = LatentNode()
    theta = theta_node.expr
    eps_obs_node = noise.gaussian(0, 1)
    eps_obs = eps_obs_node.expr
    eps_pred_node = noise.gaussian(0, 1)
    eps_pred = eps_pred_node.expr

    noisy_a = rooted_float(
        obs=0.0,
        expr=theta + eps_pred,
        eqns=[theta + eps_obs - 1.0],
        deps=(theta_node, eps_obs_node, eps_pred_node),
    )
    noisy_b = rooted_float(
        obs=0.0,
        expr=2 * theta + eps_pred,
        eqns=[theta + eps_obs - 1.0],
        deps=(theta_node, eps_obs_node, eps_pred_node),
    )

    result = plot_posterior(noisy_a, noisy_b, quadrature_points=7, grid_size=180)

    assert len(result["curves"]) == 2


def test_plot_posterior_supports_discrete_laplace_cells():
    # sympy cannot invert the Laplace CDF (its _quantile returns None), so
    # the node supplies its own closed form; the density must come out
    # centred on the observation
    from noisyvalue.dataset import DiscreteLaplaceFamily

    family = DiscreteLaplaceFamily()
    cell = family.cell(1000, family.variance_from_scale(30.0))
    result = plot_posterior(cell, quadrature_points=9, grid_size=150)

    curve = result["curves"][0]
    peak = curve["x"][np.argmax(curve["density"])]
    assert abs(peak - 1000) < 30


def test_plot_posterior_supports_suppressed_cells():
    # censored nodes have no sympy RV at all; the density comes entirely
    # through their own quantile hook, and should span [0, threshold]
    from noisyvalue.dataset import DiscreteGaussianFamily

    cell = DiscreteGaussianFamily().suppressed(90, 332.2)
    result = plot_posterior(cell, quadrature_points=15, grid_size=150)

    curve = result["curves"][0]
    assert curve["x"].min() > -100
    assert curve["x"].max() < 300
