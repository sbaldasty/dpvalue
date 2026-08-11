import itertools
import numpy as np
import sympy as sp

from sympy.stats import quantile

from .core import _filter_theta_equations
from .core import _solve_theta_substitutions
from .graph import NoiseNode


def _weighted_quantile(values, weights, q):
    if len(values) == 0:
        raise ValueError("Cannot compute quantiles of an empty sample")

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")

    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("weights must sum to a positive finite value")

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / total

    q = float(np.clip(q, 0.0, 1.0))
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def _quantile_nodes_for_rv(source, quadrature_points, eps=1e-8):
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)

    # Map Gauss-Legendre nodes from [-1, 1] to [0, 1].
    u = 0.5 * (nodes + 1.0)
    u = np.clip(u, eps, 1.0 - eps)
    w = 0.5 * weights

    # A node may supply its own closed-form quantile, for distributions
    # sympy cannot invert.
    if hasattr(source, "quantile"):
        return (np.asarray(source.quantile(u), dtype=float),
                np.asarray(w, dtype=float))

    qfun = quantile(source.sympy_rv())
    rv_values = []
    for ui in u:
        try:
            rv_values.append(float(sp.N(qfun(float(ui)))))
        except Exception as exc:
            raise ValueError(
                f"Could not evaluate quantile for noise source {source}. "
                "This method requires continuous distributions with numeric quantiles."
            ) from exc

    return np.asarray(rv_values, dtype=float), np.asarray(w, dtype=float)


def _compute_posterior_quadrature_points(noisy_value, quadrature_points=17, max_grid_points=300000):
    if quadrature_points < 2:
        raise ValueError("quadrature_points must be at least 2")

    closure_nodes = {node.expr: node for node in noisy_value._root.closure()}
    independent_noise = {
        symbol: node
        for symbol, node in closure_nodes.items()
        if isinstance(node, NoiseNode) and not node.deps
    }
    independent_noise_symbols = set(independent_noise.keys())

    thetas = noisy_value._root.latent_symbols()
    constraints = noisy_value._root.all_constraints()
    theta_constraints = _filter_theta_equations(constraints, thetas, independent_noise_symbols)

    if thetas:
        sol = _solve_theta_substitutions(thetas, theta_constraints)
        rhs_noise_symbols = set()
        for rhs in sol.values():
            rhs_expr = sp.sympify(rhs)
            rhs_noise_symbols |= rhs_expr.free_symbols & independent_noise_symbols
    else:
        sol = {}
        rhs_noise_symbols = set()

    value_expr = noisy_value.expr
    predictive_noise_symbols = value_expr.free_symbols & independent_noise_symbols
    integration_syms = sorted(rhs_noise_symbols | predictive_noise_symbols, key=str)

    if not integration_syms:
        value = float(value_expr)
        return np.asarray([value], dtype=float), np.asarray([1.0], dtype=float)

    total_points = quadrature_points ** len(integration_syms)
    if total_points > max_grid_points:
        raise ValueError(
            "Deterministic quadrature grid is too large for this expression. "
            f"Requested {total_points} points; lower quadrature_points or simplify expression."
        )

    sym_to_nodes = {}
    sym_to_weights = {}
    for sym in integration_syms:
        nodes, weights = _quantile_nodes_for_rv(independent_noise[sym], quadrature_points)
        sym_to_nodes[sym] = nodes
        sym_to_weights[sym] = weights

    z_values = np.empty(total_points, dtype=float)
    point_weights = np.empty(total_points, dtype=float)

    point_index = 0
    for combo in itertools.product(range(quadrature_points), repeat=len(integration_syms)):
        draws = {}
        weight = 1.0
        for sym, idx in zip(integration_syms, combo):
            draws[sym] = sym_to_nodes[sym][idx]
            weight *= sym_to_weights[sym][idx]

        theta_values = {theta: float(rhs.subs(draws)) for theta, rhs in sol.items()}
        value = float(value_expr.subs(theta_values).subs(draws))

        z_values[point_index] = value
        point_weights[point_index] = weight
        point_index += 1

    weight_sum = float(np.sum(point_weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        raise ValueError("Quadrature weights are invalid")

    point_weights /= weight_sum
    return z_values, point_weights


def _estimate_weighted_density(values, weights, grid_size=500, bandwidth=None, tail_quantile=0.002):
    if grid_size < 50:
        raise ValueError("grid_size must be at least 50")

    lo = _weighted_quantile(values, weights, tail_quantile)
    hi = _weighted_quantile(values, weights, 1.0 - tail_quantile)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Density bounds are not finite")

    if hi <= lo:
        hi = lo + 1.0

    spread = hi - lo
    lo -= 0.05 * spread
    hi += 0.05 * spread
    grid = np.linspace(lo, hi, grid_size)

    mu = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mu) ** 2))
    std = float(np.sqrt(max(var, 1e-16)))
    n_eff = 1.0 / float(np.sum(weights**2))

    if bandwidth is None:
        bandwidth = 1.06 * std * (n_eff ** (-1.0 / 5.0))
    bandwidth = float(max(bandwidth, 1e-6))

    delta = (grid[:, None] - values[None, :]) / bandwidth
    kernels = np.exp(-0.5 * delta * delta) / np.sqrt(2.0 * np.pi)
    density = np.sum(weights[None, :] * kernels, axis=1) / bandwidth
    density = np.asarray(density, dtype=float)

    # Normalize numerically to protect against truncation and finite grid effects.
    area = float(np.trapezoid(density, grid))
    if area > 0.0 and np.isfinite(area):
        density /= area

    return grid, density


def plot_posterior(
    *noisy_values,
    grid_size=500,
    quadrature_points=17,
    max_grid_points=300000,
    bandwidth=None,
    tail_quantile=0.002,
    ax=None,
):
    """
    Plot posterior densities of each NoisyFloat's composed expression.

    This is deterministic (no Monte Carlo): it approximates integration over all
    noise symbols using tensor-product Gauss-Legendre quadrature in quantile
    space, then uses weighted KDE to render a smooth density curve.
    """
    if not noisy_values:
        raise ValueError("plot_posteriors requires at least one NoisyFloat")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plot_posteriors") from exc

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
        created_fig = True

    curves = []
    for idx, noisy_value in enumerate(noisy_values, start=1):
        value_expr = noisy_value.expr
        z_values, z_weights = _compute_posterior_quadrature_points(
            noisy_value,
            quadrature_points=quadrature_points,
            max_grid_points=max_grid_points,
        )
        x_grid, density = _estimate_weighted_density(
            z_values,
            z_weights,
            grid_size=grid_size,
            bandwidth=bandwidth,
            tail_quantile=tail_quantile,
        )

        label = f"expr_{idx}: {sp.sstr(value_expr)}"
        ax.plot(x_grid, density, linewidth=2.0, label=label)
        curves.append(
            {
                "noisy_value": noisy_value,
                "x": x_grid,
                "density": density,
                "quadrature_values": z_values,
                "quadrature_weights": z_weights,
            }
        )

    ax.set_xlabel("Expression value")
    ax.set_ylabel("Posterior density")
    ax.set_title("Posterior Densities of Composed Expressions")
    if len(curves) > 1:
        ax.legend(loc="best")

    if created_fig:
        plt.tight_layout()

    return {"ax": ax, "curves": curves}
