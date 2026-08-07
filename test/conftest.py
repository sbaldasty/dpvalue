import sympy as sp

from noisyvalue.core import NoisyFloat
from noisyvalue.graph import DerivedNode, NormalNode, BinomialNode, DiscreteGaussianNode
from sympy import Basic, sympify


def from_node(obs, root, expr=None):
    expr = root.expr if expr is None else sp.sympify(expr)
    if expr != root.expr:
        root = DerivedNode.operational(expr, deps=[root])
    return NoisyFloat(obs, root)


def rooted_float(obs, expr, eqns=(), deps=()):
    expr = sp.sympify(expr)
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)
    root = DerivedNode(expr, constraints=eqns, deps=deps)
    return from_node(obs=obs, root=root, expr=expr)


def _to_expr(value):
    if isinstance(value, Basic):
        return value
    if isinstance(value, (int, float)):
        return value

    from noisyvalue.core import NoisyFloat, NoisyInt

    if isinstance(value, (NoisyFloat, NoisyInt)):
        return value.expr

    expr = sympify(value)
    if isinstance(expr, Basic):
        return expr
    raise TypeError(f"Unsupported value type: {type(value)}")


def gaussian(loc, scale):
    return NormalNode.create(loc=_to_expr(loc), scale=_to_expr(scale))


def binomial(n, p):
    return BinomialNode.create(trials=_to_expr(n), prob=_to_expr(p))


def discrete_gaussian(scale):
    return DiscreteGaussianNode.create(loc=_to_expr(0), scale=_to_expr(scale), low=-sp.oo, high=sp.oo)
