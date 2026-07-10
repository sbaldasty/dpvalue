import numpy as np

from numpy.random import Generator


def as_nonempty_tuple(xs, cls):
    assert len(xs) > 0
    return as_tuple(xs, cls)

def as_tuple(xs, cls):
    assert all(isinstance(x, cls) for x in xs)
    return tuple(xs)

def generator(rng):
    return rng if isinstance(rng, Generator) else np.random.default_rng(rng)

def require_subclass(accept, cls):
    if not issubclass(cls, accept):
        raise TypeError(f"Expected {accept.__name__}, got {cls.__name__}")
    return cls
