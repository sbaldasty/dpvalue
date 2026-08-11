import numpy as np

from numpy.random import Generator


def as_nonempty_tuple(xs, cls):
    if len(xs) == 0:
        raise ValueError("Expected at least one item")
    return as_tuple(xs, cls)

def as_tuple(xs, cls):
    if not all(isinstance(x, cls) for x in xs):
        raise TypeError(f"Expected all items to be {cls.__name__}")
    return tuple(xs)

def generator(rng):
    return rng if isinstance(rng, Generator) else np.random.default_rng(rng)

def require_instance(accept, obj):
    if not isinstance(obj, accept):
        raise TypeError(f"Expected {accept.__name__}, got {type(obj).__name__}")
    return obj

def require_subclass(accept, cls):
    if not issubclass(cls, accept):
        raise TypeError(f"Expected {accept.__name__}, got {cls.__name__}")
    return cls
