import numpy as np

from itertools import count
from numpy.random import Generator

_counter = count()

def fresh_name():
    return f"id_{next(_counter)}"

def reset_name_provider():
    global _counter
    _counter = count()

def as_nonempty_tuple(xs, cls):
    assert len(xs) > 0
    return as_tuple(xs, cls)

def as_tuple(xs, cls):
    assert all(isinstance(x, cls) for x in xs)
    return tuple(xs)

def generator(rng):
    return rng if isinstance(rng, Generator) else np.random.default_rng(rng)
