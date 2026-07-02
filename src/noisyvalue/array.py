import numpy as np

from .core import noisy_value_sampler


def float_array_sampler(vals):
    """Prepare a reusable sampler for tensor-like value collections.

    The prepared sampler returns arrays with the same base shape as `values`
    plus one sample axis.
    """
    values_array = np.asarray(vals, dtype=object)
    if values_array.size == 0:
        raise ValueError("At least one value is required")

    prepared = noisy_value_sampler(*values_array.reshape(-1).tolist())
    return FloatArraySampler(prepared, values_array.shape)


def sample_float_array(vals, n=1000, rng=None, axis=-1):
    """Jointly sample a tensor-like collection of values.

    Returns a float numpy array with shape `values.shape + (n,)` by default.
    Use `axis` to move the sample dimension.
    """
    sampler = float_array_sampler(vals)
    return sampler.sample(n, rng, axis)


class FloatArraySampler:
    def __init__(self, delegate, shape):
        self._delegate = delegate
        self._shape = shape

    def sample(self, n=1000, rng=None, axis=-1):
        raw = self._delegate.sample(n=n, rng=rng)
        if isinstance(raw, tuple):
            flat = np.stack([batch.draws for batch in raw], axis=0)
        else:
            flat = raw.draws[np.newaxis, :]

        shaped = np.asarray(flat.reshape(self._shape + (n,)), dtype=float)
        if axis == -1:
            return shaped

        ndim = len(self._shape) + 1
        axis = axis if axis >= 0 else axis + ndim
        if axis < 0 or axis >= ndim:
            raise np.AxisError(axis, ndim=ndim)
        return np.moveaxis(shaped, -1, axis)
