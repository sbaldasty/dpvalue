"""Backward-compatible import shim for noisyvalue.pandas.

This module is deprecated. Import from noisyvalue.pandas instead.
"""

import warnings

warnings.warn(
    "noisyvalue.pandas_ext is deprecated; import from noisyvalue.pandas instead",
    DeprecationWarning,
    stacklevel=2,
)

from .pandas import *  # noqa: F401,F403
