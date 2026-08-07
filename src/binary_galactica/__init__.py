"""Gravitational waveform model for LISA galactic binaries.

The eight-parameter circular, linearly-chirping model, written in JAX so that
it is differentiable, ``jit``-compilable and ``vmap``-able over source
catalogues.  See :mod:`binary_galactica.waveform` for the equations and sign
conventions.
"""

from __future__ import annotations

import jax as _jax

# Enable float64 before anything else touches JAX.
#
# Over a four-year observation the phase of a millihertz binary accumulates to
# ~1e6 radians.  Held in float32 that carries ~0.06 rad of error -- more than
# enough to destroy a matched filter, yet far too small to notice in a plot.
# Double precision is therefore a correctness requirement of this model, not a
# performance trade-off, and the package refuses to be importable without it.
# This is a process-global setting; see the README.
_jax.config.update("jax_enable_x64", True)

from binary_galactica.constants import DAY, YEAR  # noqa: E402
from binary_galactica.parameters import (  # noqa: E402
    NUM_PARAMS,
    PARAMETER_NAMES,
    GBParams,
)
from binary_galactica.waveform import (  # noqa: E402
    amplitudes,
    frequency,
    phase,
    polarizations,
)

__version__ = "0.1.0"

__all__ = [
    "DAY",
    "NUM_PARAMS",
    "PARAMETER_NAMES",
    "YEAR",
    "GBParams",
    "amplitudes",
    "frequency",
    "phase",
    "polarizations",
]
