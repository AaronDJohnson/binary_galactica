# Copyright 2019 Tyson B. Littenberg & Neil J. Cornish
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Drop-in shim for callers written against the existing generator.

The existing pipeline asks for a template on a fixed, contiguous block of
rfft bins (``Segment.template`` returns ``(2, n_bins)`` holding ``A`` and
``E`` in the continuous-Fourier convention).
``binary_galactica.glass`` returns a ``BW``-wide block centered on the
source. :func:`template_on_window` places that block on the requested
analysis window.

Bins the source does not reach are zero, and source power falling outside
the requested window is silently dropped --- which is the same contract the
existing generator has, and the same one that makes it the caller's job to
choose a window wide enough for the band.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import _validation
from .convert import GLASS_TO_CONTINUOUS
from .waveform import (
    DEFAULT_RESPONSE_SAMPLES,
    _options,
    _resolve_samples,
    _waveform,
)


def template_on_window(
    params,
    *,
    T,
    bin_lo: int,
    n_bins: int,
    BW: int,
    t0=0.0,
    fmt: str = "sangria",
    positions=None,
    backend="onfly",
    response_samples: int | None = DEFAULT_RESPONSE_SAMPLES,
):
    """``(2, n_bins)`` complex ``[A, E]`` on bins ``bin_lo .. bin_lo+n_bins-1``.

    In the continuous-Fourier convention, i.e. already through
    :func:`binary_galactica.glass.convert.to_continuous_ft`. The on-fly
    backend forms A and E from the response rows before any interpolation and
    transforms only those two channels, so they equal
    :func:`binary_galactica.glass.ucb_waveform`'s A and E to rounding, not
    bitwise.

    Parameters
    ----------
    params
        GLASS UCB parameter vector, as for :func:`binary_galactica.glass.ucb_waveform`.
    T
        Observation time [s].
    bin_lo, n_bins
        The analysis window, in absolute rfft bin indices. Static integers;
        ``n_bins`` must be non-negative. Zero returns an empty ``(2, 0)`` array.
    BW
        Generator bandwidth. Choose it from
        :func:`binary_galactica.glass.bandwidth.ucb_bw`, or fix it for reproducibility.
    t0
        Epoch of the first slow-time sample [s].
    fmt
        TDI convention.
    positions
        Optional cached orbit for this ``(T, t0, BW)``. On the on-fly path a
        response count below ``BW`` interpolates it, and its grid limits the
        geometry accuracy; a count at or above ``BW`` uses it unchanged.
    backend
        Algorithm: ``"onfly"`` (default) or the legacy ``"fourier"`` path.
    response_samples
        ``None`` selects a duration-dependent power-of-two grid of 32–2048
        samples, targeting 128 samples per year. An explicit static integer
        of at least four fixes the count independently of ``BW``. Counts at
        or above ``BW`` evaluate the response directly at the ``BW`` FFT
        times instead of interpolating. Unused by the legacy backend. Pass an
        explicit count when differentiating ``T`` across automatic grid
        thresholds: automatic selection is piecewise and changes the
        interpolation grid. A concrete ``T`` fixes the automatic count on
        the host before compiling; a traced ``T`` compiles every automatic
        count below ``BW``, and one direct evaluation, and emits
        :class:`~binary_galactica.glass.waveform.ResponseSamplesWarning`
        when ``BW`` exceeds 32, as for
        :func:`binary_galactica.glass.ucb_waveform`.

    """
    _validation.require_x64()
    bin_lo = _validation.integer(bin_lo, "bin_lo")
    n_bins = _validation.integer(n_bins, "n_bins", minimum=0)
    response_samples = _options(backend, response_samples)
    if backend == "onfly":
        response_samples = _resolve_samples(response_samples, T, np.float64, BW)
    return _template_on_window(
        params,
        T=T,
        bin_lo=bin_lo,
        n_bins=n_bins,
        BW=BW,
        t0=t0,
        fmt=fmt,
        positions=positions,
        backend=backend,
        response_samples=response_samples,
    )


@partial(
    jax.jit,
    static_argnames=("BW", "bin_lo", "n_bins", "fmt", "backend", "response_samples"),
)
def _template_on_window(
    params, *, T, bin_lo, n_bins, BW, t0, fmt, positions, backend, response_samples
):
    """Compile waveform generation and placement after validating static indices.

    The on-fly path forms only A and E, from the sparse response, and
    transforms two channels instead of three.
    """
    w = _waveform(
        params,
        T=T,
        BW=BW,
        t0=t0,
        fmt=fmt,
        positions=positions,
        backend=backend,
        response_samples=response_samples,
        channels="ae",
    )
    c = GLASS_TO_CONTINUOUS(T)

    idx = jnp.asarray(w["freq_bin_indices"] - bin_lo, dtype=jnp.int32)

    # ``mode="drop"`` alone is NOT enough. JAX applies Python's negative-index
    # convention first, so an index of -90 -- a bin ninety below the window --
    # is a perfectly valid reference to ``n_bins - 90`` and gets scattered
    # onto the window's upper half rather than dropped. Only |idx| >= n_bins
    # is treated as out of bounds. Masking the values and sending every
    # invalid index to a single out-of-range positive slot removes the
    # ambiguity, whatever the mode does.
    valid = (idx >= 0) & (idx < n_bins)
    safe = jnp.where(valid, idx, n_bins)
    zero = jnp.zeros((), dtype=jnp.complex128)

    out = jnp.zeros((2, n_bins), dtype=jnp.complex128)
    out = out.at[0, safe].add(jnp.where(valid, c * w["A"], zero), mode="drop")
    out = out.at[1, safe].add(jnp.where(valid, c * w["E"], zero), mode="drop")
    return out


__all__ = ["template_on_window"]
