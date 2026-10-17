"""LISA detector waveforms from the package's physical source parameters."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from binary_galactica.glass import _validation
from binary_galactica.glass.types import WaveformResult
from binary_galactica.glass.waveform import (
    DEFAULT_RESPONSE_SAMPLES,
    _options,
    _resolve_samples,
    _waveform,
)
from binary_galactica.parameters import GBParams, to_glass_vector


def tdi_waveform(  # noqa: N803 -- names follow the GLASS waveform API
    theta: ArrayLike | GBParams,
    *,
    T: ArrayLike,
    BW: int,
    t0: ArrayLike = 0.0,
    fmt: str = "sangria",
    positions: ArrayLike | None = None,
    backend: str = "onfly",
    response_samples: int | None = DEFAULT_RESPONSE_SAMPLES,
) -> WaveformResult:
    """Generate the raw GLASS TDI spectrum for physical ``GBParams``.

    This is the physical-coordinate counterpart of :func:`ucb_waveform`,
    whose input array instead uses GLASS-native coordinates. The adapter
    converts amplitude, phase, sky angles, inclination and time scales using
    :func:`to_glass_vector`. It includes GLASS's first-order amplitude
    evolution, which the original source-only ``polarizations`` omits.

    Args:
        theta: One physical ``GBParams`` or ``(8,)`` physical array. Use
            ``jax.vmap`` to evaluate multiple sources. All fields must be
            finite, ``amp`` and ``f0`` positive, ``iota`` in ``[0, pi]``,
            and ``beta`` in ``[-pi/2, pi/2]``.
        T: Positive observation duration in seconds. May be traced or
            captured as a constant in surrounding JIT functions.
        BW: Positive even number of slow samples and output bins. Static;
            choose it by convergence for the intended analysis.
        t0: Time of the first observation, in seconds from the source's
            reference epoch. The parameters remain defined at time zero.
        fmt: ``"sangria"``, ``"frequency"``, or ``"phase"`` TDI convention.
        positions: Optional ``(3, BW, 3)`` spacecraft positions on the exact
            ``(T, t0, BW)`` slow grid, as returned by ``orbit_cache``.
            On the on-fly path this cache is interpolated; its sampling limits
            geometry accuracy. Omit it to evaluate analytic geometry directly.
        backend: ``"onfly"`` (default) or the legacy ``"fourier"`` path.
        response_samples: ``None`` selects a duration-dependent power-of-two
            grid of 32–2048 samples, targeting 128 samples per year. An explicit
            static integer of at least four fixes the count independently of
            ``BW``. Unused by the legacy backend. Pass an explicit count when
            differentiating ``T`` across automatic grid thresholds: automatic
            selection is piecewise and changes the interpolation grid. A
            concrete ``T`` fixes the automatic count on the host before
            compiling; a traced ``T`` compiles every automatic grid and emits
            ``ResponseSamplesWarning``, as for :func:`ucb_waveform`.

    Returns:
        A dictionary with complex ``X/Y/Z/A/E`` arrays, ``qmin``, and absolute
        ``freq_bin_indices``. Channels have length ``BW`` and raw GLASS
        normalization. ``to_continuous_ft(result, T)`` multiplies by sqrt(T).

    Notes:
        Derivatives treat the discrete carrier and bandwidth choices as fixed.
        The backend regularizes the sky-pole coordinate singularity. Those
        coordinate-boundary derivatives are not exact physical derivatives.
        Invalid eager inputs raise ``ValueError``. Invalid values supplied
        through JAX tracers yield NaN in every output; shapes are checked
        during tracing as well. See :func:`ucb_waveform` for grid validation.

    """
    _validation.require_x64()
    response_samples = _options(backend, response_samples)
    duration, t0, positions = _validation.observation(T, BW, t0, positions)
    # Host scalar fields can be packed once by validation. Keep to_array's
    # broadcasting and traced packing for array-valued fields.
    if isinstance(theta, GBParams) and not all(np.isscalar(field) for field in theta):
        theta = theta.to_array()
    theta = _validation.finite_array(theta, "theta")
    if theta.shape != (8,):
        raise ValueError(f"theta must have shape (8,); got {theta.shape}")
    if not isinstance(theta, jax.core.Tracer):
        p = np.asarray(theta)
        if p[0] <= 0 or p[1] <= 0:
            raise ValueError("amp and f0 must be positive")
        if not 0 <= p[4] <= np.pi:
            raise ValueError("iota must lie in [0, pi]")
        if not -np.pi / 2 <= p[7] <= np.pi / 2:
            raise ValueError("beta must lie in [-pi/2, pi/2]")
    if backend == "onfly":
        response_samples = _resolve_samples(response_samples, T, np.float64)
    return _tdi_waveform(
        theta,
        T=duration,
        BW=int(BW),
        t0=t0,
        fmt=fmt,
        positions=positions,
        backend=backend,
        response_samples=response_samples,
    )


@partial(jax.jit, static_argnames=("BW", "fmt", "backend", "response_samples"))
def _tdi_waveform(
    theta: Array,
    *,
    T: Array,
    BW: int,
    t0: Array,
    fmt: str,
    positions: Array | None,
    backend: str,
    response_samples: int | None,
) -> WaveformResult:
    """Convert physical coordinates inside JIT and preserve invalid domains."""
    valid = (
        jnp.all(jnp.isfinite(theta))
        & (theta[0] > 0)
        & (theta[1] > 0)
        & (theta[4] >= 0)
        & (theta[4] <= jnp.pi)
        & (jnp.abs(theta[7]) <= jnp.pi / 2)
    )
    native = jnp.where(valid, to_glass_vector(theta, T), jnp.nan)
    return _waveform(
        native,
        T=T,
        BW=BW,
        t0=t0,
        fmt=fmt,
        positions=positions,
        backend=backend,
        response_samples=response_samples,
    )


__all__ = ["tdi_waveform"]
