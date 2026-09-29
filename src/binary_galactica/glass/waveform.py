"""Sparse-time TDI on the fly, followed by a heterodyned Fourier transform.

The mainline path combines delayed source samples into XYZ, removes the
reference-spacecraft phase, and interpolates the smooth complex response.
The explicit ``fourier`` backend keeps
the original GLASS single-link FFT implementation for reproducibility.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from . import _validation
from ._precision import cast, real_dtype
from .constants import FSTAR, PI2, YEAR
from .fourier import (  # noqa: F401 -- private compatibility
    _carrier_bin,
    _sinc,
    orbit_cache,
    params_from_native,
)
from .fourier import ucb_waveform as _fourier_waveform
from .onfly import reference_phase_shift
from .onfly import tdi_response as _raw_response
from .orbit import slow_time_grid
from .sparse import interpolate_uniform
from .tdi import FORMATS, xyz_to_ae
from .types import WaveformResult

BACKENDS = ("onfly", "fourier")
DEFAULT_RESPONSE_SAMPLES = None
_AUTO_RESPONSE_SAMPLES = (32, 64, 128, 256, 512, 1024, 2048)


def _native_parameters(params, T):
    """Validate native source coordinates without extracting tracers."""
    params = _validation.finite_array(params, "params")
    if params.ndim != 1 or params.shape[0] not in (7, 8, 9):
        raise ValueError(
            "params must be a one-dimensional array with length 7, 8, or 9; "
            f"got shape {params.shape}"
        )
    if not isinstance(params, jax.core.Tracer):
        concrete = np.asarray(params)
        if concrete[0] <= 0:
            raise ValueError("f0 must be positive")
        if np.any(np.abs(concrete[[1, 4]]) > 1):
            raise ValueError("sky and inclination cosines must lie in [-1, 1]")
        if not isinstance(T, jax.core.Tracer):
            with np.errstate(over="ignore", under="ignore"):
                frequency = concrete[0] / np.asarray(T)
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError("f0 must be finite and positive")
    return params


def _source_valid(params, T):
    """Domain mask shared by traced waveform and arbitrary-time evaluations."""
    frequency = params[0] / T
    return (
        jnp.all(jnp.isfinite(params))
        & jnp.isfinite(T)
        & (T > 0)
        & jnp.isfinite(frequency)
        & (frequency > 0)
        & (jnp.abs(params[1]) <= 1)
        & (jnp.abs(params[4]) <= 1)
    )


def _options(backend, response_samples):
    """Validate the static algorithm and response-grid controls."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}; got {backend!r}")
    if response_samples is None:
        return None
    response_samples = _validation.integer(response_samples, "response_samples")
    if response_samples < 4:
        raise ValueError("response_samples must be an integer at least 4")
    return response_samples


def tdi_response(
    params: ArrayLike,
    times: ArrayLike,
    *,
    T: ArrayLike,
    positions: ArrayLike | None = None,
) -> Array:
    """Return demodulated analytic Sangria XYZ at arbitrary times, shape ``(3,n)``.

    ``params`` uses the same native 7/8/9 coordinates as :func:`ucb_waveform`.
    ``times`` is a nonempty one-dimensional vector of seconds relative to the
    source reference epoch. ``T`` unscales the native frequency derivatives;
    it does not restrict the evaluation times. Optional positions have shape
    ``(3,n,3)`` in metres and describe the geometry at exactly these times.

    Multiply by ``exp(1j * Phi(times))`` to restore the analytic source phase,
    where ``Phi(t) = 2*pi*(f0*t + fdot*t**2/2 + fddot*t**3/6) - phi0``.
    The real part is the Sangria time signal in the package's normalization.
    This is half GLASS's raw complex spline response. A positive-frequency
    Fourier coefficient contributes another factor of one half.

    The constellation is frozen at each reception time, using actual arm
    lengths and delayed source samples. This follows GLASS's spline response;
    it does not implement point-ahead or sequential moving-arm TDI-2 delays.
    Eager invalid domains raise ``ValueError``; invalid traced inputs yield NaN.
    """
    _validation.require_x64()
    T = _validation.scalar(T, "T", positive=True)
    params = _native_parameters(params, T)
    times = _validation.finite_array(times, "times")
    if times.ndim != 1 or times.shape[0] == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if positions is not None:
        positions = _validation.finite_array(positions, "positions")
        if positions.shape != (3, times.shape[0], 3):
            raise ValueError("positions must have shape (3, len(times), 3)")
    return _tdi_response(params, times, T, positions)


@jax.jit
def _tdi_response(params, times, T, positions):
    """Evaluate validated arbitrary-time inputs inside one compiled call."""
    valid = _source_valid(params, T) & jnp.all(jnp.isfinite(times))
    if positions is not None:
        valid = valid & jnp.all(jnp.isfinite(positions))
    return jnp.where(valid, 0.5 * _raw_response(params, times, T, positions), jnp.nan)


def ucb_waveform(
    params: ArrayLike,
    *,
    T: ArrayLike,
    BW: int,
    t0: ArrayLike = 0.0,
    fmt: str = "sangria",
    positions: ArrayLike | None = None,
    q: ArrayLike | None = None,
    backend: str = "onfly",
    response_samples: int | None = DEFAULT_RESPONSE_SAMPLES,
) -> WaveformResult:
    """Generate a banded native-parameter TDI spectrum, using TDI on the fly.

    ``params`` is ``[f0*T, sin(beta), longitude, log(A_G), cos(iota), psi,
    phi0_G, fdot*T**2, fddot*T**3]`` with the last two entries optional.
    ``BW`` is the positive even FFT length and number of returned bins.
    ``response_samples`` controls the sparse response grid independently.
    The default ``None`` chooses a power of two from 32 through 2048, at least
    128 samples per observation year until the cap: quarter-year observations
    use 32, one year uses 128, and four years use 512. Supply a static integer
    at least four to choose the count explicitly. Increase both grid controls
    separately to check convergence for the source and observation duration;
    the automatic policy is not an error bound, especially beyond 16 years.

    The sparse envelope removes the source phase at spacecraft 0's retarded
    time. Local polynomials through up to eight complex response samples
    interpolate it without time derivatives or singular phase coordinates at
    response zeros. The reference phase is restored on the FFT grid.

    ``T`` (positive duration), ``t0`` (observation start) and ``q`` (optional
    integral carrier override) may be traced. Source parameters are always
    referenced to time zero. The default carrier preserves GLASS's separate
    float64 operations ``trunc((params[0]/T)*T)``. Returned ``qmin`` and
    ``freq_bin_indices`` describe the same ``BW`` bins as the Fourier backend.

    All five channels ``X/Y/Z/A/E`` use raw GLASS Fourier units; multiply by
    ``sqrt(T)`` for continuous-Fourier units. ``fmt`` selects ``sangria``,
    ``frequency`` or ``phase``. Phase DC is defined as zero and its A/E
    coefficients are conjugated to retain the GLASS convention.

    Optional ``positions`` retains the legacy ``(3,BW,3)`` cache layout on
    ``t0 + T*arange(BW)/BW``. On the on-fly path this geometry is interpolated
    locally (cubic for BW >= 4, linear for BW = 2); cache sampling then limits
    orbit accuracy independently of ``response_samples``. Omitting positions
    evaluates the analytic orbit directly at every needed response time.

    ``backend="fourier"`` explicitly selects the old single-link FFT path,
    for which ``response_samples`` has no effect. The default uses actual arm
    lengths and delayed source evolution, so it is not bitwise equivalent.

    Eager invalid inputs raise ``ValueError``; traced invalid values return
    NaN in every output. Shapes and static controls are always checked.
    AD holds the discrete carrier and automatic grid selection fixed. Automatic
    count changes introduce small interpolation jumps at duration thresholds;
    use an explicit count when differentiating across those thresholds.
    """
    _validation.require_x64()
    response_samples = _options(backend, response_samples)
    if backend == "fourier":
        return _fourier_waveform(
            params, T=T, BW=BW, t0=t0, fmt=fmt, positions=positions, q=q
        )
    T, t0, positions = _validation.observation(T, BW, t0, positions)
    params = _native_parameters(params, T)
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}; got {fmt!r}")
    if q is not None:
        q = _validation.scalar(q, "q")
        if not isinstance(q, jax.core.Tracer) and np.asarray(q) != np.trunc(q):
            raise ValueError("q must be an integral Fourier-bin index")
    return _ucb_waveform(
        params,
        T=T,
        BW=int(BW),
        t0=t0,
        fmt=fmt,
        positions=positions,
        q=q,
        response_samples=response_samples,
    )


def _cached_positions(positions, times, T, t0):
    """Evaluate a legacy uniform position cache by local polynomial interpolation.

    The result has the floating dtype of ``times``; the other inputs are cast to it.
    """
    real = real_dtype(times)
    positions = jnp.asarray(positions, dtype=real)
    T, t0 = cast(T, real), cast(t0, real)
    count = positions.shape[1]
    width = min(4, count)
    x = (times - t0) * count / T
    start = jnp.clip(jnp.floor(x).astype(jnp.int32) - 1, 0, count - width)
    local = x - start
    result = jnp.zeros((3, times.shape[0], 3), dtype=real)
    for i in range(width):
        weight = jnp.ones_like(local)
        for j in range(width):
            if i != j:
                weight = weight * (local - j) / (i - j)
        result = result + positions[:, start + i, :] * weight[None, :, None]
    return result


def _duration_boundaries(real):
    """Durations above which the automatic grid takes its next count.

    Lower precisions round the float64 thresholds down. A duration then
    exceeds one exactly when its float64 value exceeds the float64 threshold,
    so both precisions choose the same grid at identical inputs.
    """
    if real == jnp.float64:
        return YEAR * jnp.asarray(_AUTO_RESPONSE_SAMPLES[:-1], dtype=real) / 128
    exact = YEAR * np.asarray(_AUTO_RESPONSE_SAMPLES[:-1], dtype=np.float64) / 128
    rounded = exact.astype(real)
    return np.where(
        rounded > exact, np.nextafter(rounded, rounded.dtype.type(0)), rounded
    )


def _response_envelope(params, T, BW, t0, positions, response_samples):
    """Interpolate on a static grid, selecting a duration bucket when requested.

    Knots, coordinates and the duration buckets use the floating dtype of
    ``params``, to which ``T``, ``t0`` and positions are cast.
    """
    real = real_dtype(params)
    T, t0 = cast(T, real), cast(t0, real)
    if positions is not None:
        positions = jnp.asarray(positions, dtype=real)

    def sample(count, params, T, t0, positions):
        knots = t0 + T * (1 - 1 / BW) * jnp.linspace(0.0, 1.0, count, dtype=real)
        geometry = (
            None if positions is None else _cached_positions(positions, knots, T, t0)
        )
        response = _raw_response(params, knots, T, geometry, reference_phase=True).T
        # Static normalized coordinates avoid clock-subtraction rounding and
        # stencil changes when duration, epoch or source coordinates are traced.
        coordinates = jnp.arange(BW, dtype=real) / (BW - 1)
        return interpolate_uniform(response, coordinates).T

    if response_samples is not None:
        return sample(response_samples, params, T, t0, positions)
    # Every branch returns the same FFT-grid shape. Only the selected sparse
    # response is evaluated for a scalar T; JAX can still trace T and its AD.
    boundaries = _duration_boundaries(real)
    index = jnp.sum(boundaries < T, dtype=jnp.int32)
    branches = tuple(partial(sample, count) for count in _AUTO_RESPONSE_SAMPLES)
    return jax.lax.switch(index, branches, params, T, t0, positions)


@partial(jax.jit, static_argnames=("BW", "fmt", "response_samples"))
def _ucb_waveform(
    params,
    *,
    T,
    BW,
    t0=0.0,
    fmt="sangria",
    positions=None,
    q=None,
    response_samples=DEFAULT_RESPONSE_SAMPLES,
) -> WaveformResult:
    """Interpolate combined XYZ, restore the carrier residual, then transform."""
    q = (
        _carrier_bin(jax.lax.stop_gradient(params[0]), jax.lax.stop_gradient(T))
        if q is None
        else jnp.asarray(q, dtype=jnp.float64)
    )
    valid = _source_valid(params, T) & jnp.isfinite(t0) & jnp.isfinite(q)
    valid = valid & (q == jnp.trunc(q))
    if positions is not None:
        valid = valid & jnp.all(jnp.isfinite(positions))
    times = slow_time_grid(T, t0, BW)
    envelope = _response_envelope(params, T, BW, t0, positions, response_samples)
    # Restore phi(t-k.x0/c) minus the carrier without forming or subtracting
    # large source phases. Public tdi_response retains phi(t) demodulation.
    u = times / T
    cycles = (params[0] - q) * u
    if params.shape[0] > 7:
        cycles = cycles + 0.5 * params[7] * u**2
    if params.shape[0] > 8:
        cycles = cycles + params[8] * u**3 / 6.0
    phase = (
        PI2 * cycles - params[6] + reference_phase_shift(params, times, T, positions)
    )
    envelope = envelope * jnp.exp(1j * phase)
    xyz = jnp.fft.fftshift(jnp.fft.fft(envelope, axis=-1), axes=-1)
    xyz = xyz * (jnp.sqrt(T) / (4 * BW))
    qmin = q - BW // 2
    bins = qmin + jnp.arange(BW, dtype=jnp.float64)
    if fmt == "frequency":
        xyz = -1j * xyz
    elif fmt == "phase":
        x = bins / (T * FSTAR)
        xyz = jnp.where(x == 0, 0j, xyz / (2j * jnp.where(x == 0, 1.0, x)))
    X, Y, Z = xyz
    A, E = xyz_to_ae(X, Y, Z)
    if fmt == "phase":
        A, E = jnp.conj(A), jnp.conj(E)
    result = {
        "qmin": qmin,
        "freq_bin_indices": bins,
        "X": X,
        "Y": Y,
        "Z": Z,
        "A": A,
        "E": E,
    }
    return {name: jnp.where(valid, value, jnp.nan) for name, value in result.items()}


__all__ = [
    "BACKENDS",
    "DEFAULT_RESPONSE_SAMPLES",
    "orbit_cache",
    "params_from_native",
    "tdi_response",
    "ucb_waveform",
]
