"""Sparse-time TDI on the fly, followed by a heterodyned Fourier transform.

The mainline path combines delayed source samples into XYZ, removes the
reference-spacecraft phase, and interpolates the smooth complex response onto
the FFT times, or evaluates it there directly when the response grid has at
least as many samples. The explicit ``fourier`` backend keeps
the original GLASS single-link FFT implementation for reproducibility.
"""

import warnings
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
#: Channel sets of the internal on-fly pipeline. ``"xyz"`` transforms X, Y
#: and Z and forms A and E from their spectra; ``"ae"`` forms A and E from the
#: sparse response and transforms only those two.
_CHANNELS = ("xyz", "ae")


class ResponseSamplesWarning(UserWarning):
    """The automatic response grid could not be chosen before compiling.

    Emitted once by a public waveform call that receives a traced duration
    ``T`` with ``response_samples=None`` and a band ``BW`` above the smallest
    automatic count. The compiled program then holds every automatic count
    below ``BW``, and one direct evaluation for the counts of at least
    ``BW``, behind ``jax.lax.switch``, and under differentiation JAX keeps
    the intermediate values of all of them, which multiplies memory and run
    time. Pass an explicit static ``response_samples``, or close over a
    concrete ``T`` so that the count is chosen on the host.
    """


_TRACED_DURATION = (
    "response_samples=None with a traced duration T: the automatic response "
    "grid depends on the value of T, so each automatic count below BW "
    f"(from {_AUTO_RESPONSE_SAMPLES[0]} to {_AUTO_RESPONSE_SAMPLES[-1]} samples), "
    "and one direct evaluation at the BW FFT times for the counts of at least "
    "BW, is compiled behind jax.lax.switch, and under differentiation JAX "
    "keeps the intermediate values of every one, multiplying memory and run "
    "time. Pass an explicit static response_samples, or close over a concrete "
    "T (a Python or NumPy scalar) so that the count is chosen before compiling."
)


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


def _resolve_samples(response_samples, T, real, BW):
    """Choose the automatic response count on the host, before compiling.

    Call directly from a public entry, after its validation, with the
    caller's own ``T``: validation repacks a concrete JAX array with ``jnp``,
    which inside an enclosing trace makes it a tracer. A concrete ``T`` gives
    exactly the count the compiled switch would select for the same value in
    the working dtype ``real``. A traced ``T`` keeps ``None``, and the
    switch, and warns once at the caller's line, unless every automatic count
    is at least ``BW``: they then share one direct evaluation and nothing is
    selected. Compiled callers below a public entry pass its result on, so
    they neither resolve nor warn again. ``BW`` may still be unvalidated.
    """
    if response_samples is not None:
        return response_samples
    if not isinstance(T, jax.Array) and _validation._contains_jax(T):
        T = jnp.asarray(T)  # A sequence holding JAX values, packed as validation does.
    if isinstance(T, jax.core.Tracer):
        integral = isinstance(BW, (int, np.integer)) and not isinstance(BW, bool)
        if not integral or len(_automatic_counts(BW)) > 1:
            warnings.warn(_TRACED_DURATION, ResponseSamplesWarning, stacklevel=3)
        return None
    duration = np.asarray(T, dtype=real)
    if duration.shape != ():
        return None  # The entry's own validation rejects it.
    return _automatic_samples(duration, real)


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
    The real part is the Sangria time signal for the physical amplitude
    ``exp(params[3])``. This is GLASS's raw complex spline response. A
    positive-frequency Fourier coefficient contributes a factor of one half.

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
    return jnp.where(valid, _raw_response(params, times, T, positions), jnp.nan)


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
    ``A_G`` is the physical strain amplitude (LDC ``Amplitude``), the value
    GLASS's ``amplitude()`` returns: on both backends ``h+ = A_G*(1+cos^2(i))``
    and ``hx = 2*A_G*cos(i)``. ``BW`` is the positive even FFT length and number
    of returned bins.
    ``response_samples`` controls the sparse response grid independently.
    The default ``None`` chooses a power of two from 32 through 2048, at least
    128 samples per observation year until the cap: quarter-year observations
    use 32, one year uses 128, and four years use 512. Supply a static integer
    at least four to choose the count explicitly. A count below ``BW`` is
    interpolated onto the ``BW`` FFT times; a count at or above ``BW``
    evaluates the response directly at those times instead, so every such
    count gives the same result. Increase both grid controls separately to
    check convergence for the source and observation duration; the automatic
    policy is not an error bound, especially beyond 16 years.
    A concrete ``T`` (a Python or NumPy scalar, or a concrete JAX array, also
    when captured by an enclosing ``jax.jit``) fixes the automatic count on
    the host before compiling. A traced ``T`` cannot: every automatic count
    below ``BW``, and one direct evaluation for the rest, is then compiled
    behind ``jax.lax.switch``, reverse-mode derivatives keep the
    intermediates of all of them, and the call emits
    :class:`ResponseSamplesWarning`. With ``BW`` at most 32 every automatic
    count is direct, and a traced ``T`` compiles one grid without warning.
    Close over a fixed duration, and pass an explicit count when ``T`` must
    be a ``jit``, ``grad`` or ``vmap`` argument.

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
    ``t0 + T*arange(BW)/BW``. On the on-fly path a count below ``BW``
    interpolates this geometry locally (cubic for BW >= 4, linear for
    BW = 2) to its response times, and cache sampling then limits orbit
    accuracy independently of ``response_samples``; a count at or above
    ``BW`` uses the cache unchanged at the FFT times. Omitting positions
    evaluates the analytic orbit directly at every needed response time.

    ``backend="fourier"`` explicitly selects the old single-link FFT path,
    for which ``response_samples`` has no effect. The default uses actual arm
    lengths and delayed source evolution, so it is not bitwise equivalent.

    Eager invalid inputs raise ``ValueError``; traced invalid values return
    NaN in every output. Shapes and static controls are always checked.
    AD holds the discrete carrier and automatic grid selection fixed. Automatic
    count changes below ``BW`` introduce small interpolation jumps at duration
    thresholds; use an explicit count when differentiating across those
    thresholds.
    """
    _validation.require_x64()
    response_samples = _options(backend, response_samples)
    if backend == "fourier":
        return _fourier_waveform(
            params, T=T, BW=BW, t0=t0, fmt=fmt, positions=positions, q=q
        )
    duration, t0, positions = _validation.observation(T, BW, t0, positions)
    params, q = _onfly_inputs(params, duration, fmt, q)
    response_samples = _resolve_samples(response_samples, T, np.float64, BW)
    return _ucb_waveform(
        params,
        T=duration,
        BW=int(BW),
        t0=t0,
        fmt=fmt,
        positions=positions,
        q=q,
        response_samples=response_samples,
    )


def _onfly_inputs(params, T, fmt, q):
    """Validate the on-fly source, format and carrier override."""
    params = _native_parameters(params, T)
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}; got {fmt!r}")
    if q is not None:
        q = _validation.scalar(q, "q")
        if not isinstance(q, jax.core.Tracer) and np.asarray(q) != np.trunc(q):
            raise ValueError("q must be an integral Fourier-bin index")
    return params, q


def _waveform(
    params, *, T, BW, t0, fmt, positions, backend, response_samples, channels="xyz"
):
    """Validate and generate like :func:`ucb_waveform`, without resolving.

    For compiled callers below a public entry, which has already resolved
    ``response_samples`` with :func:`_resolve_samples`, or warned and kept
    ``None``, so that one public call warns at most once. ``channels="ae"``
    returns only A, E and the bins on the on-fly path, as in
    :func:`_ucb_waveform`; the legacy backend always returns all five channels.
    """
    response_samples = _options(backend, response_samples)
    if backend == "fourier":
        return _fourier_waveform(
            params, T=T, BW=BW, t0=t0, fmt=fmt, positions=positions
        )
    T, t0, positions = _validation.observation(T, BW, t0, positions)
    params, q = _onfly_inputs(params, T, fmt, None)
    return _ucb_waveform(
        params,
        T=T,
        BW=int(BW),
        t0=t0,
        fmt=fmt,
        positions=positions,
        q=q,
        response_samples=response_samples,
        channels=channels,
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

    A host NumPy array of dtype ``real``, shared by the compiled selection
    and :func:`_automatic_samples`. Lower precisions round the float64
    thresholds down. A duration then exceeds one exactly when its float64
    value exceeds the float64 threshold, so both precisions choose the same
    grid at identical inputs.
    """
    exact = YEAR * np.asarray(_AUTO_RESPONSE_SAMPLES[:-1], dtype=np.float64) / 128
    if real == jnp.float64:
        return exact
    rounded = exact.astype(real)
    return np.where(
        rounded > exact, np.nextafter(rounded, rounded.dtype.type(0)), rounded
    )


def _automatic_index(T, real):
    """Select the automatic grid inside a compiled function, as an int32 index.

    ``T`` must already have the working dtype ``real``.
    """
    return jnp.sum(_duration_boundaries(real) < T, dtype=jnp.int32)


def _automatic_samples(T, real):
    """Return the count :func:`_automatic_index` selects, for a host duration.

    The same comparison in NumPy, after casting ``T`` to the working dtype.
    """
    duration = np.asarray(T, dtype=real)
    return _AUTO_RESPONSE_SAMPLES[int(np.sum(_duration_boundaries(real) < duration))]


def _automatic_counts(BW):
    """Return the response counts the automatic grid compiles for ``BW``.

    Each count below ``BW`` interpolates and needs its own branch. Every count
    of at least ``BW`` evaluates the response directly at the FFT times, so
    those share one branch, listed as ``BW``.
    """
    below = tuple(count for count in _AUTO_RESPONSE_SAMPLES if count < BW)
    return below if len(below) == len(_AUTO_RESPONSE_SAMPLES) else (*below, BW)


def _response_envelope(
    params, T, BW, t0, positions, response_samples, *, channels="xyz"
):
    """Sample the response on the FFT grid, selecting a duration bucket if asked.

    A count below ``BW`` interpolates that many knots onto the ``BW`` FFT
    times. A count of at least ``BW`` evaluates the response directly at
    ``slow_time_grid(T, t0, BW)``, the times of the phase restore, where a
    legacy position cache is already sampled. Knots, times, coordinates and
    the duration buckets use the floating dtype of ``params``, to which
    ``T``, ``t0`` and positions are cast. The envelope has shape ``(3, BW)``
    holding X/Y/Z for ``channels="xyz"``, or ``(2, BW)`` holding A/E,
    combined from the response rows, for ``"ae"``.
    """
    if channels not in _CHANNELS:
        raise ValueError(f"channels must be one of {_CHANNELS}; got {channels!r}")
    real = real_dtype(params)
    T, t0 = cast(T, real), cast(t0, real)
    if positions is not None:
        positions = jnp.asarray(positions, dtype=real)
    times = slow_time_grid(T, t0, BW)

    def combine(response):
        if channels == "ae":
            # A and E are fixed combinations of X, Y and Z, and interpolation,
            # the phase restore, the FFT and the format factors are linear and
            # the same for every channel. Combining the response rows leaves
            # two channels to interpolate and transform instead of three.
            return jnp.stack(xyz_to_ae(*response))
        return response

    def sample(count, params, T, t0, times, positions):
        if count >= BW:
            # Interpolating at least BW knots onto the BW FFT times would cost
            # more than evaluating the response there, and add error.
            return combine(
                _raw_response(params, times, T, positions, reference_phase=True)
            )
        knots = t0 + T * (1 - 1 / BW) * jnp.linspace(0.0, 1.0, count, dtype=real)
        geometry = (
            None if positions is None else _cached_positions(positions, knots, T, t0)
        )
        response = combine(
            _raw_response(params, knots, T, geometry, reference_phase=True)
        )
        # Static normalized coordinates avoid clock-subtraction rounding and
        # stencil changes when duration, epoch or source coordinates are traced.
        coordinates = jnp.arange(BW, dtype=real) / (BW - 1)
        return interpolate_uniform(response.T, coordinates).T

    if response_samples is not None:
        return sample(response_samples, params, T, t0, times, positions)
    # Public entries resolve a concrete T on the host, so only a traced T
    # reaches the switch. Every branch returns the same FFT-grid shape and only
    # the selected one runs for a scalar T, but every branch is compiled and
    # reverse-mode AD keeps the residuals of all of them. Counts of at least
    # BW share the last, direct branch, so the index is clamped onto it.
    branches = tuple(partial(sample, count) for count in _automatic_counts(BW))
    if len(branches) == 1:
        return branches[0](params, T, t0, times, positions)
    index = jnp.minimum(_automatic_index(T, real), len(branches) - 1)
    return jax.lax.switch(index, branches, params, T, t0, times, positions)


@partial(jax.jit, static_argnames=("BW", "fmt", "response_samples", "channels"))
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
    channels="xyz",
) -> WaveformResult:
    """Sample the response on the FFT grid, restore the carrier residual, transform.

    :func:`_response_envelope` interpolates a coarser response grid onto the
    FFT times, or evaluates the response there when the grid is at least
    ``BW``. ``channels="xyz"`` transforms X, Y and Z and returns all five
    channels. ``"ae"`` forms A and E from the response rows, transforms only
    those, and returns them with ``qmin`` and ``freq_bin_indices``; they
    equal the five-channel A and E to rounding.
    """
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
    envelope = _response_envelope(
        params, T, BW, t0, positions, response_samples, channels=channels
    )
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
    # A complex exponential, not _trig.cis: XLA in JAX 0.6 and 0.9 recomputes
    # cis's cosine and sine in backward consumers here, which made float32
    # value-and-gradient calls up to three times slower on CPU, and JAX 0.11
    # ran no faster with it.
    envelope = envelope * jnp.exp(1j * phase)
    spectra = jnp.fft.fftshift(jnp.fft.fft(envelope, axis=-1), axes=-1)
    spectra = spectra * (jnp.sqrt(T) / (2 * BW))
    qmin = q - BW // 2
    bins = qmin + jnp.arange(BW, dtype=jnp.float64)
    if fmt == "frequency":
        spectra = -1j * spectra
    elif fmt == "phase":
        x = bins / (T * FSTAR)
        spectra = jnp.where(x == 0, 0j, spectra / (2j * jnp.where(x == 0, 1.0, x)))
    result = {"qmin": qmin, "freq_bin_indices": bins}
    if channels == "ae":
        A, E = spectra
    else:
        X, Y, Z = spectra
        A, E = xyz_to_ae(X, Y, Z)
        result.update(X=X, Y=Y, Z=Z)
    if fmt == "phase":
        # GLASS's LISA_tdi conjugates A and E, not X, Y and Z, after the transform.
        A, E = jnp.conj(A), jnp.conj(E)
    result.update(A=A, E=E)
    return {name: jnp.where(valid, value, jnp.nan) for name, value in result.items()}


__all__ = [
    "BACKENDS",
    "DEFAULT_RESPONSE_SAMPLES",
    "ResponseSamplesWarning",
    "orbit_cache",
    "params_from_native",
    "tdi_response",
    "ucb_waveform",
]
