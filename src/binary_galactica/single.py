"""Float32 galactic-binary TDI waveforms, for processes that keep x64 disabled.

Every function here computes and returns only float32, complex64 and int32
values, with ``jax_enable_x64`` enabled or not. They use the on-fly response
of :func:`binary_galactica.ucb_waveform`, which remains the float64
reference; there is no ``backend`` argument.

Float32 cannot hold ``f0*T``: near 1e6-1e7 cycles it resolves the frequency
to only 0.06-0.5 cycles, a waveform error of order one. Each source is
therefore an integer reference bin ``q_ref`` plus float32 coordinates whose
frequency entry is the offset from that bin: ``params[0] = f0*T - q_ref`` in
native coordinates, and ``f0 - q_ref/T`` in hertz for physical ones.
:func:`split_native` and :func:`split_physical` convert float64 priors,
proposals and catalogues, choosing ``q_ref = floor(f0*T)``. ``q_ref`` is a
Python integer or an int32 scalar, and may be traced or batched with
``jax.vmap``.

Against the float64 path at identical inputs, ``tools/diagnose_single.py``
measures A/E relative L2 errors below 1.5e-5 and mismatches below 1e-10, for
a target of 1e-4 and 1e-8, at the corners of T = 0.25-10 years, f0 = 0.1-30
mHz and t0 = 0-10 years, and for chirps driven by gravitational radiation
that drift up to 20,000 bins, some with the carrier 30,000 bins from its
default. Results are in ``tools/performance/single-precision.json``. Only the
frequency is split: the other float32 entries are inputs, and for such a chirp
their rounding from float64, which grows as ``(t/T)**2`` at late epochs, can
exceed that error; the same record measures it.

Raw channel values near 1e-21 are representable in float32, but their squares
(about 1e-42) underflow. Form inner products from whitened data, or rescale:
the waveform is linear in amplitude, so subtracting ``log(s)`` from native
``params[3]``, or dividing physical ``amp`` by ``s``, scales every channel by
``1/s``.

Example
-------
>>> import binary_galactica as bg
>>> from binary_galactica import single as bg32
>>> T = 7884000.0
>>> native = bg.params_from_native(
...     f0=1.0002e-3, fdot=1e-18, amp=7.16e-22, phi0=0.5,
...     colat=1.78, lon=0.0, psi=0.7, cos_iota=0.3, T=T)
>>> q_ref, params = bg32.split_native(native, T)
>>> q_ref, params.dtype
(7885, dtype('float32'))
>>> w = bg32.ucb_waveform(params, q_ref=q_ref, T=T, BW=64)
>>> w["A"].dtype, w["qmin"].dtype
(dtype('complex64'), dtype('int32'))

"""

from __future__ import annotations

import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import ArrayLike as HostArrayLike
from numpy.typing import NDArray

from binary_galactica.glass import _validation
from binary_galactica.glass import single as _single
from binary_galactica.glass.tdi import FORMATS
from binary_galactica.glass.types import WaveformResult
from binary_galactica.glass.waveform import _options
from binary_galactica.parameters import GBParams

#: The largest float32 below one, for offsets that round up to a whole bin.
_BELOW_ONE = np.nextafter(np.float32(1), np.float32(0))


def _format(fmt):
    """Reject unknown TDI conventions before compiling."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}; got {fmt!r}")
    return fmt


def ucb_waveform(
    params: ArrayLike,
    *,
    q_ref: ArrayLike,
    T: ArrayLike,
    BW: int,
    t0: ArrayLike = 0.0,
    fmt: str = "sangria",
    positions: ArrayLike | None = None,
    q: ArrayLike | None = None,
    response_samples: int | None = None,
) -> WaveformResult:
    """Generate a banded float32 TDI spectrum from offset native coordinates.

    The float32 counterpart of the on-fly :func:`binary_galactica.ucb_waveform`.
    ``params`` is ``[f0*T - q_ref, sin(beta), longitude, log(A_G), cos(iota),
    psi, phi0_G, fdot*T**2, fddot*T**3]`` with the last two entries optional:
    the float64 layout except for its first entry, which is the offset in
    cycles from the integer reference bin ``q_ref``. ``q_ref - BW`` and
    ``q_ref + BW`` must lie in int32 range, and ``|params[0]|`` below 2**24.
    Use :func:`split_native` to build both from float64 coordinates.

    The carrier defaults to ``q = q_ref + floor(params[0])``, so the residual
    ``f0*T - q`` lies in ``[0, 1)`` exactly. An integral ``q`` override (an
    integer, an integral host float, or an int32 scalar that may be traced)
    moves the band and must lie within 2**24 bins of the source. The float64
    default ``trunc((f0T/T)*T)`` reproduces GLASS rounding and may differ by
    one bin at an exact boundary; pin ``q`` when comparing the two.

    ``T`` is the positive duration and ``t0`` the observation start, both in
    seconds and cast to float32; source parameters are referenced to time
    zero. ``BW`` (positive even FFT length and number of bins), ``fmt`` and
    ``response_samples`` are static and behave as in the float64 function.
    Channels use raw GLASS Fourier units; multiply by ``sqrt(T)`` for
    continuous-Fourier units.

    Optional ``positions`` is the ``(3,BW,3)`` spacecraft cache in metres on
    ``t0 + T*arange(BW)/BW``, as returned by :func:`binary_galactica.orbit_cache`.
    Host float64 or float32 caches are cast to float32, whose step near
    1.5e11 m is about 5e-5 s of light time. Omitting it evaluates the analytic
    orbit, with its phase reduced in paired float32 arithmetic.

    Returns int32 ``qmin`` and ``freq_bin_indices``, exact to 2**31, and
    complex64 ``X/Y/Z/A/E`` of length ``BW``. Their squares underflow float32:
    rescale through ``params[3]`` as described in the module notes.

    Eager invalid inputs raise ``ValueError``; traced invalid values give NaN
    channels and ``INT32_MIN`` bins. AD holds the carrier and the automatic
    grid choice fixed.
    """
    response_samples = _options("onfly", response_samples)
    duration, t0, positions = _validation.observation(
        T, BW, t0, positions, dtype=np.float32
    )
    source = _single.offset_parameters(params)
    q_ref = _single.bin_index(q_ref, "q_ref", margin=BW)
    if q is not None:
        q = _single.bin_index(q, "q", margin=BW // 2, integral_floats=True)
    _single.check_frequency(source[0], q_ref, duration, BW=BW, q=q)
    return _single._ucb_waveform(
        source,
        q_ref,
        T=duration,
        BW=int(BW),
        t0=t0,
        fmt=_format(fmt),
        positions=positions,
        q=q,
        response_samples=response_samples,
        disabled=_single._DISABLED,
    )


def template_on_window(
    params: ArrayLike,
    *,
    q_ref: ArrayLike,
    T: ArrayLike,
    bin_lo: int,
    n_bins: int,
    BW: int,
    t0: ArrayLike = 0.0,
    fmt: str = "sangria",
    positions: ArrayLike | None = None,
    response_samples: int | None = None,
) -> Array:
    """Return ``(2, n_bins)`` complex64 ``[A, E]`` on bins ``bin_lo`` onwards.

    The float32 counterpart of :func:`binary_galactica.template_on_window`,
    in continuous-Fourier units (raw GLASS units times ``sqrt(T)``, with
    ``T`` in seconds). ``params``, ``q_ref``, ``T``, ``BW``, ``t0``, ``fmt``,
    ``positions`` and ``response_samples`` are as for :func:`ucb_waveform`,
    with the default carrier. ``bin_lo`` and ``n_bins`` are static absolute
    rfft bin indices; the window must lie in int32 range. Bins the ``BW``
    band does not reach are zero, and band power outside the window is
    dropped. Choose ``BW`` from :func:`binary_galactica.ucb_bw`.

    The ``sqrt(T)`` factor lifts typical values to 1e-18-1e-15, but their
    squares stay within a few decades of float32's smallest normal number,
    about 1.2e-38, and weaker sources underflow. Whiten, or rescale through
    ``params[3]``, before forming inner products. Eager invalid inputs raise
    ``ValueError``; traced invalid values make every entry NaN.
    """
    bin_lo = _validation.integer(bin_lo, "bin_lo")
    n_bins = _validation.integer(n_bins, "n_bins", minimum=0)
    if not _single.INT32_MIN <= bin_lo <= bin_lo + n_bins <= _single.INT32_MAX:
        raise ValueError("the window bin_lo .. bin_lo + n_bins must lie in int32 range")
    response_samples = _options("onfly", response_samples)
    duration, t0, positions = _validation.observation(
        T, BW, t0, positions, dtype=np.float32
    )
    source = _single.offset_parameters(params)
    q_ref = _single.bin_index(q_ref, "q_ref", margin=BW)
    _single.check_frequency(source[0], q_ref, duration, BW=BW)
    return _single._template_on_window(
        source,
        q_ref,
        T=duration,
        bin_lo=bin_lo,
        n_bins=n_bins,
        BW=int(BW),
        t0=t0,
        fmt=_format(fmt),
        positions=positions,
        response_samples=response_samples,
        disabled=_single._DISABLED,
    )


def tdi_response(
    params: ArrayLike,
    times: ArrayLike,
    *,
    q_ref: ArrayLike,
    T: ArrayLike,
    positions: ArrayLike | None = None,
) -> Array:
    """Return demodulated analytic Sangria XYZ at arbitrary times, ``(3, n)``.

    The float32 counterpart of :func:`binary_galactica.tdi_response`, in the
    same units and phase convention. ``params`` and ``q_ref`` are as for
    :func:`ucb_waveform`; ``q_ref`` and ``q_ref + floor(params[0])`` may be
    any int32 values. ``times`` is a nonempty one-dimensional vector of
    seconds from the source reference epoch, and ``T`` (seconds) unscales the
    frequency derivatives. Both are cast to float32, and each float32 time is
    taken as exact: the orbital phase is reduced from it in paired float32
    arithmetic, so late times keep their Doppler delay. Optional positions
    have shape ``(3,n,3)`` in metres at exactly these times and are cast to
    float32.

    Multiply by ``exp(1j * Phi(times))`` to restore the source phase, where
    ``Phi(t) = 2*pi*(f0*t + fdot*t**2/2 + fddot*t**3/6) - phi0``. With
    ``f0*t`` near millions of cycles, form that phase in float64, or reduce
    ``q_ref*t/T`` and ``params[0]*t/T`` modulo one separately. Values near
    1e-24 square below float32's range; rescale through ``params[3]`` before
    forming products.

    Eager invalid domains raise ``ValueError``; invalid traced inputs yield NaN.
    """
    duration = _validation.scalar(T, "T", positive=True, dtype=np.float32)
    source = _single.offset_parameters(params)
    q_ref = _single.bin_index(q_ref, "q_ref")
    samples = _validation.finite_array(times, "times", dtype=np.float32)
    if samples.ndim != 1 or samples.shape[0] == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    geometry = None
    if positions is not None:
        geometry = _validation.finite_array(positions, "positions", dtype=np.float32)
        if geometry.shape != (3, samples.shape[0], 3):
            raise ValueError("positions must have shape (3, len(times), 3)")
    _single.check_frequency(source[0], q_ref, duration)
    return _single._tdi_response(
        source, q_ref, samples, duration, geometry, disabled=_single._DISABLED
    )


def tdi_waveform(
    theta: ArrayLike | GBParams,
    *,
    q_ref: ArrayLike,
    T: ArrayLike,
    BW: int,
    t0: ArrayLike = 0.0,
    fmt: str = "sangria",
    positions: ArrayLike | None = None,
    response_samples: int | None = None,
) -> WaveformResult:
    """Generate the float32 TDI spectrum for physical offset parameters.

    The float32 counterpart of :func:`binary_galactica.tdi_waveform`.
    ``theta`` is one ``GBParams`` or ``(8,)`` array ``(amp, f0 - q_ref/T,
    fdot, phi0, iota, psi, lam, beta)``: physical parameters whose frequency
    field is the offset in hertz from ``q_ref/T``, with ``T`` in seconds.
    :func:`split_physical` builds it from float64 parameters. All fields must
    be finite, ``amp`` positive, ``iota`` in ``[0, pi]`` and ``beta`` in
    ``[-pi/2, pi/2]``; ``f0*T = q_ref + (f0 - q_ref/T)*T`` must be positive. Use
    ``jax.vmap`` over ``theta`` and ``q_ref`` for multiple sources.

    The other arguments and the result are as for :func:`ucb_waveform`, with
    the default carrier. Dividing ``amp`` by ``s`` scales every channel by
    ``1/s``, which keeps inner products clear of float32 underflow.
    """
    response_samples = _options("onfly", response_samples)
    duration, t0, positions = _validation.observation(
        T, BW, t0, positions, dtype=np.float32
    )
    # Host scalar fields can be packed once by validation. Keep to_array's
    # broadcasting and traced packing for array-valued fields.
    if isinstance(theta, GBParams) and not all(np.isscalar(field) for field in theta):
        theta = theta.to_array()
    physical = _validation.finite_array(theta, "theta", dtype=np.float32)
    if physical.shape != (8,):
        raise ValueError(f"theta must have shape (8,); got {physical.shape}")
    q_ref = _single.bin_index(q_ref, "q_ref", margin=BW)
    if not isinstance(physical, jax.core.Tracer):
        p = np.asarray(physical)
        if p[0] <= 0:
            raise ValueError("amp must be positive")
        if not 0 <= p[4] <= np.pi:
            raise ValueError("iota must lie in [0, pi]")
        if not -np.pi / 2 <= p[7] <= np.pi / 2:
            raise ValueError("beta must lie in [-pi/2, pi/2]")
        if not isinstance(duration, jax.core.Tracer):
            # The same float32 product as the compiled conversion.
            with np.errstate(over="ignore"):
                offset = np.float32(p[1]) * np.float32(duration)
            if not abs(offset) < _single.OFFSET_LIMIT:
                raise ValueError(
                    f"(f0 - q_ref/T)*T must have magnitude below 2**24; got {offset}"
                )
            _single.check_frequency(offset, q_ref, duration, BW=BW)
    return _single._tdi_waveform(
        physical,
        q_ref,
        T=duration,
        BW=int(BW),
        t0=t0,
        fmt=_format(fmt),
        positions=positions,
        response_samples=response_samples,
        disabled=_single._DISABLED,
    )


def _duration(T):
    """Validate a host duration in seconds."""
    duration = np.asarray(T, dtype=np.float64)
    if duration.shape != () or not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"T must be a positive finite scalar; got {T!r}")
    return float(duration)


def _reference_bins(scaled_frequency, error=0.0):
    """Return ``floor(f0*T)`` as a Python int or int32 array, and the offset.

    ``f0*T`` is exactly ``scaled_frequency + error``, where ``error`` is at
    most half an ulp of ``scaled_frequency``.
    """
    q_ref = np.floor(scaled_frequency)
    # A whole scaled frequency with a negative low part lies in the bin below.
    q_ref = np.where((q_ref == scaled_frequency) & (error < 0), q_ref - 1, q_ref)
    if np.any(q_ref > _single.INT32_MAX):
        raise ValueError("f0*T must lie below 2**31")
    offset = (scaled_frequency - q_ref) + error
    q_ref = q_ref.astype(np.int32)
    return (int(q_ref) if q_ref.ndim == 0 else q_ref), offset


def _halves(x):
    """Split float64 ``x`` into high and low parts of at most 26 bits each."""
    scaled = 134_217_729.0 * x  # 2**27 + 1
    high = scaled - (scaled - x)
    return high, x - high


def _exact_product(a, b):
    """Return float64 ``(p, e)`` with ``p + e`` exactly ``a*b`` (Dekker).

    NumPy rounds every operation separately, as the correction requires.
    """
    p = a * b
    a_high, a_low = _halves(a)
    b_high, b_low = _halves(b)
    error = ((a_high * b_high - p) + a_high * b_low + a_low * b_high) + a_low * b_low
    return p, error


def _float32(values, name):
    """Round host values to float32, refusing any that overflow."""
    with np.errstate(over="ignore"):
        rounded = values.astype(np.float32)
    if not np.all(np.isfinite(rounded)):
        raise ValueError(f"{name} must be representable in float32")
    return rounded


def split_native(
    params: HostArrayLike, T: HostArrayLike
) -> tuple[int | NDArray[np.int32], NDArray[np.float32]]:
    """Split float64 native coordinates into ``(q_ref, float32 params)``.

    ``params`` has trailing length 7, 8 or 9 in the native layout of
    :func:`binary_galactica.ucb_waveform`, with ``params[..., 0] = f0*T``;
    leading axes hold sources, such as a catalogue. ``q_ref =
    floor(f0*T)`` is a Python integer for one source and an int32 array
    otherwise. The float32 result has ``params[..., 0] = f0*T - q_ref`` in
    ``[0, 1)``, within half a float32 ulp (an offset that would round up to
    one takes the largest float32 below one), and every other entry rounded.

    ``T`` is the duration in seconds that scales the coordinates. It is only
    checked: ``f0 = params[0]/T`` must be positive and finite, as in the
    float64 function. Native coordinates count cycles over the observation,
    so they keep their meaning when the float32 functions round ``T``.
    This is a host NumPy helper; it does not accept JAX tracers.
    """
    duration = _duration(T)
    params = np.asarray(params, dtype=np.float64)
    if params.ndim == 0 or params.shape[-1] not in (7, 8, 9):
        raise ValueError(
            f"params must have trailing length 7, 8, or 9; got shape {params.shape}"
        )
    if not np.all(np.isfinite(params)):
        raise ValueError("params must be finite")
    scaled = params[..., 0]
    with np.errstate(over="ignore", divide="ignore"):
        frequency = scaled / duration
    if np.any(scaled <= 0) or not np.all(np.isfinite(frequency)):
        raise ValueError("f0 must be finite and positive")
    if np.any(np.abs(params[..., [1, 4]]) > 1):
        raise ValueError("sky and inclination cosines must lie in [-1, 1]")
    q_ref, offset = _reference_bins(scaled)
    result = _float32(params, "params")
    result[..., 0] = np.minimum(offset.astype(np.float32), _BELOW_ONE)
    return q_ref, result


def split_physical(
    theta: HostArrayLike | GBParams, T: HostArrayLike
) -> tuple[int | NDArray[np.int32], NDArray[np.float32]]:
    """Split float64 physical parameters into ``(q_ref, float32 theta)``.

    ``theta`` is a ``GBParams`` or array with trailing length 8, ordered
    ``(amp, f0, fdot, phi0, iota, psi, lam, beta)`` with ``f0`` in hertz;
    leading axes hold sources. The float32 functions receive ``T`` (seconds)
    rounded to float32, so the split uses that duration ``T32``: ``q_ref =
    floor(f0*T32)``, with the product formed exactly, is a Python integer for
    one source and an int32 array otherwise, and the returned frequency field
    is the offset ``f0 - q_ref/T32`` in hertz, which lies in ``[0, 1/T32)``
    before rounding. ``q_ref/T32`` plus that offset reproduces ``f0`` to
    within one float32 ulp of the offset. Other fields are rounded to float32.

    All fields must be finite, ``amp`` and ``f0`` positive, ``iota`` in
    ``[0, pi]`` and ``beta`` in ``[-pi/2, pi/2]``. This is a host NumPy
    helper; it does not accept JAX tracers.
    """
    with np.errstate(over="ignore"):
        duration = _duration(np.float32(_duration(T)))
    if isinstance(theta, GBParams):
        fields = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in theta))
        theta = np.stack(fields, axis=-1)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 0 or theta.shape[-1] != 8:
        raise ValueError(f"theta must have trailing length 8; got shape {theta.shape}")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must be finite")
    if np.any(theta[..., 0] <= 0) or np.any(theta[..., 1] <= 0):
        raise ValueError("amp and f0 must be positive")
    if np.any((theta[..., 4] < 0) | (theta[..., 4] > np.pi)):
        raise ValueError("iota must lie in [0, pi]")
    if np.any(np.abs(theta[..., 7]) > np.pi / 2):
        raise ValueError("beta must lie in [-pi/2, pi/2]")
    q_ref, offset = _reference_bins(*_exact_product(theta[..., 1], duration))
    result = _float32(theta, "theta")
    result[..., 1] = offset / duration
    return q_ref, result


__all__ = [
    "split_native",
    "split_physical",
    "tdi_response",
    "tdi_waveform",
    "template_on_window",
    "ucb_waveform",
]
