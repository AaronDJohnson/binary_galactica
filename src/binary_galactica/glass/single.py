"""Float32 TDI on the fly, with the frequency split at an integer reference bin.

Float32 cannot hold ``f0*T``: near 1e6-1e7 cycles it resolves the frequency
only to 0.06-0.5 cycles, and the waveform error reaches order one. Callers
therefore pass an integer ``q_ref`` and a float32 offset
``params[0] = f0*T - q_ref``. The response envelope is the float64 path's
dtype-generic kernel run in float32: there ``f0`` enters only through
relative quantities, and the reference phase is removed before sparse
interpolation, so rounding a knot time merely resamples a smooth envelope.

Three reformulations keep the remaining large quantities out of float32:

``carrier``
    The residual ``offset - floor(offset)`` is exact in float32; no product
    as large as ``f0*T`` enters the restored phase.
``epoch``
    The FFT-grid phase is re-expanded about ``u0 = t0/T`` as
    ``c0 + c1*v + c2*v**2 + c3*v**3`` with ``v = n/BW``. This is
    algebraically the float64 formula, including its epoch convention. The
    coefficients and the per-sample sum are float32 pairs reduced modulo one
    cycle, because the terms reach thousands of cycles for a strongly
    chirping source or a distant carrier override.
``orbit``
    The full Doppler delay ``k.x0/c`` (about 500 s) needs the orbital phase to
    about 1e-7 cycles. ``t/YEAR`` is reduced with float32 pairs: on the FFT
    grid from ``(t0, T, n/BW)`` without rounding the times, and at arbitrary
    response times from their float32 values.

``_without`` disables them one at a time for diagnostics only.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from binary_galactica.parameters import to_glass_vector

from . import _twofloat as tf
from . import _validation
from .constants import FSTAR, PI2, YEAR
from .onfly import reference_phase_shift
from .onfly import tdi_response as _raw_response
from .orbit import positions_from_phase, slow_time_grid
from .tdi import xyz_to_ae
from .waveform import _response_envelope, _source_valid

INT32_MIN = int(np.iinfo(np.int32).min)
INT32_MAX = int(np.iinfo(np.int32).max)
#: Offsets must be smaller than this, where float32 still resolves fractions.
#: The same bound applies to a carrier override's distance from the source.
OFFSET_LIMIT = 2**24
#: Reformulations that :func:`_without` can disable.
REFORMULATIONS = ("carrier", "epoch", "orbit")

#: Disabled reformulations, read by the public functions as a static argument.
_DISABLED: frozenset[str] = frozenset()

#: Orbital cycles per second, ``1/YEAR``, as a float32 pair.
_ORBIT_RATE = tf.constant(1.0 / YEAR)
_SIXTH = tf.constant(1.0 / 6.0)


@contextmanager
def _without(*names: str) -> Iterator[None]:
    """Evaluate public float32 calls without the named reformulations.

    For diagnostics only; the results are less accurate by design.
    """
    global _DISABLED
    unknown = set(names) - set(REFORMULATIONS)
    if unknown:
        raise ValueError(f"unknown reformulations {sorted(unknown)}")
    previous = _DISABLED
    _DISABLED = frozenset(names)
    try:
        yield
    finally:
        _DISABLED = previous


def bin_index(value, name, *, margin=0, integral_floats=False):
    """Validate an integer Fourier-bin scalar within int32 range, less ``margin``.

    Concrete values return an ``np.int32``; a traced value must be an int32
    scalar and is returned unchanged. Host floats with integral values are
    accepted only when ``integral_floats`` is set.
    """
    if isinstance(value, jax.core.Tracer):
        # A Python int traced with x64 enabled is int64; casting could wrap it.
        if value.shape != () or value.dtype != jnp.int32:
            raise ValueError(
                f"a traced {name} must be an int32 scalar; got "
                f"{value.aval.str_short()}. Pass numpy.int32 values into jit."
            )
        return value
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    else:
        array = np.asarray(value)
        if array.shape != ():
            raise ValueError(f"{name} must be a scalar; got shape {array.shape}")
        if array.dtype.kind in "iu" or (
            integral_floats
            and array.dtype.kind == "f"
            and np.isfinite(array)
            and array == np.floor(array)
        ):
            number = int(array)
        else:
            raise ValueError(f"{name} must be an integer; got {value!r}")
    if not INT32_MIN + margin <= number <= INT32_MAX - margin:
        raise ValueError(
            f"{name} must lie within int32 range, {margin} bins from either end; "
            f"got {number}"
        )
    return np.int32(number)


def offset_parameters(params):
    """Validate float32 offset coordinates without extracting tracers."""
    params = _validation.finite_array(params, "params", dtype=np.float32)
    if params.ndim != 1 or params.shape[0] not in (7, 8, 9):
        raise ValueError(
            "params must be a one-dimensional array with length 7, 8, or 9; "
            f"got shape {params.shape}"
        )
    if not isinstance(params, jax.core.Tracer):
        concrete = np.asarray(params)
        if np.any(np.abs(concrete[[1, 4]]) > 1):
            raise ValueError("sky and inclination cosines must lie in [-1, 1]")
        if abs(concrete[0]) >= OFFSET_LIMIT:
            raise ValueError(
                "params[0] must be the offset f0*T - q_ref, of magnitude below "
                f"2**24; got {concrete[0]}"
            )
    return params


def check_frequency(offset, q_ref, T, *, BW=None, q=None):
    """Eagerly require a positive, finite ``f0`` and representable carrier bins.

    ``offset`` is the float32 ``f0*T - q_ref``. Skipped for traced values,
    which the compiled functions mask instead.
    """
    if any(isinstance(value, jax.core.Tracer) for value in (offset, q_ref, T, q)):
        return
    offset = np.float32(offset)
    whole = math.floor(offset)
    natural = int(q_ref) + whole
    residual = np.float32(offset - np.float32(whole))
    if natural < 0 or (natural == 0 and residual == 0):
        raise ValueError("f0 must be positive: q_ref + params[0] must exceed zero")
    # The compiled carrier is int32; beyond it every output would be NaN.
    if natural > INT32_MAX:
        raise ValueError(
            f"q_ref + floor(params[0]) must lie in int32 range; got {natural}"
        )
    with np.errstate(over="ignore", divide="ignore"):
        frequency = (np.float32(natural) + residual) / np.float32(T)
    if not np.isfinite(frequency):
        raise ValueError("f0 = (q_ref + params[0])/T must be finite")
    if BW is None:
        return
    carrier = natural if q is None else int(q)
    if abs(carrier - natural) >= OFFSET_LIMIT:
        raise ValueError(
            f"q must lie within 2**24 bins of q_ref + params[0]; got {carrier}"
        )
    if not INT32_MIN + BW // 2 <= carrier <= INT32_MAX - BW // 2 + 1:
        raise ValueError(f"the carrier bin {carrier} and its band exceed int32 range")


def _carrier(offset, q_ref, q):
    """Return the carrier bin, exact residual pair, float32 ``f0*T`` and validity.

    The natural carrier is ``q_ref + floor(offset)``, leaving a residual in
    ``[0, 1)``. An override ``q`` shifts that residual by an exact integer.
    Invalid or overflowing traced bins give ``valid = False``.
    """
    whole = jnp.floor(offset)
    residual = offset - whole
    in_range = jnp.abs(offset) < OFFSET_LIMIT
    steps = jnp.where(in_range, whole, 0.0).astype(jnp.int32)
    natural = q_ref + steps
    wrapped = ((steps > 0) & (natural < q_ref)) | ((steps < 0) & (natural > q_ref))
    positive = (natural > 0) | ((natural == 0) & (residual > 0))
    valid = in_range & ~wrapped & positive
    # One rounding when the natural carrier is below 2**24.
    scaled_frequency = natural.astype(jnp.float32) + residual
    if q is None:
        return natural, tf.pair(residual), scaled_frequency, valid
    shift = natural - q
    wrapped = ((q < 0) & (shift < natural)) | ((q > 0) & (shift > natural))
    valid = valid & ~wrapped & (jnp.abs(shift) < OFFSET_LIMIT)
    shift = jnp.where(valid, shift, 0).astype(jnp.float32)
    return q, tf.two_sum(residual, shift), scaled_frequency, valid


def orbital_phase(times):
    """GLASS guiding-centre phase ``PI2*t/YEAR`` in ``[-pi, pi]``, for float32 ``t``.

    The float32 time is taken as exact and reduced modulo one year with
    float32 pairs, rather than rounding ``t/YEAR`` near ten or twenty cycles.
    """
    cycles = tf.multiply(tf.pair(times), _ORBIT_RATE)
    return PI2 * tf.value(tf.fraction(cycles))


def grid_orbital_phase(T, t0, BW):
    """:func:`orbital_phase` at ``t0 + T*n/BW`` without rounding those times."""
    start = tf.fraction(tf.multiply(tf.pair(t0), _ORBIT_RATE))
    span = tf.multiply(tf.pair(T), _ORBIT_RATE)
    cycles = tf.add(start, tf.multiply(span, tf.constant(np.arange(BW) / BW)))
    return PI2 * tf.value(tf.fraction(cycles))


def epoch_coefficients(residual, params, T, t0, first=None):
    """Return pairs ``(c0 mod 1, c1, c2, c3)`` of the restored phase in cycles.

    With ``u0 = t0/T``, ``r = residual``, ``F = params[7]`` and
    ``G = params[8]``, ``r*u + F*u**2/2 + G*u**3/6`` at ``u = u0 + v`` equals
    ``c0 + c1*v + c2*v**2 + c3*v**3`` with ``c0 = r*u0 + F*u0**2/2 +
    G*u0**3/6``, ``c1 = r + F*u0 + G*u0**2/2``, ``c2 = (F + G*u0)/2`` and
    ``c3 = G/6``. ``c0`` reaches hundreds of cycles at late epochs, and ``c1``
    and ``c2`` thousands for a strongly chirping source or a distant carrier
    override, so all four stay pairs. ``first`` optionally gives ``F`` as a
    pair, for an ``F`` computed more precisely than its float32 ``params[7]``.
    """
    u0 = tf.divide(t0, T)
    zero = jnp.zeros((), dtype=jnp.float32)
    if first is None:
        first = tf.pair(params[7] if params.shape[0] > 7 else zero)
    G = params[8] if params.shape[0] > 8 else zero
    half_F, half_G = (0.5 * first[0], 0.5 * first[1]), tf.pair(0.5 * G)
    sixth_G = tf.multiply(tf.pair(G), _SIXTH)
    c0 = tf.multiply(
        u0,
        tf.add(residual, tf.multiply(u0, tf.add(half_F, tf.multiply(u0, sixth_G)))),
    )
    c1 = tf.add(residual, tf.multiply(u0, tf.add(first, tf.multiply(u0, half_G))))
    c2 = tf.add(half_F, tf.multiply(u0, half_G))
    return tf.fraction(c0), c1, c2, sixth_G


def grid_cycles(residual, params, T, t0, BW, disabled=(), first=None):
    """Return the restored cycles less the carrier on the FFT grid, modulo one.

    The re-expanded polynomial is summed per sample in pairs: its terms reach
    thousands of cycles when the frequency moves far from the carrier, where
    float32 alone rounds the phase by about 1e-4 cycles. Disabling ``epoch``
    evaluates the float64 formula in float32 instead, without the reduction.
    """
    if "epoch" in disabled:
        # The float64 formula evaluated directly in float32.
        u = slow_time_grid(T, t0, BW) / T
        cycles = tf.value(residual) * u
        if params.shape[0] > 7:
            cycles = cycles + 0.5 * params[7] * u**2
        if params.shape[0] > 8:
            cycles = cycles + params[8] * u**3 / 6.0
        return cycles
    c0, c1, c2, c3 = epoch_coefficients(residual, params, T, t0, first)
    v = tf.constant(np.arange(BW) / BW)
    cycles = tf.add(c2, tf.multiply(v, c3))
    cycles = tf.add(c1, tf.multiply(v, cycles))
    cycles = tf.add(c0, tf.multiply(v, cycles))
    # The pair arithmetic above is ~600 dependent float32 operations. Without
    # the barrier XLA fuses a copy into each downstream channel kernel, and
    # LLVM then takes 4-9 s per program instead of ~0.3 s. Values unchanged.
    return jax.lax.optimization_barrier(tf.value(tf.fraction(cycles)))


def _waveform(
    params,
    q_ref,
    T,
    BW,
    t0,
    fmt,
    positions,
    q,
    response_samples,
    disabled,
    first=None,
    channels="xyz",
):
    """Return unmasked float32 channels, int32 bins and their validity.

    ``first`` optionally gives ``params[7]`` as a more precise pair.
    ``channels="ae"`` forms A and E from the sparse response and returns only
    those with the bins, as in the float64 ``_ucb_waveform``.
    """
    carrier, residual, scaled_frequency, valid = _carrier(params[0], q_ref, q)
    if "carrier" in disabled:
        residual = tf.pair(scaled_frequency - carrier.astype(jnp.float32))
    valid = (
        valid
        & (q_ref >= INT32_MIN + BW)
        & (q_ref <= INT32_MAX - BW)
        & (carrier >= INT32_MIN + BW // 2)
        & (carrier <= INT32_MAX - BW // 2 + 1)
        & jnp.isfinite(t0)
    )
    native = params.at[0].set(scaled_frequency)
    valid = valid & _source_valid(native, T)
    if positions is not None:
        valid = valid & jnp.all(jnp.isfinite(positions))
    envelope = _response_envelope(
        native, T, BW, t0, positions, response_samples, channels=channels
    )
    geometry = positions
    if positions is None and "orbit" not in disabled:
        geometry = positions_from_phase(grid_orbital_phase(T, t0, BW))
    # The shift multiplies the delay by the frequency at t, so rounded times suffice.
    times = slow_time_grid(T, t0, BW)
    phase = (
        PI2 * grid_cycles(residual, params, T, t0, BW, disabled, first)
        - params[6]
        + reference_phase_shift(native, times, T, geometry)
    )
    # A complex exponential, as in glass.waveform._ucb_waveform.
    envelope = envelope * jnp.exp(1j * phase)
    spectra = jnp.fft.fftshift(jnp.fft.fft(envelope, axis=-1), axes=-1)
    spectra = spectra * (jnp.sqrt(T) / (2 * BW))
    qmin = carrier - BW // 2
    bins = qmin + jnp.arange(BW, dtype=jnp.int32)
    if fmt == "frequency":
        spectra = -1j * spectra
    elif fmt == "phase":
        dc = bins == 0
        x = jnp.where(dc, 1, bins).astype(jnp.float32) / (T * FSTAR)
        spectra = jnp.where(dc, 0j, spectra / (2j * x))
    result = {"qmin": qmin, "freq_bin_indices": bins}
    if channels == "ae":
        A, E = spectra
    else:
        X, Y, Z = spectra
        A, E = xyz_to_ae(X, Y, Z)
        result.update(X=X, Y=Y, Z=Z)
    if fmt == "phase":
        A, E = jnp.conj(A), jnp.conj(E)
    result.update(A=A, E=E)
    return result, valid


def _masked(result, valid):
    """Replace invalid bins by ``INT32_MIN`` and invalid values by NaN."""
    return {
        name: jnp.where(
            valid,
            value,
            np.int32(INT32_MIN) if value.dtype == jnp.int32 else jnp.nan,
        )
        for name, value in result.items()
    }


@partial(jax.jit, static_argnames=("BW", "fmt", "response_samples", "disabled"))
def _ucb_waveform(
    params, q_ref, *, T, BW, t0, fmt, positions, q, response_samples, disabled
):
    """Compile validated float32 native inputs into a masked waveform."""
    result, valid = _waveform(
        params, q_ref, T, BW, t0, fmt, positions, q, response_samples, disabled
    )
    return _masked(result, valid)


@partial(jax.jit, static_argnames=("BW", "fmt", "response_samples", "disabled"))
def _tdi_waveform(
    theta, q_ref, *, T, BW, t0, fmt, positions, response_samples, disabled
):
    """Convert physical offset coordinates inside JIT, then generate."""
    valid = (
        jnp.all(jnp.isfinite(theta))
        & (theta[0] > 0)
        & (theta[4] >= 0)
        & (theta[4] <= jnp.pi)
        & (jnp.abs(theta[7]) <= jnp.pi / 2)
    )
    native = jnp.where(valid, to_glass_vector(theta, T), jnp.nan)
    # fdot*T**2 reaches thousands of bins for a chirping source, where rounding
    # it to float32 would move the phase by about 1e-4 cycles.
    first = tf.multiply(tf.product(theta[2], T), tf.pair(T))
    result, source_valid = _waveform(
        native,
        q_ref,
        T,
        BW,
        t0,
        fmt,
        positions,
        None,
        response_samples,
        disabled,
        first,
    )
    return _masked(result, valid & source_valid)


@partial(
    jax.jit,
    static_argnames=("bin_lo", "n_bins", "BW", "fmt", "response_samples", "disabled"),
)
def _template_on_window(
    params,
    q_ref,
    *,
    T,
    bin_lo,
    n_bins,
    BW,
    t0,
    fmt,
    positions,
    response_samples,
    disabled,
):
    """Place continuous-Fourier A and E on a window, NaN throughout if invalid.

    Only A and E are formed, from the sparse response, and transformed.
    """
    result, valid = _waveform(
        params,
        q_ref,
        T,
        BW,
        t0,
        fmt,
        positions,
        None,
        response_samples,
        disabled,
        channels="ae",
    )
    scale = jnp.sqrt(T)
    index = result["freq_bin_indices"] - np.int32(bin_lo)
    # As in the float64 shim: send every outside bin to one dropped slot so
    # negative indices never wrap onto the window.
    inside = (index >= 0) & (index < n_bins)
    safe = jnp.where(inside, index, n_bins)
    zero = jnp.zeros((), dtype=jnp.complex64)
    out = jnp.zeros((2, n_bins), dtype=jnp.complex64)
    out = out.at[0, safe].add(jnp.where(inside, scale * result["A"], zero), mode="drop")
    out = out.at[1, safe].add(jnp.where(inside, scale * result["E"], zero), mode="drop")
    return jnp.where(valid, out, jnp.nan)


@partial(jax.jit, static_argnames=("disabled",))
def _tdi_response(params, q_ref, times, T, positions, disabled):
    """Evaluate the arbitrary-time response with an accurate orbital phase."""
    _, _, scaled_frequency, valid = _carrier(params[0], q_ref, None)
    native = params.at[0].set(scaled_frequency)
    valid = valid & _source_valid(native, T) & jnp.all(jnp.isfinite(times))
    if positions is not None:
        valid = valid & jnp.all(jnp.isfinite(positions))
    elif "orbit" not in disabled:
        positions = positions_from_phase(orbital_phase(times))
    response = _raw_response(native, times, T, positions)
    return jnp.where(valid, response, jnp.nan)


__all__ = [
    "INT32_MAX",
    "INT32_MIN",
    "OFFSET_LIMIT",
    "REFORMULATIONS",
    "bin_index",
    "check_frequency",
    "epoch_coefficients",
    "grid_cycles",
    "grid_orbital_phase",
    "offset_parameters",
    "orbital_phase",
]
