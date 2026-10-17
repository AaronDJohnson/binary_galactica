"""Response grids at least as fine as the FFT grid evaluate the response there.

The on-fly envelope is needed only at the ``BW`` FFT times. Interpolating a
response grid of ``BW`` or more samples onto them costs more than evaluating
the response at those times directly, and is less accurate: the automatic 512
samples of a four-year observation interpolate onto a 128-bin band with a
relative error near 1e-10, against rounding for direct evaluation. A count at
or above ``BW`` therefore evaluates the response at ``slow_time_grid(T, t0,
BW)``, the times at which the source phase is restored, and passes a legacy
position cache, which is sampled on those times, through unchanged. Every such
count gives the same envelope, so the automatic switch over a traced duration
compiles them as one branch.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica.glass import waveform as glass_waveform
from binary_galactica.glass.constants import PI2, YEAR
from binary_galactica.glass.onfly import tdi_response as raw_response
from binary_galactica.glass.orbit import slow_time_grid
from binary_galactica.glass.sparse import interpolate_uniform
from binary_galactica.glass.tdi import xyz_to_ae
from tests.conftest import nested_jaxprs

DURATION = 4 * YEAR
EPOCH = 12_345.0
BW = 128
#: A generic 1 mHz source with both frequency derivatives, nine entries long.
NATIVE = np.array(
    [
        1e-3 * DURATION + 0.25,
        0.3,
        4.1,
        math.log(1e-22),
        0.4,
        0.7,
        1.1,
        0.5,
        0.2,
    ]
)
#: The rotation of the constellation in supplied caches, so that ignoring
#: them would show.
KAPPA = 0.3


def relative_error(actual, expected) -> float:
    """Return the relative L2 difference, formed in float64."""
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def inputs(dtype, cached, *, seconds=DURATION, bandwidth=BW):
    """Return ``params``, the duration, ``t0`` and positions in the working dtype."""
    real = np.dtype(dtype)
    params = jnp.asarray(NATIVE, dtype=real)
    duration = jnp.asarray(seconds, dtype=real)
    t0 = jnp.asarray(EPOCH, dtype=real)
    positions = None
    if cached:
        grid = EPOCH + seconds * np.arange(bandwidth) / bandwidth
        orbit = np.asarray(bg.analytic_orbits(grid, kappa_0=KAPPA))
        positions = jnp.asarray(orbit, dtype=real)
    return params, duration, t0, positions


def mixed(response, channels):
    """Return X/Y/Z rows unchanged, or combined into A and E."""
    return response if channels == "xyz" else jnp.stack(xyz_to_ae(*response))


def interpolated(
    params, duration, t0, positions, count, *, bandwidth=BW, channels="xyz"
):
    """Return the envelope as built before direct evaluation, for any ``count``.

    Knots span ``[t0, t0 + T*(1 - 1/BW)]``, the first and last FFT times. A
    cache is interpolated onto them and the response interpolated back onto
    the FFT times.
    """
    real = params.dtype
    span = duration * (1 - 1 / bandwidth)
    knots = t0 + span * jnp.linspace(0.0, 1.0, count, dtype=real)
    geometry = None
    if positions is not None:
        geometry = glass_waveform._cached_positions(positions, knots, duration, t0)
    response = raw_response(params, knots, duration, geometry, reference_phase=True)
    response = mixed(response, channels)
    coordinates = jnp.arange(bandwidth, dtype=real) / (bandwidth - 1)
    return interpolate_uniform(response.T, coordinates).T


@pytest.mark.parametrize("channels", ["xyz", "ae"])
@pytest.mark.parametrize("cached", [False, True], ids=["analytic", "cache"])
@pytest.mark.parametrize("count", [BW, 4 * BW])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_envelope_is_the_response_on_the_fft_grid(dtype, count, cached, channels):
    """Bit for bit, evaluated eagerly operation by operation."""
    params, duration, t0, positions = inputs(dtype, cached)
    actual = glass_waveform._response_envelope(
        params, duration, BW, t0, positions, count, channels=channels
    )
    times = slow_time_grid(duration, t0, BW)
    response = raw_response(params, times, duration, positions, reference_phase=True)
    expected = mixed(response, channels)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


#: Relative L2 bounds of the count == BW envelope against the interpolated
#: one, whose knots are the FFT times up to rounding of the times and of the
#: interpolating polynomials. Here at most 7.0e-15 and 4.6e-6; over 0.25-10
#: years, 0.1-30 mHz and BW of 8-512, with and without a cache, at most
#: 4.1e-14 and 7.5e-6. Float32 rounds a four-year time by about 8 s, and
#: either envelope lies 3e-6 to 1e-5 from float64, the direct one never
#: farther. Times one FFT step late move it by 0.28, a wrong channel by 1.7.
EQUAL_COUNT_BOUND = {np.float64: 1e-13, np.float32: 2e-5}


@pytest.mark.parametrize("cached", [False, True], ids=["analytic", "cache"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_count_equal_to_bw_matches_the_interpolated_envelope(dtype, cached):
    """A regression guard: interpolating at the knots returns their samples."""
    params, duration, t0, positions = inputs(dtype, cached)
    actual = glass_waveform._response_envelope(params, duration, BW, t0, positions, BW)
    expected = interpolated(params, duration, t0, positions, BW)
    for row in range(3):
        error = relative_error(actual[row], expected[row])
        assert error <= EQUAL_COUNT_BOUND[dtype], (row, error)


@pytest.mark.parametrize(
    "years, bandwidth, count",
    [(4.0, 128, 512), (1.0, 64, 128), (4.0, 256, 512)],
)
def test_count_above_bw_is_closer_to_a_dense_reference(years, bandwidth, count):
    """Direct evaluation removes the interpolation error of the coarser grid.

    The reference interpolates 4096 knots, eight times the finest count here;
    at eighth order its interpolation error is far below rounding, and it
    differs from 8192 knots by at most 1.2e-14. The old construction at
    ``count`` knots differs from it by 7.2e-11 to 9.0e-11, the direct
    envelope by at most 1.2e-14.
    """
    params, duration, t0, _ = inputs(np.float64, False, seconds=years * YEAR)
    actual = glass_waveform._response_envelope(
        params, duration, bandwidth, t0, None, count
    )
    previous = interpolated(params, duration, t0, None, count, bandwidth=bandwidth)
    reference = interpolated(params, duration, t0, None, 4096, bandwidth=bandwidth)
    for row in range(3):
        direct = relative_error(actual[row], reference[row])
        old = relative_error(previous[row], reference[row])
        assert direct <= 1e-13, (row, direct)
        assert old >= 100 * direct, (row, old, direct)
        # The change is the old interpolation error, not a new one.
        assert relative_error(actual[row], previous[row]) <= 2 * old, row


def dense_sangria(params, duration, bandwidth, epoch):
    """Raw X/Y/Z on the FFT grid from the public response, without interpolation.

    Uses the source-phase demodulated response, not the reference-spacecraft
    one, restores the phase in NumPy and transforms with NumPy.
    """
    times = epoch + duration * np.arange(bandwidth) / bandwidth
    carrier = np.trunc((params[0] / duration) * duration)
    u = times / duration
    cycles = (params[0] - carrier) * u + 0.5 * params[7] * u**2 + params[8] * u**3 / 6
    phase = PI2 * cycles - params[6]
    response = np.asarray(bg.tdi_response(params, times, T=duration))
    spectra = np.fft.fftshift(np.fft.fft(response * np.exp(1j * phase)), axes=-1)
    return spectra * (np.sqrt(duration) / (2 * bandwidth))


def test_four_year_band_matches_the_dense_construction():
    """The automatic 512 samples at four years and BW=128, through the public call.

    Interpolating them onto the band was 7.4e-11 from this construction in
    0.6.0; evaluating the response on the band agrees to 2.9e-15.
    """
    waveform = bg.ucb_waveform(NATIVE, T=DURATION, t0=EPOCH, BW=BW)
    expected = dense_sangria(NATIVE, DURATION, BW, EPOCH)
    for index, channel in enumerate("XYZ"):
        error = relative_error(waveform[channel], expected[index])
        assert error <= 1e-13, (channel, error)


def test_orbit_cache_reproduces_the_analytic_orbit():
    """A cache sampled on the FFT times enters unchanged, not interpolated.

    Interpolating the 128-sample cache cubically onto the 512 automatic knots
    moved the waveform by 3.9e-5 from the analytic orbit; the cache's own
    values reproduce it to 1.2e-14.
    """
    cache = bg.orbit_cache(DURATION, EPOCH, BW)
    cached = bg.ucb_waveform(NATIVE, T=DURATION, t0=EPOCH, BW=BW, positions=cache)
    analytic = bg.ucb_waveform(NATIVE, T=DURATION, t0=EPOCH, BW=BW)
    for channel in "XYZ":
        error = relative_error(cached[channel], analytic[channel])
        assert error <= 1e-13, (channel, error)


def test_explicit_counts_at_or_above_bw_give_one_waveform():
    """Every count of at least ``BW`` compiles the same program."""
    first = bg.ucb_waveform(NATIVE, T=DURATION, BW=BW, response_samples=BW)
    for count in (BW + 2, 2048):
        other = bg.ucb_waveform(NATIVE, T=DURATION, BW=BW, response_samples=count)
        for name, value in first.items():
            np.testing.assert_array_equal(other[name], value, err_msg=name)


def switch_branches(bandwidth):
    """Return the branches of the automatic switch, or ``None`` without one."""
    params, duration, t0, _ = inputs(np.float64, False, bandwidth=bandwidth)
    jaxpr = jax.make_jaxpr(
        lambda traced: glass_waveform._response_envelope(
            params, traced, bandwidth, t0, None, None
        )
    )(duration)
    conds = [
        equation
        for inner in nested_jaxprs(jaxpr)
        for equation in inner.eqns
        if equation.primitive.name == "cond"
    ]
    if not conds:
        return None
    (cond,) = conds
    return cond.params["branches"]


def interpolates(branch) -> bool:
    """Return whether a branch interpolates, as a windowing convolution."""
    return any(
        equation.primitive.name == "conv_general_dilated"
        for inner in nested_jaxprs(branch)
        for equation in inner.eqns
    )


@pytest.mark.parametrize(
    "bandwidth, interpolated_counts",
    [(4096, 7), (2048, 6), (256, 3), (128, 2), (64, 1)],
)
def test_switch_compiles_one_direct_branch(bandwidth, interpolated_counts):
    """Counts below ``BW`` interpolate; all the rest share one direct branch."""
    branches = switch_branches(bandwidth)
    direct = 0 if interpolated_counts == 7 else 1
    assert len(branches) == interpolated_counts + direct
    assert [interpolates(b) for b in branches[:interpolated_counts]] == [
        True
    ] * interpolated_counts
    assert not any(interpolates(b) for b in branches[interpolated_counts:])


@pytest.mark.parametrize("bandwidth", [2, 8, 32])
def test_switch_vanishes_when_every_count_is_direct(bandwidth):
    """At ``BW`` up to 32, the smallest automatic count, nothing is selected."""
    assert switch_branches(bandwidth) is None


def test_traced_duration_selects_the_host_count():
    """The clamped switch picks the branch the host count compiles, at every bucket.

    Adjacent interpolated grids differ by about 1e-10 here, so agreement to
    1e-13 identifies the branch.
    """
    params, _, t0, _ = inputs(np.float64, False)
    switched = jax.jit(
        lambda duration: glass_waveform._response_envelope(
            params, duration, BW, t0, None, None
        )
    )
    fixed = jax.jit(
        lambda duration, count: glass_waveform._response_envelope(
            params, duration, BW, t0, None, count
        ),
        static_argnums=1,
    )
    counts = set()
    for years in (0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0, 20.0):
        duration = jnp.asarray(years * YEAR)
        count = glass_waveform._automatic_samples(years * YEAR, np.float64)
        counts.add(count)
        error = relative_error(switched(duration), fixed(duration, count))
        assert error <= 1e-13, (years, count, error)
    assert counts == set(glass_waveform._AUTO_RESPONSE_SAMPLES)


@pytest.mark.parametrize("cached", [False, True], ids=["analytic", "cache"])
@pytest.mark.parametrize("fmt", bg.FORMATS)
def test_window_on_direct_grid_places_ucb_waveform_channels(fmt, cached):
    """A and E mixed from the directly evaluated rows, as in the window."""
    positions = None
    if cached:
        _, _, _, cache = inputs(np.float64, True, seconds=YEAR)
        positions = np.asarray(cache)
    options = {
        "T": YEAR,
        "t0": EPOCH,
        "BW": BW,
        "fmt": fmt,
        "positions": positions,
        "response_samples": 2 * BW,
    }
    waveform = bg.to_continuous_ft(bg.ucb_waveform(NATIVE, **options), YEAR)
    bin_lo, n_bins = int(waveform["qmin"]) + 3, BW + 8
    actual = np.asarray(
        bg.template_on_window(NATIVE, bin_lo=bin_lo, n_bins=n_bins, **options)
    )
    bins = np.asarray(waveform["freq_bin_indices"]).astype(np.int64) - bin_lo
    inside = (bins >= 0) & (bins < n_bins)
    expected = np.zeros((2, n_bins), dtype=np.complex128)
    for row, channel in enumerate("AE"):
        expected[row, bins[inside]] = np.asarray(waveform[channel])[inside]
    assert relative_error(actual, expected) <= 1e-14
