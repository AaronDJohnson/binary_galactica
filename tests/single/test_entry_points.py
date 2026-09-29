"""Each float32 entry point against its float64 counterpart at identical inputs.

Window placement, the arbitrary-time response (to twenty years, with and
without positions), the physical adapter, position caches and amplitude
rescaling through ``params[3]``.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import single as bg32
from tests.single.support import (
    DURATIONS,
    EPOCHS,
    FREQUENCIES,
    SIGNS,
    Source,
    assert_accurate,
    channels,
    chirping,
    errors,
    physical,
    source,
)

WINDOW_SOURCES = {
    "quarter year at 30 mHz, late": source(0.25, 3e-2, 10.0, (1, -1)),
    "four years at 10 mHz, late": source(4.0, 1e-2, 10.0, (-1, 1)),
    "ten years at 1 mHz": source(10.0, 1e-3, 0.0, (-1, -1)),
}


def windows(case: Source) -> dict[str, tuple[int, int]]:
    """Return ``(bin_lo, n_bins)`` windows around the band of ``case``."""
    qmin, width = case.carrier - case.bandwidth // 2, case.bandwidth
    return {
        "containing": (qmin - 11, width + 23),
        "cropped at both ends": (qmin + width // 4, width // 2),
        "overlapping the top": (qmin + width - 5, 40),
        "overlapping the bottom": (qmin - 30, 37),
        "disjoint": (qmin + width + 3, 16),
    }


@pytest.mark.parametrize("name", list(WINDOW_SOURCES))
def test_window_placement_matches_float64(name):
    """Windows hold the float32 band at the float64 bins, zero elsewhere.

    Windows that keep most of the band meet the accuracy target on their own.
    Those that catch only a few edge bins hold little power, so their error is
    measured against the whole band's norm instead.
    """
    case = WINDOW_SOURCES[name]
    # The float64 shim uses GLASS's default carrier; these sources share it.
    assert int(bg.ucb_waveform(case.native, **case.double_observation())["qmin"]) == (
        case.carrier - case.bandwidth // 2
    )
    band = case.single()
    bins = np.asarray(band["freq_bin_indices"])
    continuous = np.sqrt(np.float32(case.duration)) * channels(band)
    peak = np.max(np.abs(continuous))
    whole_band = None
    for label, (bin_lo, n_bins) in windows(case).items():
        single = bg32.template_on_window(
            case.params,
            q_ref=case.q_ref,
            bin_lo=bin_lo,
            n_bins=n_bins,
            **case.observation(),
        )
        double = bg.template_on_window(
            case.native, bin_lo=bin_lo, n_bins=n_bins, **case.double_observation()
        )
        assert single.shape == (2, n_bins) and single.dtype == np.complex64, label
        np.testing.assert_array_equal(
            np.flatnonzero(np.asarray(single)),
            np.flatnonzero(np.asarray(double)),
            err_msg=label,
        )
        # The window is the band itself, in continuous-Fourier units.
        inside = (bins >= bin_lo) & (bins < bin_lo + n_bins)
        expected = np.zeros((2, n_bins), dtype=np.complex64)
        expected[:, bins[inside] - bin_lo] = continuous[:, inside]
        np.testing.assert_allclose(
            single, expected, rtol=0, atol=1e-6 * peak, err_msg=label
        )
        if whole_band is None:
            assert label == "containing" and inside.all()
            whole_band = np.linalg.norm(np.asarray(double))
        difference = np.asarray(single, dtype=np.complex128) - np.asarray(double)
        assert np.linalg.norm(difference) <= 1e-4 * whole_band, label
        if label in ("containing", "cropped at both ends"):
            assert_accurate(single, double, label)
        if not inside.any():
            assert not np.any(np.asarray(single)), label


def test_window_with_cache_and_phase_format():
    case = WINDOW_SOURCES["four years at 10 mHz, late"]
    cache = np.asarray(
        bg.orbit_cache(float(case.duration), float(case.epoch), case.bandwidth)
    ).astype(np.float32)
    bin_lo, n_bins = windows(case)["cropped at both ends"]
    options = {"bin_lo": bin_lo, "n_bins": n_bins, "fmt": "phase"}
    single = bg32.template_on_window(
        case.params, q_ref=case.q_ref, positions=cache, **options, **case.observation()
    )
    double = bg.template_on_window(
        case.native,
        positions=cache.astype(np.float64),
        **options,
        **case.double_observation(),
    )
    assert_accurate(single, double)


#: Irregular times from before the epoch to twenty years, with a cluster of
#: samples near 6e8 s, rounded to float32.
_RNG = np.random.default_rng(7)
LATE = 48
TIMES = np.concatenate(
    [
        [-5e6, 0.0],
        np.sort(_RNG.uniform(0.0, 6.3e8, 150)),
        6.0e8 + np.linspace(0.0, 5e5, LATE),
    ]
).astype(np.float32)
#: Positions of a constellation rotated from the analytic default, so that
#: ignoring supplied positions would show.
KAPPA = 0.3
POSITIONS = np.asarray(
    bg.analytic_orbits(TIMES.astype(np.float64), kappa_0=KAPPA)
).astype(np.float32)


@pytest.mark.parametrize("given", [False, True], ids=["analytic", "positions"])
@pytest.mark.parametrize("frequency", FREQUENCIES)
@pytest.mark.parametrize("years", [0.25, 10.0])
def test_arbitrary_time_response_matches_float64(years, frequency, given):
    # Derivatives drift a few bins by twenty years.
    case = source(years, frequency, 20.0 - years, (1, -1))
    positions = POSITIONS if given else None
    single = bg32.tdi_response(
        case.params, TIMES, q_ref=case.q_ref, T=case.duration, positions=positions
    )
    double = bg.tdi_response(
        case.native,
        TIMES.astype(np.float64),
        T=float(case.duration),
        positions=None if positions is None else positions.astype(np.float64),
    )
    assert single.shape == (3, TIMES.size) and single.dtype == np.complex64
    assert_accurate(single, double, "all times")
    assert_accurate(np.asarray(single)[:, -LATE:], double[:, -LATE:], "near 6e8 s")
    if given:
        analytic = bg32.tdi_response(
            case.params, TIMES, q_ref=case.q_ref, T=case.duration
        )
        assert errors(analytic, double)[0] > 1e-2


def test_response_rounds_host_times_to_float32():
    case = source(1.0, 1e-2, 10.0)
    times = np.array([1.23456789e8, 6.0000001e8, 3.3e7])
    rounded = times.astype(np.float32)
    assert np.any(rounded.astype(np.float64) != times)
    host = bg32.tdi_response(case.params, times, q_ref=case.q_ref, T=case.duration)
    exact = bg32.tdi_response(case.params, rounded, q_ref=case.q_ref, T=case.duration)
    np.testing.assert_array_equal(np.asarray(host), np.asarray(exact))


PHYSICAL = [
    (years, frequency, epoch, signs)
    for (years, frequency, epoch), signs in zip(
        itertools.product(DURATIONS, FREQUENCIES, EPOCHS),
        itertools.cycle(SIGNS),
        strict=False,
    )
]


@pytest.mark.parametrize(
    "years, frequency, epoch, signs",
    PHYSICAL,
    ids=[f"T{y:g}yr-f{f * 1e3:g}mHz-t0{e:g}yr" for y, f, e, _ in PHYSICAL],
)
def test_physical_waveform_matches_float64(years, frequency, epoch, signs):
    case = source(years, frequency, epoch, signs)
    theta, double_theta = physical(case)
    single = bg32.tdi_waveform(theta, q_ref=case.q_ref, **case.observation())
    double = bg.tdi_waveform(double_theta, **case.double_observation())
    assert int(single["qmin"]) == int(double["qmin"])
    assert_accurate(channels(single), channels(double))


@pytest.mark.parametrize(
    "years, frequency, chirp_mass", [(4.0, 3e-2, 0.3), (10.0, 1e-2, 1.0)]
)
def test_physical_waveform_of_a_chirping_source(years, frequency, chirp_mass):
    """Chirps drifting about 3000 bins, where ``fdot*T**2`` must not be rounded.

    Rounding that product to float32 moves the phase by about 1e-4 cycles.
    """
    case = chirping(years, frequency, chirp_mass)
    theta, double_theta = physical(case)
    single = bg32.tdi_waveform(theta, q_ref=case.q_ref, **case.observation())
    double = bg.tdi_waveform(double_theta, **case.double_observation())
    assert int(single["qmin"]) == int(double["qmin"])
    assert_accurate(channels(single), channels(double))


@pytest.mark.parametrize("fmt", ["frequency", "phase"])
def test_physical_waveform_formats_and_cache(fmt):
    case = source(4.0, 3e-2, 10.0, (-1, 1))
    q_ref = case.q_ref
    theta, double_theta = physical(case)
    cache = np.asarray(
        bg.orbit_cache(float(case.duration), float(case.epoch), case.bandwidth)
    ).astype(np.float32)
    single = bg32.tdi_waveform(
        bg.GBParams(*theta),
        q_ref=q_ref,
        fmt=fmt,
        positions=cache,
        **case.observation(),
    )
    as_array = bg32.tdi_waveform(
        theta, q_ref=q_ref, fmt=fmt, positions=cache, **case.observation()
    )
    for key in single:
        np.testing.assert_array_equal(
            np.asarray(single[key]), np.asarray(as_array[key])
        )
    double = bg.tdi_waveform(
        double_theta,
        fmt=fmt,
        positions=cache.astype(np.float64),
        **case.double_observation(),
    )
    assert_accurate(channels(single), channels(double))


CACHED = list(itertools.product((0.25, 10.0), (1e-4, 3e-2), EPOCHS))


@pytest.mark.parametrize("rotated", [False, True], ids=["orbit_cache", "rotated"])
@pytest.mark.parametrize(
    "years, frequency, epoch",
    CACHED,
    ids=[f"T{y:g}yr-f{f * 1e3:g}mHz-t0{e:g}yr" for y, f, e in CACHED],
)
def test_position_caches(years, frequency, epoch, rotated):
    """Float32 and host float64 caches both mean their float32 values.

    The rotated constellation differs from the analytic orbit the functions
    use without a cache, so the cache must be used throughout.
    """
    case = source(years, frequency, epoch, (1, -1))
    if rotated:
        grid = float(case.epoch) + float(case.duration) * (
            np.arange(case.bandwidth) / case.bandwidth
        )
        cache = np.asarray(bg.analytic_orbits(grid, kappa_0=KAPPA))
    else:
        cache = np.asarray(
            bg.orbit_cache(float(case.duration), float(case.epoch), case.bandwidth)
        )
    assert cache.shape == (3, case.bandwidth, 3)
    single = case.single(positions=cache.astype(np.float32))
    host = case.single(positions=cache)
    for key in single:
        np.testing.assert_array_equal(np.asarray(host[key]), np.asarray(single[key]))
    double = case.double(positions=cache.astype(np.float32).astype(np.float64))
    assert_accurate(channels(single), channels(double))
    if rotated:
        assert errors(channels(case.single()), channels(double))[0] > 1e-2


def test_phase_format_band_through_zero_frequency():
    """A band reaching DC is zero at bin zero and ``conj(sangria/(2j*x))`` elsewhere.

    ``x = f/fstar`` at each bin. The source, near 0.7 microhertz, lies far below
    the validated range: there float32 error grows as the frequency falls, so
    the phase band is checked against the float32 sangria band instead.
    """
    case = dataclasses.replace(source(0.25, 1e-4, 0.0), q_ref=5, bandwidth=32)
    phase, sangria = case.single(fmt="phase"), case.single()
    bins = np.asarray(phase["freq_bin_indices"])
    double = case.double(fmt="phase")
    np.testing.assert_array_equal(bins, np.asarray(double["freq_bin_indices"]))
    assert bins[0] < 0 < bins[-1]
    dc = bins == 0
    assert np.all(channels(phase)[:, dc] == 0)
    assert np.all(channels(double)[:, dc] == 0)
    x = bins[~dc] / (float(case.duration) * bg.FSTAR)
    expected = np.conj(channels(sangria)[:, ~dc].astype(np.complex128) / (2j * x))
    np.testing.assert_allclose(
        channels(phase)[:, ~dc], expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )


SCALE = 1e-20


def rescaled(params: np.ndarray) -> tuple[np.ndarray, float]:
    """Subtract ``log(SCALE)`` from ``params[3]``; return the exact factor.

    The factor is the ratio of the two float32 amplitudes, ``exp`` of the
    difference of the float32 log amplitudes, evaluated in float64.
    """
    shifted = params.copy()
    shifted[3] = params[3] - np.float32(np.log(SCALE))
    return shifted, math.exp(float(params[3]) - float(shifted[3]))


def assert_rescaled(scaled, raw, factor, *, raw_underflows=True):
    """Require ``scaled`` to be ``raw/factor``, with squares clear of underflow.

    Squares of raw float32 strains fall below float32's normal range; those of
    continuous-Fourier windows do not. Rescaled values give an accurate
    float32 energy.
    """
    scaled64 = np.asarray(scaled, dtype=np.complex128)
    relative_l2, _ = errors(scaled64 * factor, raw)
    assert relative_l2 < 2e-6
    tiny = np.finfo(np.float32).tiny
    raw_energy = np.sum(np.abs(np.asarray(raw)) ** 2, dtype=np.float32)
    assert (raw_energy < tiny) == raw_underflows
    energy = np.sum(np.abs(np.asarray(scaled)) ** 2, dtype=np.float32)
    assert energy > tiny
    assert abs(energy / np.sum(np.abs(scaled64) ** 2) - 1) < 1e-5


def test_amplitude_rescaling_scales_every_entry_point():
    """Subtracting ``log(s)`` from ``params[3]`` divides every output by ``s``."""
    case = source(1.0, 1e-3, 10.0, (1, -1))
    params, factor = rescaled(case.params)
    assert factor == pytest.approx(SCALE, rel=1e-5)
    shifted = dataclasses.replace(case, params=params)
    for key in "XYZAE":
        assert_rescaled(shifted.single()[key], case.single()[key], factor)

    bin_lo, n_bins = windows(case)["containing"]

    def window(values):
        return bg32.template_on_window(
            values,
            q_ref=case.q_ref,
            bin_lo=bin_lo,
            n_bins=n_bins,
            **case.observation(),
        )

    scaled_window = window(params)
    assert_rescaled(scaled_window, window(case.params), factor, raw_underflows=False)
    assert np.all(scaled_window[np.asarray(window(case.params)) == 0] == 0)

    def response(values):
        return bg32.tdi_response(values, TIMES[:40], q_ref=case.q_ref, T=case.duration)

    assert_rescaled(response(params), response(case.params), factor)


def test_physical_amplitude_rescaling():
    case = source(1.0, 1e-3, 10.0, (1, -1))
    q_ref = case.q_ref
    theta, _ = physical(case)
    scaled_theta = theta.copy()
    scaled_theta[0] = theta[0] / np.float32(SCALE)
    factor = float(theta[0]) / float(scaled_theta[0])
    scaled = bg32.tdi_waveform(scaled_theta, q_ref=q_ref, **case.observation())
    raw = bg32.tdi_waveform(theta, q_ref=q_ref, **case.observation())
    for key in "XYZAE":
        assert_rescaled(scaled[key], raw[key], factor)
