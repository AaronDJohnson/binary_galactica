"""Float32 waveform accuracy across the validated range, at identical inputs.

The grid is T in {0.25, 1, 4, 10} years, f0 in {0.1, 1, 10, 30} mHz and t0 in
{0, 10} years, with every sign combination of the two frequency-derivative
terms and every TDI format: 384 waveforms. Durations and epochs are the
float32 values nearest those years. The float64 waveform receives the exact
float64 values of the float32 inputs, with its carrier pinned to the float32
default carrier, and A/E must agree to relative L2 1e-4 and mismatch 1e-8.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import single as bg32
from tests.single.support import (
    DURATIONS,
    EPOCHS,
    FREQUENCIES,
    SIGNS,
    assert_accurate,
    channels,
    chirping,
    errors,
    middle_bin,
    source,
)

CORNERS = list(itertools.product(DURATIONS, FREQUENCIES, EPOCHS))


def corner_id(corner):
    """Name a grid point by duration, frequency and epoch."""
    years, frequency, epoch = corner
    return f"T{years:g}yr-f{frequency * 1e3:g}mHz-t0{epoch:g}yr"


def assert_same_bins(single, double):
    """Require int32 bins equal to the pinned float64 bins."""
    assert np.asarray(single["qmin"]).dtype == np.int32
    assert np.asarray(single["freq_bin_indices"]).dtype == np.int32
    assert int(single["qmin"]) == int(double["qmin"])
    np.testing.assert_array_equal(
        np.asarray(single["freq_bin_indices"], dtype=np.int64),
        np.asarray(double["freq_bin_indices"]).astype(np.int64),
    )


@pytest.mark.parametrize("fmt", bg.FORMATS)
@pytest.mark.parametrize("corner", CORNERS, ids=[corner_id(c) for c in CORNERS])
def test_waveform_meets_the_target_across_the_range(corner, fmt):
    for signs in SIGNS:
        case = source(*corner, signs)
        single = case.single(fmt=fmt)
        double = case.double(fmt=fmt)
        assert {np.asarray(single[k]).dtype for k in "XYZAE"} == {
            np.dtype(np.complex64)
        }
        assert_same_bins(single, double)
        assert_accurate(channels(single), channels(double), f"signs {signs}")


@pytest.mark.parametrize("corner", CORNERS, ids=[corner_id(c) for c in CORNERS])
def test_grid_bands_hold_the_signal(corner):
    """The chosen bands keep all but a small fraction of the power off the edges.

    Otherwise the comparison above would be of aliased spectra.
    """
    for signs in SIGNS:
        case = source(*corner, signs)
        power = np.abs(channels(case.double())) ** 2
        edge = case.bandwidth // 8
        outside = power[:, :edge].sum() + power[:, -edge:].sum()
        assert outside < 1e-2 * power.sum(), f"signs {signs}"


BASE = source(4.0, 1e-2, 10.0, (1, -1))


@pytest.mark.parametrize("shift", [-100_000, -1000, -2, 1, 3, 100_000])
def test_any_reference_bin_represents_the_source(shift):
    """``q_ref`` need not be ``floor(f0*T)``: the offset may exceed one or be negative.

    The float32 offset then rounds more coarsely, so each variant is compared
    with float64 at its own float32 inputs; the default carrier stays at
    ``q_ref + floor(offset)``.
    """
    params = BASE.params.copy()
    params[0] = np.float32(float(BASE.params[0]) - shift)
    case = dataclasses.replace(BASE, q_ref=BASE.q_ref + shift, params=params)
    single, double = case.single(), case.double()
    assert int(single["qmin"]) == BASE.carrier - BASE.bandwidth // 2
    assert_same_bins(single, double)
    assert_accurate(channels(single), channels(double))


@pytest.mark.parametrize(
    "offset",
    [0.0, 1e-7, 0.5, np.nextafter(np.float32(1), np.float32(0))],
    ids=["zero", "tiny", "half", "below one"],
)
def test_offsets_across_the_bin(offset):
    params = BASE.params.copy()
    params[0] = offset
    case = dataclasses.replace(BASE, params=params)
    single, double = case.single(), case.double()
    assert int(single["qmin"]) == BASE.q_ref - BASE.bandwidth // 2
    assert_same_bins(single, double)
    assert_accurate(channels(single), channels(double))


@pytest.mark.parametrize("length", [7, 8])
def test_shorter_native_vectors(length):
    """Without the second or both derivatives, as in the float64 layout."""
    case = source(10.0, 3e-2, 10.0, length=length)
    single, double = case.single(), case.double()
    assert_same_bins(single, double)
    assert_accurate(channels(single), channels(double))


@pytest.mark.parametrize("epoch", [-10.0, -3.0, 0.5, 20.0])
def test_epochs_outside_the_grid(epoch):
    """Negative and later epochs follow the same float64 convention."""
    case = source(4.0, 1e-2, epoch, (1, -1))
    assert_accurate(channels(case.single()), channels(case.double()))


@pytest.mark.parametrize("drift", [(10.0, 3.0), (30.0, 10.0)])
@pytest.mark.parametrize("years, frequency", [(0.25, 1e-4), (0.25, 3e-2), (1.0, 1e-2)])
def test_larger_drifts_in_a_wider_band(years, frequency, drift):
    """Tens of bins of drift at a late epoch, in a band wide enough to hold it.

    Here the float64 phase formula evaluated directly in float32 misses the
    target; the epoch re-expansion does not.
    """
    case = source(years, frequency, 10.0, (1, 1), drift=drift)
    assert_accurate(channels(case.single()), channels(case.double()))


@pytest.mark.parametrize(
    "years, frequency, chirp_mass, epoch",
    [(4.0, 3e-2, 0.3, 0.0), (10.0, 1e-2, 1.0, 0.0), (0.25, 3e-2, 1.0, 10.0)],
)
def test_chirping_sources(years, frequency, chirp_mass, epoch):
    """Gravitational-radiation derivatives, drifting hundreds to thousands of bins.

    The phase polynomial's terms then reach thousands of cycles, which float32
    rounds by about 1e-4 cycles per sample (relative L2 2.8e-4 at four years
    and 30 mHz before the per-sample sum was made in pairs). At the late epoch
    the carrier is pinned mid-observation, thousands of bins from the default.
    """
    case = chirping(years, frequency, chirp_mass, epoch)
    carrier = {"q": middle_bin(case)} if epoch else {}
    for fmt in bg.FORMATS:
        single = case.single(fmt=fmt, **carrier)
        double = case.double(fmt=fmt, **carrier)
        assert_same_bins(single, double)
        assert_accurate(channels(single), channels(double), fmt)


@pytest.mark.parametrize("step", [-1000, 1000])
def test_distant_carrier_override(step):
    """A carrier a thousand bins from the source, in a band that still holds it.

    The restored phase then runs through a thousand cycles.
    """
    case = source(4.0, 1e-2, 0.0, bandwidth=4096)
    # f0*T is whole here; a fractional offset also exercises the residual.
    params = case.params.copy()
    params[0] = np.float32(0.3719)
    case = dataclasses.replace(case, params=params)
    q = case.carrier + step
    single, double = case.single(q=q), case.double(q=float(q))
    assert int(single["qmin"]) == q - case.bandwidth // 2
    assert_same_bins(single, double)
    assert_accurate(channels(single), channels(double))


@pytest.mark.parametrize("step", [-5, -1, 1, 7])
@pytest.mark.parametrize("kind", [int, np.int32, float])
def test_carrier_override_moves_the_band(step, kind):
    """An integral ``q`` override agrees with the float64 override."""
    q = BASE.carrier + step
    single = BASE.single(q=kind(q))
    double = BASE.double(q=float(q))
    assert int(single["qmin"]) == q - BASE.bandwidth // 2
    assert_same_bins(single, double)
    assert_accurate(channels(single), channels(double))


def test_override_at_the_default_carrier_changes_nothing():
    default = BASE.single()
    pinned = BASE.single(q=BASE.carrier)
    for key in default:
        np.testing.assert_array_equal(np.asarray(pinned[key]), np.asarray(default[key]))


@pytest.mark.parametrize("years", [0.25, 10.0])
def test_automatic_grid_matches_the_explicit_count(years):
    """``response_samples=None`` picks the count these tests pass explicitly."""
    case = source(years, 3e-2, 10.0)
    automatic = case.single(response_samples=None)
    relative_l2, _ = errors(channels(automatic), channels(case.single()))
    assert relative_l2 < 1e-6
    assert_accurate(channels(automatic), channels(case.double(response_samples=None)))


def test_host_duration_and_epoch_are_rounded_to_float32():
    """Python-float ``T`` and ``t0`` mean their float32 values, as documented."""
    case = source(4.0, 1e-2, 10.0)
    exact = case.single()
    host = bg32.ucb_waveform(
        case.params,
        q_ref=case.q_ref,
        T=float(case.duration) + 0.1,
        t0=float(case.epoch) + 1.0,
        BW=case.bandwidth,
        response_samples=case.samples,
    )
    for key in exact:
        np.testing.assert_array_equal(np.asarray(host[key]), np.asarray(exact[key]))
