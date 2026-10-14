"""Float32 waveform accuracy against float64, at identical inputs.

The float64 waveform receives the exact float64 values of the float32 inputs,
with its carrier pinned to the float32 default carrier, and A/E must agree to
relative L2 1e-4 and mismatch 1e-8. The fast tests share one nine-entry
program, ``BASE``'s band and response grid, whose first compilation takes
seconds; duration and epoch are traced, so sources that differ only in those
reuse it. The slow tests cover the corners of the validated range, a strongly
chirping source and the automatic response grid.
"""

from __future__ import annotations

import dataclasses
import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica import single as bg32
from binary_galactica.glass import single as single_impl
from binary_galactica.glass.waveform import (
    _AUTO_RESPONSE_SAMPLES,
    _duration_boundaries,
)
from tests.single.support import (
    DURATIONS,
    EPOCHS,
    FREQUENCIES,
    SIGNS,
    YEAR,
    assert_accurate,
    channels,
    chirping,
    errors,
    source,
)


def assert_same_bins(single, double):
    """Require int32 bins equal to the pinned float64 bins."""
    assert np.asarray(single["qmin"]).dtype == np.int32
    assert np.asarray(single["freq_bin_indices"]).dtype == np.int32
    assert int(single["qmin"]) == int(double["qmin"])
    np.testing.assert_array_equal(
        np.asarray(single["freq_bin_indices"], dtype=np.int64),
        np.asarray(double["freq_bin_indices"]).astype(np.int64),
    )


#: The corners of the validated range: T in {0.25, 10} years, f0 in
#: {0.1, 30} mHz and t0 in {0, 10} years.
CORNERS = list(
    itertools.product(
        (DURATIONS[0], DURATIONS[-1]), (FREQUENCIES[0], FREQUENCIES[-1]), EPOCHS
    )
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "corner",
    CORNERS,
    ids=[f"T{y:g}yr-f{f * 1e3:g}mHz-t0{e:g}yr" for y, f, e in CORNERS],
)
def test_waveform_meets_the_target_across_the_range(corner):
    """Every sign combination of the derivative terms, in the phase format.

    The three formats share the band and differ only after the FFT; measured
    over these corners their errors agree to a few parts in a million, with
    the phase format's the largest (relative L2 at most 0.11 of the target).
    It is also the only one that reweights and conjugates the band, so it is
    the one checked. The band from :func:`binary_galactica.ucb_bw` must keep
    all but 1% of the float64 power off its outer eighths, or the comparison
    would be of aliased spectra; the phase format's ``1/f`` weight stays
    within 2% of its band-centre value in every one of these bands.
    """
    for signs in SIGNS:
        case = source(*corner, signs)
        single = case.single(fmt="phase")
        double = case.double(fmt="phase")
        assert {np.asarray(single[k]).dtype for k in "XYZAE"} == {
            np.dtype(np.complex64)
        }
        assert_same_bins(single, double)
        assert_accurate(channels(single), channels(double), f"signs {signs}")
        power = np.abs(channels(double)) ** 2
        edge = case.bandwidth // 8
        outside = power[:, :edge].sum() + power[:, -edge:].sum()
        assert outside < 1e-2 * power.sum(), f"signs {signs}: band edges"


#: One year at 10 mHz, ten years late, with both derivative terms: a 256-bin
#: band on 256 response samples.
BASE = source(1.0, 1e-2, 10.0, (1, -1))


def test_waveform_matches_float64():
    """A quarter year at 20 mHz, ten years late, where every reformulation matters.

    Evaluated on ``BASE``'s band and response grid, so it shares that program.
    """
    case = dataclasses.replace(
        source(0.25, 2e-2, 10.0, (1, -1)),
        bandwidth=BASE.bandwidth,
        samples=BASE.samples,
    )
    single, double = case.single(), case.double()
    assert_same_bins(single, double)
    assert_accurate(channels(single), channels(double))


def test_any_reference_bin_represents_the_source():
    """``q_ref`` need not be ``floor(f0*T)``: the offset may exceed one or be negative.

    The float32 offset then rounds more coarsely, so each variant is compared
    with float64 at its own float32 inputs; the default carrier stays at
    ``q_ref + floor(offset)``. Negative shifts make large positive offsets,
    positive shifts negative ones.
    """
    for shift in (-100_000, -1000, -2, 1, 3, 100_000):
        params = BASE.params.copy()
        params[0] = np.float32(float(BASE.params[0]) - shift)
        case = dataclasses.replace(BASE, q_ref=BASE.q_ref + shift, params=params)
        single, double = case.single(), case.double()
        assert int(single["qmin"]) == BASE.carrier - BASE.bandwidth // 2, shift
        assert_same_bins(single, double)
        assert_accurate(channels(single), channels(double), f"shift {shift}")


def test_offsets_across_the_bin():
    """Zero, tiny, half and the largest float32 offset below one."""
    below_one = np.nextafter(np.float32(1), np.float32(0))
    for offset in (0.0, 1e-7, 0.5, below_one):
        params = BASE.params.copy()
        params[0] = offset
        case = dataclasses.replace(BASE, params=params)
        single, double = case.single(), case.double()
        assert int(single["qmin"]) == BASE.q_ref - BASE.bandwidth // 2, offset
        assert_same_bins(single, double)
        assert_accurate(channels(single), channels(double), f"offset {offset}")


def test_shorter_native_vectors():
    """Without the second or both derivatives, as in the float64 layout."""
    for length in (7, 8):
        case = source(10.0, 3e-2, 10.0, length=length)
        single, double = case.single(), case.double()
        assert_same_bins(single, double)
        assert_accurate(channels(single), channels(double), f"length {length}")


def test_epochs_outside_the_grid():
    """Negative and later epochs follow the same float64 convention."""
    for epoch in (-10.0, -3.0, 0.5, 20.0):
        case = source(1.0, 1e-2, epoch, (1, -1))
        assert_accurate(channels(case.single()), channels(case.double()), epoch)


#: ``BASE`` without the second derivative: the carrier tests compile a second
#: program, which takes a fraction of a second with eight entries.
BASE8 = dataclasses.replace(BASE, params=BASE.params[:8])


def test_carrier_override_moves_the_band():
    """An integral ``q`` override agrees with the float64 override.

    Below and above the default carrier, as a Python int, an int32 scalar and
    an integral float.
    """
    for step, kind in itertools.product((-5, -1, 1, 7), (int, np.int32, float)):
        label = f"{kind.__name__} {step:+d}"
        q = BASE8.carrier + step
        single = BASE8.single(q=kind(q))
        double = BASE8.double(q=float(q))
        assert int(single["qmin"]) == q - BASE8.bandwidth // 2, label
        assert_same_bins(single, double)
        assert_accurate(channels(single), channels(double), label)


def test_override_at_the_default_carrier_changes_nothing():
    default = BASE8.single()
    pinned = BASE8.single(q=BASE8.carrier)
    for key in default:
        np.testing.assert_array_equal(np.asarray(pinned[key]), np.asarray(default[key]))


def test_host_duration_and_epoch_are_rounded_to_float32():
    """Python-float ``T`` and ``t0`` mean their float32 values, as documented."""
    exact = BASE.single()
    host = bg32.ucb_waveform(
        BASE.params,
        q_ref=BASE.q_ref,
        T=float(BASE.duration) + 0.1,
        t0=float(BASE.epoch) + 1.0,
        BW=BASE.bandwidth,
        response_samples=BASE.samples,
    )
    for key in exact:
        np.testing.assert_array_equal(np.asarray(host[key]), np.asarray(exact[key]))


@pytest.mark.slow
def test_chirping_sources():
    """Gravitational-radiation derivatives drifting 3200 bins in four years.

    0.3 solar masses at 30 mHz. The phase polynomial's terms then reach
    thousands of cycles, which float32 rounds by about 1e-4 cycles per sample
    (relative L2 2.8e-4 here before the per-sample sum was made in pairs).
    The phase format, as across the range.
    """
    case = chirping(4.0, 3e-2, 0.3)
    single = case.single(fmt="phase")
    double = case.double(fmt="phase")
    assert_same_bins(single, double)
    assert_accurate(channels(single), channels(double))


@pytest.mark.slow
def test_automatic_grid_matches_the_explicit_count():
    """``response_samples=None`` picks the count these tests pass explicitly."""
    case = source(0.25, 3e-2, 10.0)
    automatic = case.single(response_samples=None)
    relative_l2, _ = errors(channels(automatic), channels(case.single()))
    assert relative_l2 < 1e-6
    assert_accurate(channels(automatic), channels(case.double(response_samples=None)))


def test_automatic_grid_matches_float64_at_identical_durations():
    """Thresholds sit at 0.25-8 years, where float32 durations round across them."""
    counts = _AUTO_RESPONSE_SAMPLES

    def count(duration, dtype):
        boundaries = _duration_boundaries(np.dtype(dtype))
        return counts[int(jnp.sum(boundaries < jnp.asarray(duration, dtype)))]

    for years in (0.25, 0.5, 1, 2, 4, 8):
        duration = np.float32(years * YEAR)
        for _ in range(3):
            duration = np.nextafter(duration, np.float32(0))
        for _ in range(7):
            assert count(duration, np.float32) == count(float(duration), np.float64)
            duration = np.nextafter(duration, np.float32(np.inf))


def test_reformulation_switch_is_scoped():
    """The diagnostic switch restores every reformulation on exit."""
    assert frozenset() == single_impl._DISABLED
    with single_impl._without("orbit", "epoch"):
        assert {"orbit", "epoch"} == single_impl._DISABLED
    assert frozenset() == single_impl._DISABLED
    with pytest.raises(ValueError, match="unknown"), single_impl._without("bogus"):
        pass
