"""The host split helpers: exact round trips and end-to-end waveforms.

``split_native`` and ``split_physical`` turn float64 coordinates into an
integer reference bin and float32 coordinates. Recombining them must give the
float64 frequency to within one float32 ulp of the offset, checked here in
rational arithmetic, over catalogues and at bin boundaries.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import single as bg32
from tests.single.support import DURATIONS, YEAR, assert_accurate, channels

COUNT = 400


def native_catalogue(rng, duration: float) -> np.ndarray:
    """Return float64 native coordinates of random sources from 0.1 to 30 mHz."""
    frequency = 10 ** rng.uniform(-4, math.log10(3e-2), COUNT)
    return np.column_stack(
        [
            frequency * duration,
            rng.uniform(-1, 1, COUNT),
            rng.uniform(0, 2 * np.pi, COUNT),
            rng.uniform(-55, -45, COUNT),
            rng.uniform(-1, 1, COUNT),
            rng.uniform(0, np.pi, COUNT),
            rng.uniform(0, 2 * np.pi, COUNT),
            rng.normal(0, 3, COUNT),
            rng.normal(0, 1, COUNT),
        ]
    )


def physical_catalogue(rng) -> np.ndarray:
    """Return float64 physical parameters of random sources from 0.1 to 30 mHz."""
    return np.column_stack(
        [
            10 ** rng.uniform(-23, -21, COUNT),
            10 ** rng.uniform(-4, math.log10(3e-2), COUNT),
            rng.normal(0, 1e-17, COUNT),
            rng.uniform(0, 2 * np.pi, COUNT),
            rng.uniform(0, np.pi, COUNT),
            rng.uniform(0, np.pi, COUNT),
            rng.uniform(0, 2 * np.pi, COUNT),
            rng.uniform(-np.pi / 2, np.pi / 2, COUNT),
        ]
    )


def boundary_scaled_frequencies() -> list[float]:
    """Return ``f0*T`` values at and beside whole bins, and near the int32 limit."""
    return [
        7.0,
        7.0 - 2.0**-30,
        7.0 + 2.0**-40,
        1_234_567.0,
        np.nextafter(1_234_567.0, 0.0),
        np.nextafter(1_234_567.0, np.inf),
        2.0**31 - 1 + 0.25,
    ]


def assert_native_round_trip(q_ref, params, native):
    """Require ``q_ref + offset`` within one ulp of ``f0*T``, offsets in ``[0, 1)``.

    Every other entry must be the float32 rounding of the float64 value.
    """
    offset = params[..., 0]
    assert np.all((offset >= 0) & (offset < 1))
    np.testing.assert_array_equal(q_ref, np.floor(native[..., 0]))
    for q, value, scaled in zip(
        np.ravel(q_ref), np.ravel(offset), np.ravel(native[..., 0]), strict=True
    ):
        error = abs(Fraction(int(q)) + Fraction(float(value)) - Fraction(scaled))
        assert error <= Fraction(float(np.spacing(value))), (q, value, scaled)
    np.testing.assert_array_equal(params[..., 1:], native[..., 1:].astype(np.float32))


@pytest.mark.parametrize("years", DURATIONS)
def test_split_native_round_trips_a_catalogue(years):
    rng = np.random.default_rng(int(years * 4))
    duration = years * YEAR
    native = native_catalogue(rng, duration)
    native[: len(boundary_scaled_frequencies()), 0] = boundary_scaled_frequencies()
    q_ref, params = bg32.split_native(native, duration)
    assert isinstance(q_ref, np.ndarray) and q_ref.dtype == np.int32
    assert q_ref.shape == (COUNT,)
    assert params.dtype == np.float32 and params.shape == native.shape
    assert_native_round_trip(q_ref, params, native)


@pytest.mark.parametrize("scaled", boundary_scaled_frequencies())
@pytest.mark.parametrize("length", [7, 8, 9])
def test_split_native_single_sources(scaled, length):
    native = np.array([scaled, 0.2, 1.0, -50.0, 0.1, 0.3, 0.4, 2.5, -0.5])[:length]
    q_ref, params = bg32.split_native(native, 1e8)
    assert type(q_ref) is int
    assert params.shape == (length,) and params.dtype == np.float32
    assert_native_round_trip(q_ref, params, native)


def test_offsets_that_round_up_to_one_stay_below_it():
    native = np.array([7.0 - 2.0**-30, 0.2, 1.0, -50.0, 0.1, 0.3, 0.4])
    q_ref, params = bg32.split_native(native, 1e8)
    assert q_ref == 6
    assert params[0] == np.nextafter(np.float32(1), np.float32(0))


def assert_physical_round_trip(q_ref, split, theta, duration32):
    """Require ``q_ref/T32 + offset`` within one ulp of the offset of ``f0``.

    ``q_ref`` must be the floor of the exact ``f0*T32``, and every other field
    the float32 rounding of the float64 value.
    """
    duration = Fraction(duration32)
    for q, value, frequency in zip(
        np.ravel(q_ref), np.ravel(split[..., 1]), np.ravel(theta[..., 1]), strict=True
    ):
        exact = Fraction(frequency) * duration
        assert q == math.floor(exact), (q, frequency)
        error = abs(
            Fraction(int(q)) / duration + Fraction(float(value)) - Fraction(frequency)
        )
        assert error <= Fraction(float(np.spacing(value))), (q, value, frequency)
        # The offset in cycles lies in [0, 1) before rounding.
        assert 0 <= (exact - q) < 1
    np.testing.assert_array_equal(
        np.delete(split, 1, axis=-1), np.delete(theta, 1, axis=-1).astype(np.float32)
    )


@pytest.mark.parametrize("years", DURATIONS)
def test_split_physical_round_trips_a_catalogue(years):
    rng = np.random.default_rng(int(years * 4) + 1)
    duration = years * YEAR
    duration32 = float(np.float32(duration))
    theta = physical_catalogue(rng)
    # Frequencies at whole bins of the float32 duration, and just beside them.
    whole = np.array([7.0, 1_234_567.0, 3_000_001.0]) / duration32
    theta[:9, 1] = np.concatenate(
        [whole, np.nextafter(whole, 0.0), np.nextafter(whole, 1.0)]
    )
    q_ref, split = bg32.split_physical(theta, duration)
    assert isinstance(q_ref, np.ndarray) and q_ref.dtype == np.int32
    assert split.dtype == np.float32 and split.shape == theta.shape
    assert_physical_round_trip(q_ref, split, theta, duration32)


def test_split_physical_accepts_broadcast_parameters():
    theta = bg.GBParams(
        amp=1e-22,
        f0=np.array([1e-3, 2.5e-3, 2.9e-2]),
        fdot=1e-17,
        phi0=0.1,
        iota=0.5,
        psi=0.2,
        lam=1.0,
        beta=0.3,
    )
    duration = 4 * YEAR
    q_ref, split = bg32.split_physical(theta, duration)
    assert q_ref.shape == (3,) and split.shape == (3, 8)
    full = np.stack(np.broadcast_arrays(*(np.asarray(x) for x in theta)), axis=-1)
    assert_physical_round_trip(q_ref, split, full, float(np.float32(duration)))
    one_q, one = bg32.split_physical(bg.GBParams(*full[1]), duration)
    assert type(one_q) is int and one_q == q_ref[1]
    np.testing.assert_array_equal(one, split[1])


#: A float32-representable observation, so that the float64 functions can be
#: given the original float64 catalogue with the same duration and epoch.
DURATION32 = float(np.float32(4 * YEAR))
EPOCH32 = float(np.float32(10 * YEAR))
OBSERVATION = {"T": DURATION32, "t0": EPOCH32, "BW": 1024, "response_samples": 1024}


def test_split_native_waveforms_match_the_original_float64_catalogue():
    """End to end: float64 catalogue, split, float32 waveform."""
    rng = np.random.default_rng(21)
    native = native_catalogue(rng, DURATION32)[:3]
    native[:, 0] = np.array([0.1, 3.3, 29.0]) * 1e-3 * DURATION32 + rng.uniform(0, 1, 3)
    native[:, 7:] = [[2.0, -0.1], [-1.5, 0.05], [0.3, 0.0]]
    q_ref, params = bg32.split_native(native, DURATION32)
    for index in range(3):
        single = bg32.ucb_waveform(params[index], q_ref=q_ref[index], **OBSERVATION)
        carrier = float(q_ref[index] + np.floor(params[index, 0]))
        double = bg.ucb_waveform(native[index], q=carrier, **OBSERVATION)
        assert_accurate(channels(single), channels(double), f"source {index}")


def test_split_physical_waveforms_match_the_original_float64_catalogue():
    theta = np.array(
        [
            [1e-22, 1.0e-4 + 1.1e-9, 1e-19, 0.3, 0.4, 0.5, 0.6, 0.7],
            [3e-22, 3.3e-3 + 2.3e-9, -2e-18, 1.3, 2.4, 1.5, 4.6, -0.2],
            [2e-23, 2.9e-2 + 1.7e-9, 3e-17, 2.3, 1.4, 2.5, 5.6, -1.2],
        ]
    )
    q_ref, split = bg32.split_physical(theta, DURATION32)
    for index in range(3):
        single = bg32.tdi_waveform(split[index], q_ref=q_ref[index], **OBSERVATION)
        double = bg.tdi_waveform(theta[index], **OBSERVATION)
        assert int(single["qmin"]) == int(double["qmin"])
        assert_accurate(channels(single), channels(double), f"source {index}")
