"""The window functions form A and E before the transform, in both precisions.

A and E are fixed linear combinations of X, Y and Z, and every later stage of
the on-fly path -- interpolation onto the FFT grid, the common phase factor,
the FFT and the per-bin format factors -- is linear and the same for every
channel. ``template_on_window`` therefore combines the three sparse response
rows into A and E first and interpolates, rotates and transforms two channels
instead of three. Its windows must equal the placement of ``ucb_waveform``'s A
and E, which still transforms X, Y and Z, to rounding, in every format, with
and without a position cache and for 7, 8 or 9 native entries, and its NaN
masks for invalid traced inputs must be the placement's.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import single as bg32
from binary_galactica.glass import single as single_impl
from binary_galactica.glass import waveform as glass_waveform
from tests.conftest import nested_jaxprs
from tests.single.support import Source, source

BW = 32
SAMPLES = 16
DURATION = 7_884_000.0
EPOCH = 12_345.0
#: A generic source in float64 native coordinates, nine entries long.
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
#: A quarter year at 1 mHz, ten years late, nine entries long.
CASE32 = dataclasses.replace(
    source(0.25, 1e-3, 10.0, (1, -1), bandwidth=BW), samples=SAMPLES
)
#: The rotation of the constellation in supplied caches, so that ignoring
#: them would show.
KAPPA = 0.3
#: Relative L2 bounds against the placed ucb_waveform A and E, as fractions
#: of the window's norm. The two paths differ only in the order of linear
#: operations, so the difference is rounding: at most 4.7e-16 and 2.8e-7 on
#: these cases with JAX 0.6.0, 0.9.0.1 and 0.11.2. Much coarser response
#: grids amplify interpolation rounding, to 1.1e-14 and 5.4e-6 with eight
#: samples over one to four years. A wrong sign, channel or format factor
#: moves a window by order one.
FLOAT64_BOUND = 1e-14
FLOAT32_BOUND = 2e-6


def cache(duration: float, epoch: float) -> np.ndarray:
    """Return float64 positions of the rotated constellation on the FFT grid."""
    grid = epoch + duration * np.arange(BW) / BW
    return np.asarray(bg.analytic_orbits(grid, kappa_0=KAPPA))


def window_of(bins: np.ndarray, channels: np.ndarray, bin_lo: int, n_bins: int):
    """Place ``(2, BW)`` channels at absolute ``bins`` on a window, on the host."""
    bins = np.asarray(bins).astype(np.int64)
    inside = (bins >= bin_lo) & (bins < bin_lo + n_bins)
    placed = np.zeros((2, n_bins), dtype=channels.dtype)
    placed[:, bins[inside] - bin_lo] = channels[:, inside]
    return placed


def relative_error(actual, expected) -> float:
    """Return the relative L2 difference, formed in float64."""
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def float64_window(params, *, fmt, positions):
    """Return the float64 window and ucb_waveform's A and E placed on it."""
    options = {
        "T": DURATION,
        "t0": EPOCH,
        "BW": BW,
        "fmt": fmt,
        "positions": positions,
        "response_samples": SAMPLES,
    }
    waveform = bg.to_continuous_ft(bg.ucb_waveform(params, **options), DURATION)
    # Crop the band's three lowest bins and run eleven bins past its top.
    bin_lo, n_bins = int(waveform["qmin"]) + 3, BW + 8
    actual = bg.template_on_window(params, bin_lo=bin_lo, n_bins=n_bins, **options)
    channels = np.stack([np.asarray(waveform["A"]), np.asarray(waveform["E"])])
    expected = window_of(waveform["freq_bin_indices"], channels, bin_lo, n_bins)
    return np.asarray(actual), expected


def float32_window(case: Source, *, fmt, positions):
    """Return the float32 window and float32 ucb_waveform's A and E placed on it."""
    options = {**case.observation(), "fmt": fmt, "positions": positions}
    waveform = bg32.ucb_waveform(case.params, q_ref=case.q_ref, **options)
    bin_lo, n_bins = int(waveform["qmin"]) + 3, BW + 8
    actual = bg32.template_on_window(
        case.params, q_ref=case.q_ref, bin_lo=bin_lo, n_bins=n_bins, **options
    )
    # The window scales by sqrt(T) in float32, as here.
    scale = np.sqrt(np.float32(case.duration))
    channels = scale * np.stack([np.asarray(waveform["A"]), np.asarray(waveform["E"])])
    expected = window_of(waveform["freq_bin_indices"], channels, bin_lo, n_bins)
    return np.asarray(actual), expected


@pytest.mark.parametrize("length", [7, 8, 9])
@pytest.mark.parametrize("cached", [False, True], ids=["analytic", "cache"])
@pytest.mark.parametrize("fmt", bg.FORMATS)
def test_float64_window_places_ucb_waveform_channels(fmt, cached, length):
    params = NATIVE[:length]
    positions = cache(DURATION, EPOCH) if cached else None
    actual, expected = float64_window(params, fmt=fmt, positions=positions)
    assert actual.dtype == np.complex128
    assert relative_error(actual, expected) <= FLOAT64_BOUND
    # Bins the band does not reach are exactly zero in both.
    np.testing.assert_array_equal(actual[expected == 0], 0)
    if cached:
        analytic, _ = float64_window(params, fmt=fmt, positions=None)
        assert relative_error(analytic, actual) > 1e-2


@pytest.mark.parametrize("length", [7, 8, 9])
@pytest.mark.parametrize("cached", [False, True], ids=["analytic", "cache"])
@pytest.mark.parametrize("fmt", bg.FORMATS)
def test_float32_window_places_ucb_waveform_channels(fmt, cached, length):
    case = dataclasses.replace(CASE32, params=CASE32.params[:length])
    positions = None
    if cached:
        positions = cache(float(case.duration), float(case.epoch)).astype(np.float32)
    actual, expected = float32_window(case, fmt=fmt, positions=positions)
    assert actual.dtype == np.complex64
    assert relative_error(actual, expected) <= FLOAT32_BOUND
    np.testing.assert_array_equal(actual[expected == 0], 0)
    if cached:
        analytic, _ = float32_window(case, fmt=fmt, positions=None)
        assert relative_error(analytic, actual) > 1e-2


def channel_counts(jaxpr):
    """Return the channel counts entering interpolation and the FFT.

    Interpolation stacks the real and imaginary parts of every channel as the
    rows of one convolution, and the FFT transforms ``(channels, BW)``.
    """
    interpolated, transformed = set(), set()
    for inner in nested_jaxprs(jaxpr):
        for equation in inner.eqns:
            if equation.primitive.name == "conv_general_dilated":
                interpolated.add(equation.invars[0].aval.shape[0] // 2)
            elif equation.primitive.name == "fft":
                transformed.add(equation.invars[0].aval.shape[0])
    return interpolated, transformed


WINDOW = {"bin_lo": 7_870, "n_bins": BW}
#: Each on-fly entry, with the number of channels it should transform.
ENTRIES = {
    "template_on_window": (
        lambda: bg.template_on_window(
            NATIVE, T=DURATION, BW=BW, response_samples=SAMPLES, **WINDOW
        ),
        2,
    ),
    "single.template_on_window": (
        lambda: bg32.template_on_window(
            CASE32.params,
            q_ref=CASE32.q_ref,
            bin_lo=CASE32.carrier - BW // 2,
            n_bins=BW,
            **CASE32.observation(),
        ),
        2,
    ),
    "ucb_waveform": (
        lambda: bg.ucb_waveform(NATIVE, T=DURATION, BW=BW, response_samples=SAMPLES),
        3,
    ),
    "single.ucb_waveform": (
        lambda: bg32.ucb_waveform(
            CASE32.params, q_ref=CASE32.q_ref, **CASE32.observation()
        ),
        3,
    ),
    "tdi_waveform": (
        lambda: bg.tdi_waveform(
            bg.from_glass_vector(NATIVE[:8], DURATION),
            T=DURATION,
            BW=BW,
            response_samples=SAMPLES,
        ),
        3,
    ),
}


@pytest.mark.parametrize("name", list(ENTRIES))
def test_windows_transform_only_a_and_e(name):
    """Windows interpolate and transform two channels; full spectra keep three."""
    entry, channels = ENTRIES[name]
    interpolated, transformed = channel_counts(jax.make_jaxpr(entry)())
    assert interpolated == {channels}, name
    assert transformed == {channels}, name


def test_full_spectra_keep_five_channels():
    keys = {"qmin", "freq_bin_indices", "X", "Y", "Z", "A", "E"}
    for name in ("ucb_waveform", "single.ucb_waveform", "tdi_waveform"):
        assert set(ENTRIES[name][0]()) == keys, name


def place64(waveform, duration, bin_lo, n_bins):
    """Place float64 A and E on a window as the shim does, inside a trace."""
    scale = jnp.sqrt(duration)
    index = jnp.asarray(waveform["freq_bin_indices"] - bin_lo, dtype=jnp.int32)
    inside = (index >= 0) & (index < n_bins)
    safe = jnp.where(inside, index, n_bins)
    zero = jnp.zeros((), dtype=jnp.complex128)
    out = jnp.zeros((2, n_bins), dtype=jnp.complex128)
    for row, channel in enumerate("AE"):
        value = jnp.where(inside, scale * waveform[channel], zero)
        out = out.at[row, safe].add(value, mode="drop")
    return out


def place32(result, valid, duration, bin_lo, n_bins):
    """Place float32 A and E as the float32 window does, NaN if invalid."""
    index = result["freq_bin_indices"] - np.int32(bin_lo)
    inside = (index >= 0) & (index < n_bins)
    safe = jnp.where(inside, index, n_bins)
    zero = jnp.zeros((), dtype=jnp.complex64)
    out = jnp.zeros((2, n_bins), dtype=jnp.complex64)
    for row, channel in enumerate("AE"):
        value = jnp.where(inside, jnp.sqrt(duration) * result[channel], zero)
        out = out.at[row, safe].add(value, mode="drop")
    return jnp.where(valid, out, jnp.nan)


def corrupt(params, positions, changes):
    """Return copies of the inputs with entries replaced by invalid values."""
    params, positions = params.copy(), positions.copy()
    for index, value in changes.get("params", {}).items():
        params[index] = value
    if changes.get("positions"):
        positions[1, 5, 2] = np.nan
    return params, positions


#: Invalid traced inputs, as replacements of valid ones.
INVALID64 = {
    "valid": {},
    "inclination": {"params": {4: 1.5}},
    "sky": {"params": {1: -1.25}},
    "frequency": {"params": {0: -5.0}},
    "non-finite amplitude": {"params": {3: np.nan}},
    "duration": {"T": -DURATION},
    "infinite duration": {"T": np.inf},
    "epoch": {"t0": np.nan},
    "cache": {"positions": True},
}


@pytest.mark.parametrize("fmt", ["sangria", "phase"])
def test_float64_nan_masks_match_placed_waveform(fmt):
    """Under jit with every input traced, invalid sources mask identically.

    An invalid source has NaN bins as well as NaN channels, so where the
    window holds NaN depends on how the compiler converts NaN to an index.
    The A/E-only spectra the window places are therefore also compared with
    ucb_waveform's A and E directly: NaN in every bin exactly when invalid.
    """
    bin_lo, n_bins = 7_870, BW + 8
    observation = {"BW": BW, "fmt": fmt, "response_samples": SAMPLES}

    @jax.jit
    def compiled(params, duration, t0, positions):
        window = bg.template_on_window(
            params,
            T=duration,
            t0=t0,
            positions=positions,
            bin_lo=bin_lo,
            n_bins=n_bins,
            **observation,
        )
        waveform = bg.ucb_waveform(
            params, T=duration, t0=t0, positions=positions, **observation
        )
        spectra = glass_waveform._ucb_waveform(
            params,
            T=duration,
            t0=t0,
            positions=positions,
            channels="ae",
            **observation,
        )
        return (
            window,
            place64(waveform, duration, bin_lo, n_bins),
            jnp.stack([spectra["A"], spectra["E"]]),
            jnp.stack([waveform["A"], waveform["E"]]),
        )

    for label, changes in INVALID64.items():
        params, positions = corrupt(NATIVE, cache(DURATION, EPOCH), changes)
        outputs = compiled(
            params, changes.get("T", DURATION), changes.get("t0", EPOCH), positions
        )
        window, expected, spectra, channels = (np.asarray(x) for x in outputs)
        np.testing.assert_array_equal(np.isnan(window), np.isnan(expected), label)
        np.testing.assert_array_equal(np.isnan(spectra), np.isnan(channels), label)
        if label == "valid":
            assert not np.isnan(window).any()
            assert relative_error(window, expected) <= FLOAT64_BOUND
        else:
            assert np.isnan(spectra).all(), label


INVALID32 = {
    "valid": {},
    "inclination": {"params": {4: np.float32(1.5)}},
    "non-finite longitude": {"params": {2: np.float32(np.nan)}},
    "negative frequency": {"q_ref": np.int32(-100)},
    "duration": {"T": -CASE32.duration},
    "epoch": {"t0": np.float32(np.nan)},
    "cache": {"positions": True},
}


@pytest.mark.parametrize("fmt", ["sangria", "phase"])
def test_float32_nan_masks_match_placed_waveform(fmt):
    """The float32 window is NaN throughout exactly when the waveform is invalid.

    The public float32 waveform masks invalid bins with ``INT32_MIN``, which a
    placement cannot reproduce, so the reference places the unmasked
    five-channel waveform with its validity flag, as 0.6.0's window did.
    """
    case = CASE32
    bin_lo, n_bins = case.carrier - BW // 2 + 3, BW + 8
    disabled = single_impl._DISABLED

    @jax.jit
    def compiled(params, q_ref, duration, t0, positions):
        window = bg32.template_on_window(
            params,
            q_ref=q_ref,
            T=duration,
            t0=t0,
            BW=BW,
            fmt=fmt,
            positions=positions,
            response_samples=SAMPLES,
            bin_lo=bin_lo,
            n_bins=n_bins,
        )
        result, valid = single_impl._waveform(
            params, q_ref, duration, BW, t0, fmt, positions, None, SAMPLES, disabled
        )
        return window, place32(result, valid, duration, bin_lo, n_bins)

    positions = cache(float(case.duration), float(case.epoch)).astype(np.float32)
    for label, changes in INVALID32.items():
        params, cached = corrupt(case.params, positions, changes)
        window, expected = compiled(
            params,
            changes.get("q_ref", np.int32(case.q_ref)),
            changes.get("T", case.duration),
            changes.get("t0", case.epoch),
            cached,
        )
        window, expected = np.asarray(window), np.asarray(expected)
        np.testing.assert_array_equal(np.isnan(window), np.isnan(expected), label)
        if label == "valid":
            assert not np.isnan(window).any()
            assert relative_error(window, expected) <= FLOAT32_BOUND
        else:
            assert np.isnan(window).all(), label
