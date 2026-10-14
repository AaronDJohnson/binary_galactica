"""Public contracts for the native GLASS waveform backend."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica.glass as glass


def test_native_waveform_contract(glass_params, injection):
    """The native API returns the five TDI channels on one band.

    Every format and parameter count is value-checked against stored float64
    output by ``tests/test_float64_golden.py``.
    """
    T, BW = injection["T"], 32
    waveform = glass.ucb_waveform(glass_params, T=T, BW=BW, fmt="sangria")

    assert set(waveform) == {
        "qmin",
        "freq_bin_indices",
        "X",
        "Y",
        "Z",
        "A",
        "E",
    }
    assert np.array_equal(
        np.asarray(waveform["freq_bin_indices"]),
        np.asarray(waveform["qmin"]) + np.arange(BW),
    )
    for channel in ("X", "Y", "Z", "A", "E"):
        assert waveform[channel].shape == (BW,)
        assert waveform[channel].dtype == jnp.complex128
        assert np.all(np.isfinite(np.asarray(waveform[channel])))


def test_invalid_parameter_shapes_are_rejected():
    """One guard; an unvectorized batch is the realistic mistake."""
    with pytest.raises(ValueError, match="one-dimensional.*7, 8, or 9"):
        glass.ucb_waveform(np.zeros((2, 8)), T=1024.0, BW=32)


# The message names two requirements: positive (0) and even (31).
@pytest.mark.parametrize("bandwidth", [0, 31])
def test_invalid_bandwidths_are_rejected(glass_params, bandwidth):
    with pytest.raises(ValueError, match="positive even"):
        glass.ucb_waveform(glass_params, T=1024.0, BW=bandwidth)


def test_cached_positions_must_match_the_static_bandwidth(glass_params):
    with pytest.raises(ValueError, match=r"positions must have shape \(3, BW, 3\)"):
        glass.ucb_waveform(
            glass_params,
            T=1024.0,
            BW=32,
            positions=np.zeros((3, 16, 3)),
        )
