"""Public contracts for the native GLASS waveform backend."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica.glass as glass


@pytest.mark.parametrize("n_params", [7, 8, 9])
@pytest.mark.parametrize("fmt", glass.FORMATS)
def test_all_native_parameter_counts_and_formats(
    glass_params, injection, n_params, fmt
):
    """Every native GLASS branch returns the five TDI channels on one band."""
    T, BW = injection["T"], 32
    params = np.pad(glass_params, (0, 1), constant_values=1e-8)[:n_params]
    waveform = glass.ucb_waveform(params, T=T, BW=BW, fmt=fmt)

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


@pytest.mark.parametrize(
    "params",
    [
        np.zeros(6),
        np.zeros(10),
        np.zeros((2, 8)),
        np.array(1.0),
    ],
)
def test_invalid_parameter_shapes_are_rejected(params):
    with pytest.raises(ValueError, match="one-dimensional.*7, 8, or 9"):
        glass.ucb_waveform(params, T=1024.0, BW=32)


@pytest.mark.parametrize("bandwidth", [-2, 0, 31, 33])
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


def test_dynamic_observation_time_survives_jit(glass_params):
    """T stays traced while BW and format control compiled output structure."""

    @jax.jit
    def waveform_at(T):
        return glass.ucb_waveform(glass_params, T=T, BW=32)["A"]

    a = waveform_at(jnp.asarray(7_884_000.0))
    b = waveform_at(jnp.asarray(7_884_001.0))
    assert a.shape == b.shape == (32,)
    assert np.all(np.isfinite(np.asarray(a)))
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_q_override_pins_the_output_window(glass_params, injection):
    q = 12_345
    waveform = glass.ucb_waveform(glass_params, T=injection["T"], BW=32, q=q)
    assert int(waveform["qmin"]) == q - 16
