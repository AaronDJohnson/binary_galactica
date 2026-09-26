"""Carrier bins retain GLASS's float64 round trip inside outer JIT calls."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg


@pytest.mark.parametrize("duration", [7_884_000.0, 31_536_000.0, 126_144_000.0])
def test_constant_duration_jit_preserves_round_tripped_carriers(duration, glass_params):
    """Constant folding must not replace division with reciprocal multiplication."""
    integers = np.array([61.0, 122.0, 127.0, 1952.0, 7884.0, 15616.0, 31536.0])
    frequencies = np.concatenate(
        [np.nextafter(integers, 0), integers, np.nextafter(integers, np.inf)]
    )
    sources = np.tile(glass_params, (len(frequencies), 1))
    sources[:, 0] = frequencies
    # NumPy executes the separate IEEE division and multiplication used in C.
    expected_q = np.trunc((frequencies / duration) * duration)
    if duration == 7_884_000.0:
        assert expected_q[len(integers)] == 60.0  # params[0] is exactly 61 here.

    def channels(params):
        return bg.ucb_waveform(params, T=duration, BW=32)

    actual = jax.jit(jax.vmap(channels))(sources)
    np.testing.assert_array_equal(actual["qmin"], expected_q - 16)
    np.testing.assert_array_equal(
        actual["freq_bin_indices"], expected_q[:, None] - 16 + np.arange(32)
    )


@pytest.mark.parametrize(
    "n_params,fmt", [(7, "phase"), (8, "sangria"), (9, "frequency")]
)
def test_constant_duration_jit_matches_direct_and_dynamic_spectra(
    n_params, fmt, glass_params
):
    """The fix must preserve every channel, not only repair bin labels."""
    duration = 7_884_000.0
    params = np.pad(glass_params, (0, 1), constant_values=0.04)[:n_params]
    params[0] = 15616.0
    direct = bg.ucb_waveform(params, T=duration, BW=32, fmt=fmt)
    constant = jax.jit(lambda p: bg.ucb_waveform(p, T=duration, BW=32, fmt=fmt))(params)
    dynamic = jax.jit(lambda p, t: bg.ucb_waveform(p, T=t, BW=32, fmt=fmt))(
        params, duration
    )
    # Also cover an outer function whose source and duration are both constants.
    closed = jax.jit(lambda: bg.ucb_waveform(params, T=duration, BW=32, fmt=fmt))()
    assert float(direct["qmin"]) == 15599.0
    for result in (constant, dynamic, closed):
        np.testing.assert_array_equal(
            result["freq_bin_indices"], direct["freq_bin_indices"]
        )
        for channel in ("X", "Y", "Z", "A", "E"):
            expected = np.asarray(direct[channel])
            residual = np.linalg.norm(np.asarray(result[channel]) - expected)
            assert residual / np.linalg.norm(expected) < 1e-9


def test_constant_duration_carrier_composes_with_batched_derivatives(glass_params):
    """Protect both transform orders when carrier inputs have zero derivatives."""
    params = np.tile(glass_params, (2, 1))
    params[:, 0] = [61.25, 122.25]
    sources = jnp.asarray(params)

    def channels(p):
        waveform = bg.ucb_waveform(p, T=7_884_000.0, BW=32)
        return jnp.concatenate([waveform["A"].real, waveform["E"].imag]) / 1e-22

    forward = jax.jit(jax.vmap(jax.jacfwd(channels)))(sources)
    reverse = jax.jit(jax.vmap(jax.jacrev(channels)))(sources)
    assert np.all(np.isfinite(forward))
    np.testing.assert_allclose(reverse, forward, rtol=1e-9, atol=1e-14)

    def powers(batch):
        return jnp.sum(jax.vmap(channels)(batch) ** 2)

    gradient = jax.jit(jax.grad(powers))(sources)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.linalg.norm(gradient, axis=1) > 0)


def test_carrier_handles_nested_source_and_duration_batches(glass_params):
    """Independent source and duration batch axes must not be conflated."""
    sources = np.tile(glass_params, (3, 1))
    sources[:, 0] = [61.0, 122.0, 127.0]
    durations = np.array([7_884_000.0, 31_536_000.0])

    def carrier(p, duration):
        return bg.ucb_waveform(p, T=duration, BW=32)["qmin"] + 16

    evaluate = jax.jit(
        jax.vmap(jax.vmap(carrier, in_axes=(0, None)), in_axes=(None, 0))
    )
    expected = np.trunc((sources[:, 0] / durations[:, None]) * durations[:, None])
    np.testing.assert_array_equal(evaluate(sources, durations), expected)


def test_carrier_barrier_preserves_observation_duration_derivatives(glass_params):
    """Only the discrete carrier is constant when differentiating duration."""
    params = glass_params.copy()
    params[0] = 7884.25
    duration = 7_884_000.0

    def channels(t):
        waveform = bg.ucb_waveform(params, T=t, BW=32)
        return jnp.concatenate([waveform["A"].real, waveform["E"].imag]) / 1e-22

    actual = np.asarray(jax.jit(jax.jacfwd(channels))(duration))
    step = 10.0
    expected = (np.asarray(channels(duration + step)) - channels(duration - step)) / (
        2 * step
    )
    assert np.all(np.isfinite(actual))
    assert np.linalg.norm(expected) > 0
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-5


def test_constant_duration_jit_preserves_physical_and_window_adapters(glass_params):
    """Adapters must inherit the carrier fix without shifting their spectra."""
    duration = 7_884_000.0
    native = glass_params.copy()
    native[0] = 15616.0
    physical = np.array(
        [
            np.exp(native[3]) / 2,
            native[0] / duration,
            native[7] / duration**2,
            -native[6],
            np.arccos(native[4]),
            native[5],
            native[2],
            np.arcsin(native[1]),
        ]
    )
    direct = bg.tdi_waveform(physical, T=duration, BW=32)
    constant = jax.jit(lambda p: bg.tdi_waveform(p, T=duration, BW=32))(physical)
    dynamic = jax.jit(lambda p, t: bg.tdi_waveform(p, T=t, BW=32))(physical, duration)
    expected_q = np.trunc(((physical[1] * duration) / duration) * duration)
    assert float(direct["qmin"]) == expected_q - 16
    for result in (constant, dynamic):
        np.testing.assert_array_equal(
            result["freq_bin_indices"], direct["freq_bin_indices"]
        )
        for channel in ("X", "Y", "Z", "A", "E"):
            scale = np.linalg.norm(direct[channel])
            assert np.linalg.norm(result[channel] - direct[channel]) / scale < 1e-9

    def window(p, t):
        return bg.template_on_window(p, T=t, BW=32, bin_lo=15599, n_bins=32)

    expected = window(native, duration)
    constant_window = jax.jit(lambda p: window(p, duration))(native)
    dynamic_window = jax.jit(window)(native, duration)
    for result in (constant_window, dynamic_window):
        assert np.linalg.norm(result - expected) / np.linalg.norm(expected) < 1e-9
