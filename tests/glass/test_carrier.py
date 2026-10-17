"""Carrier bins retain GLASS's float64 round trip inside outer JIT calls."""

import warnings

import jax
import numpy as np

import binary_galactica as bg


def test_constant_duration_jit_preserves_round_tripped_carriers(glass_params):
    """Constant folding must not replace division with reciprocal multiplication.

    A quarter year suffices: every duration tried (also one and four years)
    has five round-trip victims among these frequencies.
    """
    duration = 7_884_000.0
    integers = np.array([61.0, 122.0, 127.0, 1952.0, 7884.0, 15616.0, 31536.0])
    frequencies = np.concatenate(
        [np.nextafter(integers, 0), integers, np.nextafter(integers, np.inf)]
    )
    sources = np.tile(glass_params, (len(frequencies), 1))
    sources[:, 0] = frequencies
    # NumPy executes the separate IEEE division and multiplication used in C.
    expected_q = np.trunc((frequencies / duration) * duration)
    assert expected_q[len(integers)] == 60.0  # params[0] is exactly 61 here.

    def channels(params):
        return bg.ucb_waveform(params, T=duration, BW=32)

    actual = jax.jit(jax.vmap(channels))(sources)
    np.testing.assert_array_equal(actual["qmin"], expected_q - 16)
    np.testing.assert_array_equal(
        actual["freq_bin_indices"], expected_q[:, None] - 16 + np.arange(32)
    )


def test_constant_duration_jit_matches_direct_and_dynamic_spectra(glass_params):
    """The fix must preserve every channel, not only repair bin labels.

    The carrier rule does not depend on the format or parameter count, so one
    native branch (eight parameters, Sangria) covers it.
    """
    duration, fmt = 7_884_000.0, "sangria"
    params = glass_params.copy()
    params[0] = 15616.0
    direct = bg.ucb_waveform(params, T=duration, BW=32, fmt=fmt)
    constant = jax.jit(lambda p: bg.ucb_waveform(p, T=duration, BW=32, fmt=fmt))(params)
    # Every automatic count is at least BW=32 and evaluates the response
    # directly, so a traced duration selects no grid and has nothing to warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", bg.ResponseSamplesWarning)
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
    # Both durations' automatic counts are at least BW=32, so they share the
    # direct evaluation and the traced durations compile no switch.
    with warnings.catch_warnings():
        warnings.simplefilter("error", bg.ResponseSamplesWarning)
        actual = evaluate(sources, durations)
    np.testing.assert_array_equal(actual, expected)
