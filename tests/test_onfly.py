"""The public default constructs sparse-time TDI before the Fourier transform."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica.glass.constants import FSTAR, PI2

DURATION = 7_884_000.0
SOURCE = np.array([7884.25, -0.21, 1.1, np.log(8e-22), 0.37, 0.63, 0.41, 0.03])


def _dense_sangria(params, duration, bandwidth, epoch=0.0, carrier=None):
    """Direct response on every FFT time, without sparse interpolation."""
    from binary_galactica.glass.onfly import tdi_response

    times = epoch + duration * np.arange(bandwidth) / bandwidth
    if carrier is None:
        carrier = np.trunc((params[0] / duration) * duration)
    u = times / duration
    cycles = (params[0] - carrier) * u
    if len(params) > 7:
        cycles = cycles + 0.5 * params[7] * u**2
    if len(params) > 8:
        cycles = cycles + params[8] * u**3 / 6
    phase = PI2 * cycles - params[6]
    response = np.asarray(tdi_response(jnp.asarray(params), times, duration))
    return np.fft.fftshift(np.fft.fft(response * np.exp(1j * phase)), axes=-1) * (
        np.sqrt(duration) / (2 * bandwidth)
    )


def test_default_uses_onfly_response_before_fft():
    bandwidth = 128
    waveform = bg.ucb_waveform(SOURCE, T=DURATION, BW=bandwidth)
    expected = _dense_sangria(SOURCE, DURATION, bandwidth)
    for index, channel in enumerate(("X", "Y", "Z")):
        residual = np.linalg.norm(waveform[channel] - expected[index])
        assert residual / np.linalg.norm(expected[index]) < 2e-6
    legacy = bg.ucb_waveform(SOURCE, T=DURATION, BW=bandwidth, backend="fourier")
    assert (
        np.linalg.norm(legacy["X"] - waveform["X"]) / np.linalg.norm(waveform["X"])
        > 1e-5
    )


def test_orbital_phase_removal_allows_sparse_high_frequency_response():
    """A 30 mHz source over a quarter year needs only 32 response samples.

    Retaining the annual Doppler phase would make this sparse grid inaccurate.
    """
    duration, samples = 0.25 * 31_557_600.0, 32
    source = np.array(
        [0.03 * duration + 0.25, -0.17, 3.7, np.log(1e-22), 0.91, 1.5, 1.1, 0.05, 0.2]
    )
    expected = _dense_sangria(source, duration, 1024, epoch=12_345.0)
    actual = bg.ucb_waveform(
        source, T=duration, BW=1024, t0=12_345.0, response_samples=samples
    )
    for index, channel in enumerate(("X", "Y", "Z")):
        relative_error = np.linalg.norm(
            actual[channel] - expected[index]
        ) / np.linalg.norm(expected[index])
        assert relative_error < 2e-6


def test_automatic_response_grid_tracks_traced_observation_duration():
    """Short and long observations resolve the response with one compiled API."""
    generate = jax.jit(
        lambda p, duration: bg.ucb_waveform(
            p, T=duration, BW=1024, t0=12_345.0, response_samples=None
        )
    )
    for years in (0.25, 1.0, 4.0):
        duration = years * 31_557_600.0
        source = np.array(
            [0.1 * duration + 0.25, 0.38, 3.7, np.log(1e-22), -0.8, 1.5, 1.1, 0.05]
        )
        expected = _dense_sangria(source, duration, 1024, epoch=12_345.0)
        actual = generate(source, duration)
        for index, channel in enumerate(("X", "Y", "Z")):
            relative_error = np.linalg.norm(
                actual[channel] - expected[index]
            ) / np.linalg.norm(expected[index])
            assert relative_error < 2e-6


def test_onfly_formats_preserve_existing_unit_relationships():
    waveforms = {
        fmt: bg.ucb_waveform(SOURCE, T=DURATION, BW=64, fmt=fmt) for fmt in bg.FORMATS
    }
    sangria = waveforms["sangria"]
    ratio = 2j * sangria["freq_bin_indices"] / DURATION / FSTAR
    for channel in ("X", "Y", "Z", "A", "E"):
        np.testing.assert_allclose(
            waveforms["frequency"][channel], -1j * sangria[channel], rtol=1e-11, atol=0
        )
        phase = waveforms["phase"][channel]
        if channel in ("A", "E"):
            phase = jnp.conj(phase)
        np.testing.assert_allclose(phase * ratio, sangria[channel], rtol=1e-11, atol=0)


@pytest.mark.parametrize("backend", ["onfly", "fourier"])
def test_physical_and_window_adapters_select_the_requested_backend(backend):
    physical = bg.from_glass_vector(SOURCE, DURATION)
    native = bg.ucb_waveform(
        SOURCE, T=DURATION, BW=64, backend=backend, response_samples=128
    )
    result = bg.tdi_waveform(
        physical, T=DURATION, BW=64, backend=backend, response_samples=128
    )
    for channel in ("X", "Y", "Z", "A", "E"):
        error = np.linalg.norm(result[channel] - native[channel])
        assert error / np.linalg.norm(native[channel]) < 1e-10
    window = bg.template_on_window(
        SOURCE,
        T=DURATION,
        BW=64,
        bin_lo=int(native["qmin"]),
        n_bins=64,
        backend=backend,
        response_samples=128,
    )
    np.testing.assert_allclose(
        window,
        np.stack([native["A"], native["E"]]) * np.sqrt(DURATION),
        rtol=1e-11,
        atol=0,
    )


# A bool, a non-integer and an integer just below the minimum of four.
@pytest.mark.parametrize("samples", [True, 8.5, 3])
def test_bad_response_resolution_is_rejected(samples):
    with pytest.raises(ValueError, match="response_samples"):
        bg.ucb_waveform(SOURCE, T=DURATION, BW=64, response_samples=samples)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        bg.ucb_waveform(SOURCE, T=DURATION, BW=64, backend="unknown")


def test_phase_dc_is_finite_and_zero():
    source = SOURCE.copy()
    source[0] = 2.25
    waveform = bg.ucb_waveform(source, T=DURATION, BW=16, fmt="phase")
    dc = np.asarray(waveform["freq_bin_indices"]) == 0
    assert dc.sum() == 1
    for channel in ("X", "Y", "Z", "A", "E"):
        assert np.all(np.isfinite(waveform[channel]))
        assert np.asarray(waveform[channel])[dc].item() == 0


def test_arbitrary_time_response_is_public_and_differentiable():
    times = jnp.array([0.0, 17.0, 12345.0, 981234.0])
    response = bg.tdi_response(SOURCE, times, T=DURATION)
    assert response.shape == (3, 4)
    assert response.dtype == jnp.complex128
    derivative = jax.jacfwd(lambda p: bg.tdi_response(p, times, T=DURATION).real)(
        jnp.asarray(SOURCE)
    )
    assert np.all(np.isfinite(derivative))
    # The source's constant initial phase is removed in this response.
    np.testing.assert_allclose(derivative[..., 6], 0.0, atol=1e-35)


def test_nonzero_epoch_and_carrier_override_match_dense_response():
    """Both frequency derivatives, a nonzero epoch and an explicit carrier."""
    source = np.append(SOURCE, 0.5)
    epoch, carrier = 5321234.0, 7880
    expected = _dense_sangria(source, DURATION, 64, epoch, carrier)
    actual = bg.ucb_waveform(source, T=DURATION, BW=64, t0=epoch, q=carrier)
    assert float(actual["qmin"]) == carrier - 32
    for index, channel in enumerate(("X", "Y", "Z")):
        assert (
            np.linalg.norm(actual[channel] - expected[index])
            / np.linalg.norm(expected[index])
            < 2e-6
        )


def test_cached_geometry_converges_to_direct_analytic_orbits():
    from binary_galactica.glass.waveform import _cached_positions

    times = jnp.linspace(12345.0, 0.9 * DURATION, 81)
    expected = bg.analytic_orbits(times)
    errors = []
    for count in (8, 32, 128):
        positions = bg.orbit_cache(DURATION, 12345.0, count)
        actual = _cached_positions(positions, times, DURATION, 12345.0)
        errors.append(float(jnp.linalg.norm(actual - expected)))
    assert errors[1] < errors[0] / 100
    assert errors[2] < errors[1] / 100


def test_arbitrary_time_positions_and_source_phase_conventions():
    times = np.array([-12345.0, 0.0, 15700.0, 0.9 * DURATION])
    response = bg.tdi_response(SOURCE, times, T=DURATION)
    cached = bg.tdi_response(
        SOURCE, times, T=DURATION, positions=bg.analytic_orbits(times)
    )
    assert np.linalg.norm(response - cached) / np.linalg.norm(response) < 1e-12
    from binary_galactica.glass.onfly import tdi_response as raw_response

    raw = raw_response(SOURCE, times, DURATION)
    np.testing.assert_allclose(response, raw, rtol=1e-12, atol=0)


@pytest.mark.parametrize("times", [0.0, [], [np.inf]], ids=["scalar", "empty", "inf"])
def test_arbitrary_time_response_rejects_invalid_times(times):
    with pytest.raises(ValueError, match="times"):
        bg.tdi_response(SOURCE, times, T=DURATION)


def test_arbitrary_time_traced_invalid_sources_and_times_return_nan():
    evaluate = jax.jit(lambda p, t: bg.tdi_response(p, t, T=DURATION))
    invalid = SOURCE.copy()
    invalid[1] = 1.5
    assert np.all(np.isnan(evaluate(invalid, np.array([0.0, 1.0]))))
    assert np.all(np.isnan(evaluate(SOURCE, np.array([0.0, np.inf]))))
