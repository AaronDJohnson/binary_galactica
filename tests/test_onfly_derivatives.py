"""Derivative accuracy of sparse TDI and its restored reference phase."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica.glass.constants import YEAR
from binary_galactica.glass.onfly import tdi_response as raw_response


def _quadratures(a, e):
    """Keep both complex quadratures measurable at physical strain scales."""
    channels = jnp.concatenate((a, e)) / 1e-19
    return jnp.concatenate((channels.real, channels.imag))


def test_source_jacobian_matches_dense_response_without_sparse_phase_reference():
    duration, epoch, bandwidth = 4 * YEAR, 0.7 * YEAR, 128
    params = jnp.array(
        [0.03 * duration + 0.25, 0.2, 2.3, np.log(1e-22), 0.0, 0.8, 0.4, -0.2, 2.0]
    )
    carrier = np.trunc((float(params[0]) / duration) * duration)

    def sparse_vector(p):
        waveform = bg.ucb_waveform(
            p, T=duration, t0=epoch, BW=bandwidth, response_samples=256
        )
        return _quadratures(waveform["A"], waveform["E"])

    def dense_vector(p):
        # This response removes phi(t), independently of the sparse path's
        # spacecraft phase reference and interpolation. Restore the native
        # source phase minus a fixed carrier directly on every FFT sample.
        times = epoch + duration * jnp.arange(bandwidth) / bandwidth
        u = times / duration
        phase = (
            2 * jnp.pi * ((p[0] - carrier) * u + p[7] * u**2 / 2 + p[8] * u**3 / 6)
            - p[6]
        )
        response = raw_response(p, times, duration)
        xyz = jnp.fft.fftshift(
            jnp.fft.fft(response * jnp.exp(1j * phase), axis=-1), axes=-1
        ) * (jnp.sqrt(duration) / (2 * bandwidth))
        a, e = bg.xyz_to_ae(*xyz)
        return _quadratures(a, e)

    actual = np.asarray(jax.jit(jax.jacfwd(sparse_vector))(params))
    expected = np.asarray(jax.jit(jax.jacfwd(dense_vector))(params))
    assert np.all(np.isfinite(actual))
    column_norm = np.linalg.norm(expected, axis=0)
    assert np.all(column_norm > 1e-6), column_norm
    relative_error = np.linalg.norm(actual - expected, axis=0) / column_norm
    # A deliberately coarse 256-point response over four years has measured
    # interpolation derivative errors of order 1e-7. Bound every coordinate,
    # including the second chirp and edge-on inclination, independently.
    assert np.all(relative_error < 3e-6), relative_error


@pytest.mark.parametrize("cached", [False, True], ids=["analytic", "regenerated-cache"])
def test_fixed_grid_duration_epoch_derivatives_across_auto_grid_threshold(cached):
    duration, bandwidth = YEAR / 4, 128
    params = jnp.array(
        [0.03 * duration + 0.25, 0.2, 2.3, np.log(1e-22), 0.0, 0.8, 0.4, -0.2, 2.0]
    )

    def vector(observation):
        elapsed = duration * jnp.exp(observation[0])
        epoch = YEAR * observation[1]
        # Regeneration must remain inside AD: orbit motion contributes to
        # duration/epoch derivatives for both cache and analytic paths.
        positions = bg.orbit_cache(elapsed, epoch, bandwidth) if cached else None
        waveform = bg.ucb_waveform(
            params,
            T=elapsed,
            t0=epoch,
            BW=bandwidth,
            positions=positions,
            response_samples=64,
        )
        return _quadratures(waveform["A"], waveform["E"])

    # The central differences cross the automatic quarter-year threshold.
    # Fixing the response count isolates the smooth physical derivatives.
    observation = jnp.array([0.0, 0.7])
    forward = np.asarray(jax.jit(jax.jacfwd(vector))(observation))
    reverse = np.asarray(jax.jit(jax.jacrev(vector))(observation))
    step = 2e-7
    finite_difference = np.stack(
        [
            np.asarray(vector(observation + step * direction))
            - np.asarray(vector(observation - step * direction))
            for direction in jnp.eye(2)
        ],
        axis=-1,
    ) / (2 * step)
    assert np.all(np.isfinite(forward))
    assert np.all(np.isfinite(reverse))
    column_norm = np.linalg.norm(finite_difference, axis=0)
    assert np.all(column_norm > 1e-6), column_norm
    for actual in (forward, reverse):
        relative_error = (
            np.linalg.norm(actual - finite_difference, axis=0) / column_norm
        )
        assert np.all(relative_error < 2e-6), relative_error

    def power(observation):
        return jnp.sum(vector(observation) ** 2)

    hessian = np.asarray(jax.jit(jax.hessian(power))(observation))
    assert np.all(np.isfinite(hessian))
