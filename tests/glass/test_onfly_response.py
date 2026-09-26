"""Independent GLASS delayed-response checks and removable-limit tests."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica.glass.onfly import tdi_response


@pytest.mark.parametrize("nparams", [7, 8, 9])
@pytest.mark.parametrize("cached", [False, True])
def test_reference_phase_reconstruction_and_derivatives(nparams, cached):
    from binary_galactica.glass.onfly import reference_phase_shift
    from binary_galactica.glass.orbit import analytic_orbits

    duration = jnp.array(31457280.0)
    params = jnp.array(
        [duration * 0.03, -0.3, 5.1, np.log(1e-22), -0.8, 1.2, 2.1, 1300.0, 90.0]
    )[:nparams]
    times = jnp.array([0.0, 12345.0, 2.7e7, 1.0e8])
    positions = analytic_orbits(times) if cached else None

    def original(p, t, observation):
        return tdi_response(p, t, observation, positions)

    def reconstructed(p, t, observation):
        reference = tdi_response(p, t, observation, positions, reference_phase=True)
        return reference * jnp.exp(
            1j * reference_phase_shift(p, t, observation, positions)
        )

    def compare(actual, expected):
        scale = np.max(np.abs(expected))
        assert np.isfinite(actual).all()
        if scale == 0:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(
                actual / scale, expected / scale, rtol=2e-11, atol=2e-11
            )

    compare(
        jax.jit(reconstructed)(params, times, duration),
        original(params, times, duration),
    )
    primals = (params, times, duration)
    tangents = (
        (jnp.linspace(0.01, 0.03, nparams), jnp.zeros_like(times), jnp.array(0.0)),
        (jnp.zeros_like(params), jnp.ones_like(times), jnp.array(0.0)),
        (jnp.zeros_like(params), jnp.zeros_like(times), jnp.array(1.0)),
    )
    for tangent in tangents:
        compare(
            jax.jvp(reconstructed, primals, tangent)[1],
            jax.jvp(original, primals, tangent)[1],
        )
    if nparams == 9 and not cached:
        compare(
            jax.hessian(lambda p: reconstructed(p, times, duration).real.sum())(params),
            jax.hessian(lambda p: original(p, times, duration).real.sum())(params),
        )


def test_independent_c_delayed_response():
    fixture = json.loads(
        (Path(__file__).parents[1] / "data/onfly/reference.json").read_text()
    )
    for case in fixture["cases"]:
        actual = tdi_response(
            jnp.array(case["params"]),
            jnp.array(case["times"]),
            case["T"],
            jnp.array(case["positions"]),
        )
        expected = np.array(case["real"]) + 1j * np.array(case["imag"])
        # Carrier subtraction in C loses precision, particularly when the
        # TDI combination cancels at low frequency. Scale the absolute bound
        # to the response so amplitudes of 1e-22 cannot pass as numerical zero.
        # At 0.1 mHz the original full-carrier C sum loses about 8e-8 of
        # the peak response; >=1 mHz cases agree within 6e-9.
        scale = np.max(np.abs(expected))
        frequency = case["params"][0] / case["T"]
        reference_roundoff = 1e-8 * max(0.001 / frequency, 1.0)
        np.testing.assert_allclose(
            actual / scale, expected / scale, rtol=2e-8, atol=reference_roundoff
        )


@pytest.mark.parametrize("count", [7, 8, 9])
def test_phase_independence_and_jax_composition(count):
    params = jnp.array([1200.0, 0.3, 2.1, -1.0, 0.4, 0.7, 0.9, 0.12, 0.03])[:count]
    times = jnp.array([0.0, 1234.0, 32103.0, 1.0e5])
    compiled = jax.jit(tdi_response)
    response = compiled(params, times, 1.0e6)
    np.testing.assert_array_equal(
        response, compiled(params.at[6].set(2.4), times, 1.0e6)
    )
    gradient = jax.jacfwd(lambda p: jnp.real(tdi_response(p, times, 1.0e6)))(params)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient[..., 3], response.real, rtol=1e-12, atol=1e-14)
    batched = jax.jit(jax.vmap(lambda p: tdi_response(p, times, 1.0e6)))(
        jnp.stack([params, params.at[6].set(2.4)])
    )
    np.testing.assert_allclose(
        batched, jnp.stack([response, response]), rtol=1e-12, atol=1e-14
    )


def test_aligned_arm_removable_limit_and_derivatives():
    from binary_galactica.glass.constants import LARM

    positions = jnp.array(
        [[0.0, 0.0, 0.0], [LARM, 0.0, 0.0], [0.5 * LARM, np.sqrt(3) * 0.5 * LARM, 0.0]]
    )[:, None, :]
    params = jnp.array([5000.0, 0.0, 0.0, 0.0, 0.4, 0.7, 0.9, 0.12, 0.03])

    def fn(p):
        return tdi_response(p, jnp.array([12000.0]), 1.0e6, positions)

    exact = fn(params)
    assert np.isfinite(exact).all()
    assert np.isfinite(jax.jacfwd(lambda p: fn(p).real)(params)).all()
    assert np.isfinite(jax.hessian(lambda p: fn(p).real.sum())(params)).all()
    np.testing.assert_allclose(fn(params.at[2].set(1e-7)), exact, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("frequency", [1e-4, 0.005, 0.03])
def test_equal_arm_monochromatic_normalization_against_reversed_links(frequency):
    from binary_galactica.glass.constants import CLIGHT, FSTAR, LARM
    from binary_galactica.glass.orbit import arm_vectors
    from binary_galactica.glass.response import (
        LINKS,
        detector_tensor,
        polarization_tensor,
    )
    from binary_galactica.glass.tdi import synthesize

    positions = jnp.array(
        [[0.0, 0.0, 0.0], [LARM, 0.0, 0.0], [0.5 * LARM, np.sqrt(3) * 0.5 * LARM, 0.0]]
    )[:, None, :]
    times, T = jnp.array([12345.0]), 1.0e6
    params = jnp.array([frequency * T, 0.3, 2.1, 0.0, 0.4, 0.7, 0.9])
    ep, ec, k = polarization_tensor(params[1], params[2])
    dp, dc, kr = detector_tensor(arm_vectors(positions), k, ep, ec)
    plus = 0.5 * (1 + 0.4**2) * np.cos(1.4) + 1j * 0.4 * np.sin(1.4)
    cross = -0.5 * (1 + 0.4**2) * np.sin(1.4) + 1j * 0.4 * np.cos(1.4)
    kx = np.asarray(positions[:, 0]) @ np.asarray(k) / CLIGHT
    links = []
    for sc, pair, sign in LINKS:
        arg = 0.5 * frequency / FSTAR * (1 + sign * kr[pair, 0])
        links.append(
            0.25
            * np.sinc(arg / np.pi)
            * (dp[pair, 0] * plus + dc[pair, 0] * cross)
            * np.exp(-1j * (arg + 2 * np.pi * frequency * kx[sc]))
        )
    # The upstream spline response uses the reverse directed-link placement
    # from upstream fast-slow synthesis. This difference persists for static
    # equal arms, and cannot be repaired by a scalar phase/normalization.
    reversed_links = jnp.array(links)[jnp.array([2, 4, 0, 5, 1, 3])]
    legacy = np.array(
        synthesize(reversed_links[:, None], jnp.array([frequency]), 1.0, "sangria")[:3]
    )
    # No FFT here: the independent 1/(2*BW) factor is not included.
    np.testing.assert_allclose(
        tdi_response(params, times, T, positions), 2 * legacy, rtol=1e-11, atol=1e-15
    )
