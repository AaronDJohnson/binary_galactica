"""Sparse complex interpolation, including singular polar representations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica.glass.interpolation import interpolate_response


@pytest.mark.parametrize("scale", [1.0, 1e-22])
@pytest.mark.parametrize("kind", ["crossing", "touch", "near_miss", "knot"])
def test_interpolates_zero_neighborhoods_without_phase_artifacts(scale, kind):
    root = 0.5 if kind == "knot" else 0.371234

    def response(t):
        x = t - root
        a = x**2 if kind == "touch" else x
        if kind == "near_miss":
            a = a + 0.003j
        return (scale * a * jnp.exp(0.3j * t))[:, None]

    knots = jnp.linspace(0.0, 1.0, 9)
    query = jnp.linspace(0.0, 1.0, 301)
    actual = jax.jit(lambda q: interpolate_response(response, knots, q))(query)
    np.testing.assert_allclose(actual / scale, response(query) / scale, atol=2e-6)


def test_polar_interpolation_resolves_smooth_phase_and_arbitrary_query_shape():
    def response(t):
        return jnp.stack([(1 + t) * jnp.exp(19j * t), jnp.exp(-9j * t)], -1)

    knots = jnp.linspace(0.0, 1.0, 7)
    query = jnp.linspace(0.0, 1.0, 42).reshape(6, 7)
    actual = interpolate_response(response, knots, query)
    assert actual.shape == (6, 7, 2)
    np.testing.assert_allclose(
        actual, response(query.ravel()).reshape(6, 7, 2), atol=2e-12
    )


def test_refinement_evaluates_true_response_and_improves_a_crossing():
    def response(t):
        return ((jnp.expm1(2 * t) - 1.1) * jnp.exp(0.6j * t))[:, None]

    knots = jnp.array([0.0, 0.7, 1.0])
    query = jnp.linspace(0.0, 0.7, 501)
    coarse = interpolate_response(response, knots, query, max_refinements=0)
    refined = interpolate_response(response, knots, query, max_refinements=2)
    exact = response(query)
    assert jnp.max(jnp.abs(refined - exact)) < 0.25 * jnp.max(jnp.abs(coarse - exact))


def test_zero_channels_and_exact_zero_have_finite_jacobian_and_hessian():
    knots = jnp.linspace(0.0, 1.0, 5)
    query = jnp.array([0.0, 0.25, 0.371, 0.5, 1.0])

    def interpolate(p):
        def response(t):
            z = (t - p[0]) * jnp.exp(1j * p[1] * t)
            return jnp.stack([z, jnp.zeros_like(z)], -1)

        return interpolate_response(response, knots, query).real

    p = jnp.array([0.5, 0.3])
    jac = jax.jit(jax.jacfwd(interpolate))(p)
    hessian = jax.jit(jax.jacfwd(jax.jacfwd(interpolate)))(p)
    assert np.all(np.isfinite(jac))
    assert np.all(np.isfinite(hessian))
    np.testing.assert_array_equal(jac[:, 1], 0)
    step = 1e-5
    fd = np.stack(
        [
            (interpolate(p + step * e) - interpolate(p - step * e)) / (2 * step)
            for e in jnp.eye(2)
        ],
        -1,
    )
    np.testing.assert_allclose(jac, fd, atol=3e-5, rtol=3e-5)


def test_sparse_grid_convergence_and_source_batching():
    query = jnp.linspace(0.0, 1.0, 301)

    def response(p, t):
        return ((t - p) * jnp.exp(2j * t**2))[:, None]

    def error(n):
        return jnp.max(
            jnp.abs(
                interpolate_response(
                    lambda t: response(0.37, t), jnp.linspace(0.0, 1.0, n), query
                )
                - response(0.37, query)
            )
        )

    assert error(17) < error(9) / 5
    batched = jax.jit(
        jax.vmap(
            lambda p: interpolate_response(
                lambda t: response(p, t), jnp.linspace(0.0, 1.0, 9), query
            )
        )
    )(jnp.array([0.2, 0.7]))
    assert batched.shape == (2, 301, 1)
    assert np.all(np.isfinite(batched))


def test_moving_refined_minimum_has_correct_scaled_derivatives():
    knots = jnp.linspace(0.0, 1.0, 7)
    query = jnp.array([0.31, 0.37, 0.43, 0.69])

    def objective(p):
        def response(t):
            return (1e-22 * (jnp.expm1(t - p[0]) + 0.003j) * jnp.exp(1j * p[1] * t))[
                :, None
            ]

        z = interpolate_response(response, knots, query) / 1e-22
        return jnp.sum(z.real + 0.3 * z.imag)

    p = jnp.array([0.371, 0.4])
    gradient = jax.jit(jax.grad(objective))(p)
    hessian = jax.jit(jax.hessian(objective))(p)
    assert np.all(np.isfinite(hessian))
    step = 1e-5
    fd = jnp.array(
        [
            (objective(p + step * e) - objective(p - step * e)) / (2 * step)
            for e in jnp.eye(2)
        ]
    )
    np.testing.assert_allclose(gradient, fd, rtol=3e-6, atol=3e-6)
    fd_hessian = jnp.stack(
        [
            (jax.grad(objective)(p + step * e) - jax.grad(objective)(p - step * e))
            / (2 * step)
            for e in jnp.eye(2)
        ],
        -1,
    )
    np.testing.assert_allclose(hessian, fd_hessian, rtol=3e-5, atol=3e-5)


def test_nonuniform_grid_preserves_endpoint_nulls_and_zero_channel():
    def response(t):
        return jnp.stack([t * (1 - t), jnp.zeros_like(t)], -1).astype(complex)

    knots = jnp.array([0.0, 0.03, 0.2, 0.8, 0.99, 1.0])
    query = jnp.linspace(0.0, 1.0, 103)
    result = interpolate_response(response, knots, query, max_refinements=1)
    np.testing.assert_allclose(result, response(query), atol=2e-15)


def test_interval_without_representable_interior_uses_unsplit_interpolation():
    lower = np.float64(30_000_000)
    upper = np.nextafter(lower, np.inf)
    knots = jnp.array([lower, upper])

    def response(t):
        return (t - lower).astype(complex)[:, None]

    result = interpolate_response(response, knots, knots)
    np.testing.assert_array_equal(result, response(knots))


def test_sampled_tangent_zero_has_correct_scaled_reverse_gradient_and_hessian():
    knots = jnp.linspace(0.0, 1.0, 5)
    query = jnp.array([0.3, 0.5, 0.8])

    def objective(root):
        def response(t):
            return (1e-22 * (t - root) ** 2).astype(complex)[:, None]

        return jnp.sum(interpolate_response(response, knots, query).real / 1e-22)

    root = 0.5
    gradient = jax.jit(jax.grad(objective))(root)
    hessian = jax.jit(jax.hessian(objective))(root)
    assert np.isfinite(gradient)
    assert np.isfinite(hessian)
    np.testing.assert_allclose(gradient, -2 * jnp.sum(query - root), atol=2e-13)
    np.testing.assert_allclose(hessian, 2 * query.size, atol=2e-12)


def test_refinement_capacity_overflow_keeps_values_and_jacobian_finite():
    knots = jnp.linspace(0.0, 1.0, 13)
    query = jnp.linspace(0.0, 1.0, 97)

    def interpolate(root):
        def response(t):
            z = 1e-22 * jnp.sin(12 * jnp.pi * (t - root)) * jnp.exp(0.2j * t)
            return z[:, None]

        return interpolate_response(response, knots, query, max_refinements=1) / 1e-22

    # Every cell contains one crossing, exceeding the single refinement slot.
    root = 1 / 48
    values = jax.jit(interpolate)(root)
    jacobian = jax.jit(jax.jacfwd(interpolate))(root)
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(jacobian))
