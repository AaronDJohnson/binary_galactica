"""Value-only complex interpolation and source derivatives through nulls."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica.glass.sparse import interpolate_uniform


@pytest.mark.parametrize("count", [4, 6, 8, 17])
def test_reproduces_polynomials_through_degree_seven_including_endpoints(count):
    knots = jnp.linspace(0.0, 1.0, count)
    query = jnp.linspace(0.0, 1.0, 133).reshape(7, 19)
    degree = min(7, count - 1)

    def response(t):
        return jnp.stack(
            ((t - 0.37) ** degree + 1j * (0.5 + t) ** 3, jnp.ones_like(t) * (2 - 3j)),
            axis=-1,
        )

    actual = jax.jit(interpolate_uniform)(response(knots), query)
    assert actual.shape == (7, 19, 2)
    np.testing.assert_allclose(actual, response(query), atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("scale", [1.0, 1e-22])
@pytest.mark.parametrize("kind", ["crossing", "touch", "near_miss", "exact_knot"])
def test_nulls_and_near_misses_preserve_the_complex_response(scale, kind):
    root = 0.5 if kind == "exact_knot" else 0.371

    def response(t):
        amplitude = (t - root) ** 2 if kind == "touch" else t - root
        if kind == "near_miss":
            amplitude = amplitude + 0.003j
        z = scale * amplitude * jnp.exp(0.3j * t)
        return jnp.stack((z, jnp.zeros_like(z)), axis=-1)

    query = jnp.linspace(0.0, 1.0, 257)
    actual = interpolate_uniform(response(jnp.linspace(0.0, 1.0, 33)), query)
    np.testing.assert_allclose(actual / scale, response(query) / scale, atol=2e-12)
    np.testing.assert_array_equal(actual[:, 1], 0)


@pytest.mark.parametrize("root", [0.371, 0.5])
def test_source_gradients_and_hessians_match_analytic_tangent_response(root):
    knots = jnp.linspace(0.0, 1.0, 17)
    query = jnp.array([0.3, 0.5, 0.8])

    def response(p, t):
        z = 1e-22 * (t - p[0]) ** 2 * jnp.exp(1j * p[1] * t)
        return jnp.stack((z, jnp.zeros_like(z)), axis=-1)

    def interpolated(p):
        return interpolate_uniform(response(p, knots), query) / 1e-22

    def exact(p):
        return response(p, query) / 1e-22

    p = jnp.array([root, 0.3])
    actual_jac = jax.jit(jax.jacfwd(interpolated))(p)
    np.testing.assert_allclose(actual_jac, jax.jacfwd(exact)(p), atol=2e-11)

    def objective(p):
        return jnp.sum(interpolated(p).real)

    def exact_objective(p):
        return jnp.sum(exact(p).real)

    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(p), jax.grad(exact_objective)(p), atol=2e-11
    )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(objective))(p), jax.hessian(exact_objective)(p), atol=2e-10
    )
    batch = jax.jit(jax.vmap(interpolated))(jnp.array([[root, 0.3], [0.4, 0.4]]))
    assert np.all(np.isfinite(batch))


def test_eighth_order_convergence_includes_one_sided_endpoint_stencils():
    query = jnp.linspace(0.0, 1.0, 1001)

    def response(t):
        return ((t - 0.37) * jnp.exp(1j * (11 * t + 3 * t**2)))[:, None]

    errors = []
    for count in (17, 33):
        actual = interpolate_uniform(response(jnp.linspace(0.0, 1.0, count)), query)
        errors.append(jnp.max(jnp.abs(actual - response(query))))
    assert errors[1] < errors[0] / 60


def test_query_derivative_at_a_sample_is_not_artificially_zero():
    knots = jnp.linspace(0.0, 1.0, 17)
    values = (knots**3 + 1j * knots**2)[:, None]
    derivative = jax.jacfwd(lambda t: interpolate_uniform(values, t))(0.5)
    np.testing.assert_allclose(derivative, jnp.array([0.75 + 1j]), atol=2e-13)


@pytest.mark.parametrize("shape", [(3, 1), (4,), (4, 1, 1), (4, 0)])
def test_rejects_invalid_sample_shapes(shape):
    with pytest.raises(ValueError):
        interpolate_uniform(jnp.zeros(shape), jnp.array([0.5]))
