"""Value-only complex interpolation of the sparse response, including through nulls."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica.glass.sparse import _coefficient_matrix, interpolate_uniform


# 4: the shortest grid, a single window of fewer than eight points; 17: full
# eight-point windows with one-sided endpoint stencils.
@pytest.mark.parametrize("count", [4, 17])
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


def test_nulls_preserve_the_complex_response():
    """Channels: a null between knots, a null on a knot, and exact zero.

    The interpolant is linear in the complex samples, so every null takes the
    same code path; these two are the ones that break amplitude/phase (polar)
    interpolation. Physical strain amplitude rules out absolute thresholds.
    """
    scale = 1e-22

    def response(t):
        def null(root):
            return scale * (t - root) * jnp.exp(0.3j * t)

        return jnp.stack((null(0.371), null(0.5), jnp.zeros_like(t)), axis=-1)

    query = jnp.linspace(0.0, 1.0, 257)
    actual = interpolate_uniform(response(jnp.linspace(0.0, 1.0, 33)), query)
    np.testing.assert_allclose(actual / scale, response(query) / scale, atol=2e-12)
    np.testing.assert_array_equal(actual[:, 2], 0)


def test_query_derivative_at_a_sample_is_not_artificially_zero():
    knots = jnp.linspace(0.0, 1.0, 17)
    values = (knots**3 + 1j * knots**2)[:, None]
    derivative = jax.jacfwd(lambda t: interpolate_uniform(values, t))(0.5)
    np.testing.assert_allclose(derivative, jnp.array([0.75 + 1j]), atol=2e-13)


# One guard: (3, 1) has too few samples for a stencil, (4, 0) no channels.
@pytest.mark.parametrize("shape", [(3, 1), (4, 0)])
def test_rejects_invalid_sample_shapes(shape):
    with pytest.raises(ValueError, match="values must have shape"):
        interpolate_uniform(jnp.zeros(shape), jnp.array([0.5]))


def _equations(jaxpr):
    """Yield every equation, descending into nested (for example pjit) jaxprs."""
    for equation in jaxpr.eqns:
        yield equation
        for value in equation.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    yield from _equations(inner)


def test_window_samples_are_not_gathered_from_the_response():
    # XLA fuses an elementwise producer into a consuming gather and recomputes it
    # per gathered element. Gathering the overlapping windows straight from the
    # response therefore repeated the response ~8 times; the windows must come
    # from an operation that materializes the samples once.
    values = jnp.zeros((17, 2), dtype=jnp.complex128)
    jaxpr = jax.make_jaxpr(interpolate_uniform)(values, jnp.linspace(0.0, 1.0, 5))
    equations = list(_equations(jaxpr.jaxpr))
    gathered = [
        tuple(var.aval.shape)
        for eq in equations
        if eq.primitive.name == "gather"
        for var in eq.invars[:1]
    ]
    assert values.shape not in gathered
    assert any(eq.primitive.name == "conv_general_dilated" for eq in equations)


def _gathered_windows_reference(values, coordinates):
    """Interpolate with the original direct window gather, as an exact reference."""
    values = jnp.asarray(values, dtype=jnp.complex128)
    count, channels = values.shape
    points = min(8, count)
    center = points // 2 - 1
    windows = jnp.arange(count - points + 1)[:, None] + jnp.arange(points)[None, :]
    samples = values[windows]
    anchor = samples[:, center, :]
    matrix = jnp.asarray(_coefficient_matrix(points), dtype=jnp.float64)
    coefficients = jnp.einsum("kj,wjc->wck", matrix, samples - anchor[:, None, :])
    coefficients = coefficients.at[..., 0].add(anchor)
    sample_coordinate = coordinates.ravel() * (count - 1)
    cell = jnp.floor(sample_coordinate).astype(jnp.int32)
    window = jnp.clip(cell - center, 0, count - points)
    local = sample_coordinate - (window + center)
    selected = coefficients[window]
    result = selected[..., -1]
    for degree in range(points - 2, -1, -1):
        result = result * local[:, None] + selected[..., degree]
    return result.reshape(coordinates.shape + (channels,))


# 5: an odd grid shorter than one window (centre from the point count);
# 8: exactly one window; 33: many overlapping eight-point windows.
@pytest.mark.parametrize("count", [5, 8, 33])
def test_window_selection_matches_direct_indexing_exactly(count):
    rng = np.random.default_rng(count)
    values = rng.normal(size=(count, 3)) + 1j * rng.normal(size=(count, 3))
    query = jnp.asarray(np.concatenate(([0.0, 1.0], rng.uniform(size=300))))
    np.testing.assert_array_equal(
        jax.jit(interpolate_uniform)(values, query),
        jax.jit(_gathered_windows_reference)(values, query),
    )


def test_nonfinite_sample_only_affects_windows_that_contain_it():
    count, bad = 33, 16
    values = np.ones((count, 1), dtype=complex)
    values[bad] = np.nan
    query = jnp.linspace(0.0, 1.0, 513)
    actual = np.asarray(interpolate_uniform(values, query))[:, 0]
    window = np.clip(np.floor(np.asarray(query) * (count - 1)).astype(int) - 3, 0, 25)
    contains = (window <= bad) & (bad <= window + 7)
    np.testing.assert_array_equal(np.isnan(actual), contains)
    np.testing.assert_array_equal(actual[~contains], 1.0)
