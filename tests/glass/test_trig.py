"""Cosine-sine pairs whose derivatives reuse the forward values.

``glass._trig`` replaces ``jnp.exp(1j * x)`` in the on-fly response kernel. Its
values must be exactly ``jnp.cos`` and ``jnp.sin``, every derivative order must
be the analytic one, and the barrier that lets XLA reuse the pair in backward
consumers must appear only under differentiation, so undifferentiated programs
keep their fusion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.conftest import nested_jaxprs

from binary_galactica.glass._trig import cis, cos_sin
from binary_galactica.glass.onfly import tdi_response

#: Zero, both signs, small and large arguments, and full phases near 1e7 rad.
POINTS = np.array([0.0, 1e-9, -0.3, 0.1, 1.0, -2.5, 3.7, 40.0, -1.2e3, 9.8e6])
#: Analytic derivatives of (cos, sin), order 0 to 3.
DERIVATIVES = (
    (np.cos, np.sin),
    (lambda x: -np.sin(x), np.cos),
    (lambda x: -np.cos(x), lambda x: -np.sin(x)),
    (np.sin, lambda x: -np.cos(x)),
)
DURATION = 7_884_000.0
NATIVE = np.array(
    [3e-3 * DURATION + 0.25, -0.3, 5.1, np.log(1e-22), -0.8, 1.2, 2.1, -0.4, 0.2]
)
TIMES = np.linspace(0.0, DURATION, 9)


def primitives(jaxpr):
    """Return the names of every primitive in a jaxpr and its inner jaxprs."""
    return {eqn.primitive.name for inner in nested_jaxprs(jaxpr) for eqn in inner.eqns}


def complex_exponentials(jaxpr):
    """Return every ``exp`` equation with a complex result."""
    return [
        eqn
        for inner in nested_jaxprs(jaxpr)
        for eqn in inner.eqns
        if eqn.primitive.name == "exp"
        and np.issubdtype(eqn.outvars[0].aval.dtype, np.complexfloating)
    ]


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
def test_values_are_exactly_the_cosine_and_sine(dtype, mode):
    """Forward values are bitwise those of ``jnp.cos`` and ``jnp.sin``."""
    x = jnp.asarray(POINTS, dtype=dtype)

    def plain(y):
        return jnp.cos(y), jnp.sin(y)

    pair, reference = cos_sin, plain
    if mode == "jit":
        pair, reference = jax.jit(pair), jax.jit(reference)
    elif mode == "vmap":
        pair, reference = jax.vmap(pair), jax.vmap(reference)
    for actual, expected in zip(pair(x), reference(x), strict=True):
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)
    rotation = cis(x)
    assert rotation.dtype == jnp.promote_types(dtype, jnp.complex64)
    np.testing.assert_array_equal(rotation.real, jnp.cos(x))
    np.testing.assert_array_equal(rotation.imag, jnp.sin(x))
    np.testing.assert_allclose(
        rotation, jnp.exp(1j * x), rtol=4 * np.finfo(dtype).eps, atol=0
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_derivatives_through_third_order(compiled):
    """Reverse, forward and mixed modes, batched; this differentiates the barrier."""
    x = jnp.asarray(POINTS)
    for component in range(2):

        def value(y, component=component):
            return cos_sin(y)[component]

        reverse = forward = value
        for order in range(4):
            expected = DERIVATIVES[order][component](POINTS)
            for function in (reverse, forward):
                batched = jax.vmap(function)
                if compiled:
                    batched = jax.jit(batched)
                np.testing.assert_allclose(
                    batched(x),
                    expected,
                    rtol=1e-15,
                    atol=1e-15,
                    err_msg=f"component {component}, order {order}",
                )
            reverse = jax.grad(reverse)
            forward = jax.jacfwd(forward)
    hessian = jax.hessian(lambda y: jnp.sum(jnp.abs(cis(y) * (1 + 2j)) ** 2 * y**2))
    # |cis(y) (1 + 2i)|^2 = 5, so the Hessian is 10 times the identity.
    np.testing.assert_allclose(hessian(x[:4]), 10 * np.eye(4), rtol=1e-13, atol=1e-13)


def test_barrier_appears_only_under_differentiation():
    x = jnp.asarray(POINTS)

    def loss(y):
        cos, sin = cos_sin(y)
        return jnp.sum(cos * y + sin)

    assert "optimization_barrier" not in primitives(jax.make_jaxpr(cos_sin)(x))
    assert "optimization_barrier" not in primitives(jax.make_jaxpr(cis)(x))
    for derivative in (
        jax.grad(loss),
        lambda y: jax.jvp(loss, (y,), (jnp.ones_like(y),)),
        jax.hessian(loss),
    ):
        assert "optimization_barrier" in primitives(jax.make_jaxpr(derivative)(x))


@pytest.mark.parametrize("reference_phase", [False, True])
def test_response_kernel_uses_the_pair(reference_phase):
    """No complex exponential; a barrier only once differentiated."""

    def response(params):
        return tdi_response(params, TIMES, DURATION, reference_phase=reference_phase)

    def loss(params):
        return jnp.sum(jnp.abs(response(params)) ** 2)

    params = jnp.asarray(NATIVE)
    forward = jax.make_jaxpr(response)(params)
    assert not complex_exponentials(forward)
    assert "optimization_barrier" not in primitives(forward)
    reverse = jax.make_jaxpr(jax.grad(loss))(params)
    assert not complex_exponentials(reverse)
    assert "optimization_barrier" in primitives(reverse)
