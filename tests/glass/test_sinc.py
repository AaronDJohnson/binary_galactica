"""Sinc values and derivatives against an independent high-precision series."""

from decimal import Decimal, localcontext
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica.glass.waveform import _sinc


def _reference(z, derivative):
    """Differentiate the entire sinc power series using exact coefficients."""
    with localcontext() as context:
        context.prec = 70
        x = Decimal(str(z))
        return float(
            sum(
                Decimal((-1) ** k * factorial(2 * k))
                * (x ** (2 * k - derivative) if 2 * k != derivative else Decimal(1))
                / Decimal(factorial(2 * k + 1) * factorial(2 * k - derivative))
                for k in range((derivative + 1) // 2, 30)
                if 2 * k >= derivative
            )
        )


@pytest.mark.parametrize("derivative", [0, 1, 2])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_sinc_and_derivatives_near_zero_and_across_taylor_boundary(
    derivative, compiled
):
    # Include both sides of the former 1e-8 cutoff and of the stable series
    # transition, plus values where cancellation used to spoil the Hessian.
    positive = np.array(
        [
            1e-10,
            np.nextafter(1e-8, 0),
            1e-8,
            np.nextafter(1e-8, np.inf),
            1e-6,
            1e-4,
            0.01,
            np.nextafter(0.1, 0),
            0.1,
            np.nextafter(0.1, np.inf),
            0.2,
            1.0,
        ]
    )
    points = np.concatenate([-positive[::-1], [0.0], positive])
    evaluate = _sinc
    for _ in range(derivative):
        evaluate = jax.grad(evaluate)
    evaluate = jax.vmap(evaluate)
    if compiled:
        evaluate = jax.jit(evaluate)

    expected = np.array([_reference(z, derivative) for z in points])
    actual = np.asarray(evaluate(jnp.asarray(points)))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-24)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_inactive_taylor_branch_does_not_overflow_large_argument_derivatives(compiled):
    # At large z, the trigonometric derivative formulas are well conditioned.
    # An unguarded high-order polynomial in the inactive branch can overflow
    # and contaminate otherwise finite derivatives through 0*inf in AD.
    z = 1e40
    expected = [np.sin(z) / z, np.cos(z) / z, -np.sin(z) / z]
    evaluate = _sinc
    for derivative in range(3):
        actual = jax.jit(evaluate)(z) if compiled else evaluate(z)
        # Terms omitted above are O(1/z**2), far below float64 precision.
        np.testing.assert_allclose(actual, expected[derivative], rtol=2e-12, atol=0)
        evaluate = jax.grad(evaluate)
