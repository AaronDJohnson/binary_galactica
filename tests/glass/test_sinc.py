"""Sinc values and derivatives against an independent high-precision series.

Each backend has its own guarded ``sin(z)/z``: the Fourier backend's
``fourier._sinc`` (re-exported as ``waveform._sinc``) and the second output of
``_trig.cos_sinc``, which the default on-fly response uses. Both are checked,
and so is the cosine that ``cos_sinc`` returns beside its sinc.
"""

from decimal import Decimal, localcontext
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica.glass._trig import cos_sinc
from binary_galactica.glass.fourier import _sinc as _fourier_sinc


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


# Both sides of the former 1e-8 cutoff and of the stable series transition at
# 0.1, plus values where cancellation used to spoil the Hessian, and zero.
_POSITIVE = np.array(
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
_POINTS = np.concatenate([-_POSITIVE[::-1], [0.0], _POSITIVE])


_SINCS = {"fourier": _fourier_sinc, "onfly": lambda z: cos_sinc(z)[1]}


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("backend", sorted(_SINCS))
def test_sinc_and_derivatives_near_zero_and_at_large_argument(backend, compiled):
    """Value, gradient and Hessian against the exact series, including at 0.

    At large z, the trigonometric derivative formulas are well conditioned.
    An unguarded high-order polynomial in the inactive series branch can
    overflow there and contaminate otherwise finite derivatives through
    0*inf in AD.
    """
    large = 1e40
    # Terms omitted here are O(1/z**2), far below float64 precision.
    asymptotic = [np.sin(large) / large, np.cos(large) / large, -np.sin(large) / large]
    evaluate = _SINCS[backend]
    for derivative in range(3):
        expected = np.array([_reference(z, derivative) for z in _POINTS])
        batched = jax.vmap(evaluate)
        single = evaluate
        if compiled:
            batched, single = jax.jit(batched), jax.jit(single)
        np.testing.assert_allclose(
            np.asarray(batched(jnp.asarray(_POINTS))),
            expected,
            rtol=2e-12,
            atol=1e-24,
            err_msg=f"derivative order {derivative} near zero",
        )
        np.testing.assert_allclose(
            single(large),
            asymptotic[derivative],
            rtol=2e-12,
            atol=0,
            err_msg=f"derivative order {derivative} at z={large}",
        )
        evaluate = jax.grad(evaluate)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_cosine_beside_the_onfly_sinc(compiled):
    """The shared cosine and its derivatives, including in the series range."""
    expected = (np.cos, lambda z: -np.sin(z), lambda z: -np.cos(z))
    points = np.append(_POINTS, 1e40)

    def cosine(z):
        return cos_sinc(z)[0]

    evaluate = cosine
    for derivative in range(3):
        batched = jax.vmap(evaluate)
        if compiled:
            batched = jax.jit(batched)
        np.testing.assert_allclose(
            np.asarray(batched(jnp.asarray(points))),
            expected[derivative](points),
            rtol=2e-12,
            atol=1e-24,
            err_msg=f"derivative order {derivative}",
        )
        evaluate = jax.grad(evaluate)
