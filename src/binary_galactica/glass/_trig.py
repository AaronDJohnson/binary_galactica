"""Cosine-sine pairs whose derivatives reuse their forward values.

The on-fly kernel rotates each delayed source difference by a phase. Written
as ``jnp.exp(1j * x)``, every rotation is a complex exponential, which XLA
evaluates as ``exp(0) * (cos(x), sin(x))``: an extra real exponential per
element. :func:`cis` forms ``cos(x) + 1j*sin(x)`` from the pair directly, and
:func:`cos_sinc` lets one pair supply both ``cos(h)`` and ``sin(h)/h``.

The derivative of ``cos`` is ``-sin``, so reverse mode needs the pair again.
:func:`cos_sin` has a custom JVP rule that evaluates the pair once behind
``jax.lax.optimization_barrier`` and reuses it for the tangent. How much that
helps depends on XLA. With JAX 0.11 on CPU, backward consumers then read the
stored pair, and a value-and-gradient call of the on-fly waveform evaluates
as many transcendentals as its forward pass, about a fifth of the count with
``jnp.exp``. Without the barrier XLA fuses fresh cosines and sines into those
consumers and evaluates more transcendentals than with ``jnp.exp``. XLA in
JAX 0.6 and 0.9 does that even with the barrier. The waveform's dense phase
restore therefore keeps ``jnp.exp``; see ``waveform._ucb_waveform``.

The barrier appears only under differentiation: undifferentiated programs
compile as with ``jnp.cos`` and ``jnp.sin``. Higher orders differentiate the
rule itself, including the barrier, which requires JAX 0.6 or later.
"""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def cos_sin(x):
    """Return ``(cos(x), sin(x))`` in the dtype of real ``x``.

    The values are those of ``jnp.cos`` and ``jnp.sin``; derivatives reuse
    them as described in the module docstring.
    """
    return jnp.cos(x), jnp.sin(x)


@cos_sin.defjvp
def _cos_sin_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    # Materialize the pair once; the tangent and the backward pass read it.
    cos, sin = jax.lax.optimization_barrier((jnp.cos(x), jnp.sin(x)))
    return (cos, sin), (-sin * dx, cos * dx)


def cis(x):
    """Return ``exp(1j * x)`` for real ``x`` as ``cos(x) + 1j*sin(x)``.

    The result is complex64 for float32 ``x`` and complex128 for float64.
    """
    return jax.lax.complex(*cos_sin(x))


def cos_sinc(x):
    """Return ``(cos(x), sin(x)/x)`` from one evaluation of the pair.

    Below ``|x| = 0.1`` the ratio is its Taylor series through ``x**10``,
    whose first omitted term is below float64 rounding there. The division
    sees 1 in place of a small ``x`` and the series sees 0 in place of a large
    one, so neither ``0/0`` nor an overflowing polynomial reaches the value or
    any derivative order through the inactive branch.
    """
    small = jnp.abs(x) < 0.1
    safe = jnp.where(small, 1.0, x)
    z = jnp.where(small, x, 0.0) ** 2
    series = 1 + z * (
        -1 / 6 + z * (1 / 120 + z * (-1 / 5040 + z * (1 / 362880 - z / 39916800)))
    )
    cos, sin = cos_sin(x)
    return cos, jnp.where(small, series, sin / safe)
