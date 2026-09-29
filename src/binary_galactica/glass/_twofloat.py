"""Float32 pairs that carry about twice float32's precision.

A value is an unevaluated sum ``(hi, lo)`` of float32 arrays, as in
double-single arithmetic: products and quotients keep about 2**-46 relative
precision. Plain float32 resolves ``t/YEAR`` at multi-year times, or a
phase of hundreds of cycles, only to about 1e-6 cycles.

The textbook error-free product subtracts a rounded product from the exact
one. XLA may contract ``a*b + c`` into a fused multiply-add, as measured on
an arm64 CPU, which silently changes that correction. Products here are
instead formed from operands split into halves of at most 12 significant
bits, so every partial product is exact in float32 and fusing it into an
addition rounds the same exact sum. Additions use Knuth's two-sum, which has
no products. Splitting masks the low mantissa bits, so it involves no
arithmetic.

Derivatives pass through: split halves carry no tangent, a two-sum's error
term carries none, and each pair's derivative is that of the exact
expression, evaluated in float32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

#: An unevaluated sum ``hi + lo`` of float32 values.
Pair = tuple[Array, Array]

#: Clears the low 12 of float32's 23 stored mantissa bits.
_HIGH_BITS = np.int32(-(1 << 12))


def split(x) -> Pair:
    """Return ``(hi, lo)`` with ``hi + lo == x`` and at most 12 bits in each."""
    x = jnp.asarray(x, dtype=jnp.float32)
    bits = jax.lax.bitcast_convert_type(x, jnp.int32)
    hi = jax.lax.bitcast_convert_type(bits & _HIGH_BITS, jnp.float32)
    return hi, x - hi


def two_sum(a, b) -> Pair:
    """Return the rounded sum and its exact rounding error (Knuth)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def pair(x) -> Pair:
    """Return float32 ``x`` as a pair with a zero low part."""
    x = jnp.asarray(x, dtype=jnp.float32)
    return x, jnp.zeros_like(x)


def constant(value) -> tuple[np.ndarray, np.ndarray]:
    """Round a host float64 constant or array to a float32 pair."""
    value = np.asarray(value, dtype=np.float64)
    hi = value.astype(np.float32)
    return hi, (value - hi).astype(np.float32)


def product(x, y) -> Pair:
    """Return float32 ``x*y`` as a pair, from exact partial products."""
    x1, x2 = split(x)
    y1, y2 = split(y)
    s, e1 = two_sum(x1 * y1, x1 * y2)
    s, e2 = two_sum(s, x2 * y1)
    return two_sum(s, (e1 + e2) + x2 * y2)


def add(a: Pair, b: Pair) -> Pair:
    """Return the pair sum ``a + b``."""
    s, e = two_sum(a[0], b[0])
    return two_sum(s, e + (a[1] + b[1]))


def multiply(a: Pair, b: Pair) -> Pair:
    """Return the pair product ``a*b``."""
    hi, lo = product(a[0], b[0])
    # The cross terms are about 2**-24 of the product; rounding them is harmless.
    return two_sum(hi, lo + (a[0] * b[1] + a[1] * b[0]))


def divide(a, b) -> Pair:
    """Return float32 ``a/b`` as a pair, correcting the quotient's remainder."""
    quotient = a / b
    hi, lo = product(quotient, b)
    # a - hi is exact because hi is within a rounding of a.
    return two_sum(quotient, ((a - hi) - lo) / b)


def fraction(a: Pair) -> Pair:
    """Subtract the integer nearest ``a``, leaving a pair near ``[-0.5, 0.5]``."""
    hi = a[0] - jnp.round(a[0])
    return two_sum(hi, a[1])


def value(a: Pair) -> Array:
    """Round a pair to float32."""
    return a[0] + a[1]


__all__ = [
    "Pair",
    "add",
    "constant",
    "divide",
    "fraction",
    "multiply",
    "pair",
    "product",
    "split",
    "two_sum",
    "value",
]
