"""Paired float32 reductions on the executing backend, compiled and op by op.

Compiled code may contract ``a*b + c`` into a fused multiply-add, and
dispatching one operation at a time rounds every product; the pairs must be
accurate either way. The spec needs the orbital phase to about 1e-7 cycles,
which plain float32 misses at multi-year times.
"""

from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica.glass import _twofloat as tf
from binary_galactica.glass import single as single_impl
from tests.single.support import YEAR, source

PI2 = 2 * np.pi


def compiled(function, *args):
    """Evaluate in one XLA computation."""
    return jax.jit(function)(*args)


def dispatched(function, *args):
    """Evaluate one operation at a time."""
    with jax.disable_jit():
        return function(*args)


EVALUATIONS = pytest.mark.parametrize("evaluate", [compiled, dispatched])


def total(pair) -> np.ndarray:
    """Return ``hi + lo`` in float64, exact for float32 pairs."""
    return np.asarray(pair[0], dtype=np.float64) + np.asarray(pair[1], dtype=np.float64)


def full_mantissas(rng, count):
    """Signed float32 values over sixty binades, most with 24 significant bits."""
    values = rng.uniform(1, 2, count) * 2.0 ** rng.integers(-30, 30, count)
    return (values * rng.choice([-1, 1], count)).astype(np.float32)


@EVALUATIONS
def test_two_sum_is_exact(evaluate):
    rng = np.random.default_rng(11)
    a, b = full_mantissas(rng, 500), full_mantissas(rng, 500)
    s, e = evaluate(tf.two_sum, a, b)
    for i in range(a.size):
        exact = Fraction(float(a[i])) + Fraction(float(b[i]))
        assert Fraction(float(s[i])) + Fraction(float(e[i])) == exact, i


@EVALUATIONS
def test_products_and_quotients(evaluate):
    rng = np.random.default_rng(12)
    x, y = full_mantissas(rng, 20_000), full_mantissas(rng, 20_000)
    for function, exact in (
        (tf.product, x.astype(np.float64) * y),
        (tf.divide, x.astype(np.float64) / y),
    ):
        pair = evaluate(function, x, y)
        assert pair[0].dtype == pair[1].dtype == np.float32
        relative = np.abs(total(pair) - exact) / np.abs(exact)
        assert np.max(relative) < 2.0**-44, function.__name__


@EVALUATIONS
def test_pair_arithmetic(evaluate):
    rng = np.random.default_rng(13)
    a = tf.constant(rng.uniform(-1e3, 1e3, 20_000))
    b = tf.constant(rng.uniform(-1e3, 1e3, 20_000))
    exact_a, exact_b = total(a), total(b)
    product = total(evaluate(tf.multiply, a, b))
    assert np.max(np.abs(product / (exact_a * exact_b) - 1)) < 2.0**-44
    sum_ = total(evaluate(tf.add, a, b))
    scale = np.abs(exact_a) + np.abs(exact_b)
    assert np.max(np.abs(sum_ - (exact_a + exact_b)) / scale) < 2.0**-45


@EVALUATIONS
def test_fraction_removes_whole_cycles_exactly(evaluate):
    rng = np.random.default_rng(14)
    cycles = tf.constant(rng.uniform(-40, 40, 5000))
    reduced = evaluate(tf.fraction, cycles)
    value = total(reduced)
    assert np.all(np.abs(value) <= 0.5 + 2.0**-20)
    whole = total(cycles) - value
    np.testing.assert_array_equal(whole, np.round(whole))


def phase_error_cycles(phase, times) -> float:
    """Largest angular error of ``phase`` from ``2*pi*t/YEAR``, in cycles."""
    exact = PI2 * np.asarray(times, dtype=np.float64) / YEAR
    difference = np.asarray(phase, dtype=np.float64) - exact
    return float(np.max(np.abs(np.remainder(difference + np.pi, PI2) - np.pi)) / PI2)


@EVALUATIONS
def test_orbital_phase_at_arbitrary_times(evaluate):
    """Float32 times to twenty years, each taken as exact."""
    rng = np.random.default_rng(15)
    times = np.concatenate(
        [np.linspace(-3.2e8, 6.4e8, 20_001), rng.uniform(5.9e8, 6.4e8, 5000)]
    ).astype(np.float32)
    phase = evaluate(single_impl.orbital_phase, times)
    assert phase.dtype == np.float32
    assert phase_error_cycles(phase, times) < 1e-7
    plain = PI2 * jnp.asarray(times) / np.float32(YEAR)
    assert phase_error_cycles(plain, times) > 1e-6


@EVALUATIONS
@pytest.mark.parametrize("years, epoch", [(10.0, 10.0), (0.25, 19.0), (4.0, 0.0)])
def test_orbital_phase_on_the_fft_grid(evaluate, years, epoch):
    """The grid ``t0 + T*n/BW`` is reduced without rounding its times."""
    duration = np.float32(years * YEAR)
    start = np.float32(epoch * YEAR)
    bandwidth = 4096
    phase = evaluate(
        lambda d, s: single_impl.grid_orbital_phase(d, s, bandwidth), duration, start
    )
    times = float(start) + float(duration) * np.arange(bandwidth) / bandwidth
    assert phase_error_cycles(phase, times) < 1e-7


@EVALUATIONS
@pytest.mark.parametrize(
    "years, epoch, signs",
    [(0.25, 10.0, (1, 1)), (0.25, 20.0, (-1, 1)), (10.0, 10.0, (1, -1))],
)
def test_epoch_expansion_matches_exact_arithmetic(evaluate, years, epoch, signs):
    """``c0 + c1*v + c2*v**2 + c3*v**3`` is the float64 phase modulo whole cycles.

    Compared with rational arithmetic at the float32 inputs. The coefficients
    are good to float32 rounding of terms of a few cycles. Evaluating the
    float64 formula directly in float32 rounds terms of tens of cycles at late
    epochs, which the spec's re-expansion avoids.
    """
    case = source(years, 1e-2, epoch, signs)
    residual = tf.pair(case.params[0])
    c0, c1, c2, c3 = (
        Fraction(float(c))
        for c in evaluate(
            single_impl.epoch_coefficients,
            residual,
            case.params,
            case.duration,
            case.epoch,
        )
    )
    r, first, second = (Fraction(float(case.params[i])) for i in (0, 7, 8))
    u0 = Fraction(float(case.epoch)) / Fraction(float(case.duration))
    p = case.params
    v32 = np.arange(case.bandwidth, dtype=np.float32) / case.bandwidth
    u32 = (case.epoch + case.duration * v32) / case.duration
    direct = p[0] * u32 + np.float32(0.5) * p[7] * u32**2 + p[8] * u32**3 / 6

    def cycles_off(value, exact):
        return abs(float((value - exact + Fraction(1, 2)) % 1 - Fraction(1, 2)))

    expanded_error = direct_error = 0.0
    for n in range(0, case.bandwidth, max(1, case.bandwidth // 64)):
        v = Fraction(n, case.bandwidth)
        u = u0 + v
        exact = r * u + first * u**2 / 2 + second * u**3 / 6
        expanded = c0 + v * (c1 + v * (c2 + v * c3))
        expanded_error = max(expanded_error, cycles_off(expanded, exact))
        direct_error = max(direct_error, cycles_off(Fraction(float(direct[n])), exact))
    assert expanded_error < 5e-7
    if epoch / years >= 40:
        assert direct_error > 1e-5
