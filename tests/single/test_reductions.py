"""Paired float32 reductions on the executing backend, compiled and op by op.

Compiled code may contract ``a*b + c`` into a fused multiply-add, and
dispatching one operation at a time rounds every product; the pairs must be
accurate either way. The spec needs the orbital phase to about 1e-7 cycles,
which plain float32 misses at multi-year times.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica.glass import _twofloat as tf
from binary_galactica.glass import single as single_impl
from tests.single.support import YEAR, chirping, middle_bin, source

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
def test_orbital_phase_on_the_fft_grid(evaluate):
    """The grid ``t0 + T*n/BW`` is reduced without rounding its times.

    Ten years late, a quarter year at nineteen years, and four years at once.
    """
    bandwidth = 4096
    for years, epoch in ((10.0, 10.0), (0.25, 19.0), (4.0, 0.0)):
        duration = np.float32(years * YEAR)
        start = np.float32(epoch * YEAR)
        phase = evaluate(
            lambda d, s: single_impl.grid_orbital_phase(d, s, bandwidth),
            duration,
            start,
        )
        times = float(start) + float(duration) * np.arange(bandwidth) / bandwidth
        assert phase_error_cycles(phase, times) < 1e-7, (years, epoch)


@EVALUATIONS
def test_epoch_expansion_matches_exact_arithmetic(evaluate):
    """``c0 + c1*v + c2*v**2 + c3*v**3`` is the float64 phase modulo whole cycles.

    Compared with rational arithmetic at the float32 inputs, for quarter-year
    observations ten and twenty years late and a ten-year one ten years late.
    The coefficients are float32 pairs. Evaluating the float64 formula
    directly in float32 rounds terms of tens of cycles at late epochs, which
    the spec's re-expansion avoids.
    """
    for years, epoch, signs in (
        (0.25, 10.0, (1, 1)),
        (0.25, 20.0, (-1, 1)),
        (10.0, 10.0, (1, -1)),
    ):
        assert_epoch_expansion_exact(evaluate, years, epoch, signs)


def assert_epoch_expansion_exact(evaluate, years, epoch, signs):
    """Require the re-expansion exact at one observation, the direct form not."""
    label = f"T{years:g}yr t0{epoch:g}yr signs {signs}"
    case = source(years, 1e-2, epoch, signs)
    residual = tf.pair(case.params[0])
    c0, c1, c2, c3 = (
        Fraction(float(hi)) + Fraction(float(lo))
        for hi, lo in evaluate(
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
    assert expanded_error < 1e-10, label
    if epoch / years >= 40:
        assert direct_error > 1e-5, label


def large_terms(kind):
    """Return a source, its observation and a carrier far from its frequency.

    ``chirp``: a 0.3 solar-mass chirp at 30 mHz, drifting about 3000 bins in
    four years, with the default carrier. ``late chirp``: the same source ten
    years later, observed for ten years with the carrier mid-observation, about
    30000 bins from the default. ``override``: a steady source at 10 mHz with
    the carrier a thousand bins away.
    """
    if kind == "override":
        case = source(4.0, 1e-2, 0.0)
        params = case.params.copy()
        params[0] = np.float32(0.3719)
        case = dataclasses.replace(case, params=params)
        return case, case.carrier - 1000
    if kind == "chirp":
        case = chirping(4.0, 3e-2, 0.3)
        return case, case.carrier
    case = chirping(10.0, 3e-2, 0.3, 10.0)
    return case, middle_bin(case)


@EVALUATIONS
def test_grid_cycles_keep_large_terms_exact(evaluate):
    """The restored grid phase stays exact when its terms reach thousands of cycles.

    For a chirp, the same chirp late with the carrier mid-observation, and a
    distant carrier override. The per-sample polynomial is summed in pairs and
    reduced modulo one before rounding, so only the final float32 rounding,
    below 2**-25 cycles, remains. Float32 Horner steps on the coefficients
    rounded to float32 miss by 1e-5 to 1e-3 cycles.
    """
    for kind in ("chirp", "late chirp", "override"):
        assert_grid_cycles_exact(evaluate, kind)


def assert_grid_cycles_exact(evaluate, kind):
    """Require the paired grid phase exact for one ``large_terms`` kind."""
    case, carrier = large_terms(kind)
    bandwidth = 4096
    offset = case.params[0]
    shift = np.float32(case.carrier - carrier)
    residual = tf.two_sum(offset, shift)

    def cycles(residual, params, duration, epoch):
        return single_impl.grid_cycles(residual, params, duration, epoch, bandwidth)

    arguments = (residual, case.params, case.duration, case.epoch)
    reduced = np.asarray(evaluate(cycles, *arguments), dtype=np.float64)
    # Reduced by the nearest whole cycle of the leading float32 part.
    assert np.all(np.abs(reduced) < 0.501), kind
    rounded = [
        np.float32(hi) + np.float32(lo)
        for hi, lo in evaluate(single_impl.epoch_coefficients, *arguments)
    ]
    v = np.arange(bandwidth, dtype=np.float32) / np.float32(bandwidth)
    plain = rounded[0] + v * (rounded[1] + v * (rounded[2] + v * rounded[3]))

    r = Fraction(float(offset)) + Fraction(float(shift))
    first, second = (Fraction(float(case.params[i])) for i in (7, 8))
    u0 = Fraction(float(case.epoch)) / Fraction(float(case.duration))

    def cycles_off(value, exact):
        difference = Fraction(value) - exact
        return abs(float((difference + Fraction(1, 2)) % 1 - Fraction(1, 2)))

    pair_error = plain_error = 0.0
    for n in range(0, bandwidth, bandwidth // 128):
        u = u0 + Fraction(n, bandwidth)
        exact = r * u + first * u**2 / 2 + second * u**3 / 6
        pair_error = max(pair_error, cycles_off(float(reduced[n]), exact))
        plain_error = max(plain_error, cycles_off(float(plain[n]), exact))
    assert pair_error < 2.0**-25 + 1e-9, kind
    assert plain_error > 1e-5, kind
