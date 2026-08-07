"""Double precision is a correctness requirement, so it is tested like one."""

from __future__ import annotations

from decimal import Decimal, getcontext

import jax
import jax.numpy as jnp
import numpy as np

from binary_galactica import YEAR, phase, polarizations

#: pi to 50 significant figures, for a reference phase computed outside
#: floating point entirely.
PI_50 = Decimal("3.1415926535897932384626433832795028841971693993751")


def exact_phase(theta, time: float) -> Decimal:
    """Phi(t) in 50-digit decimal arithmetic."""
    getcontext().prec = 50
    t = Decimal(repr(float(time)))
    phi0 = Decimal(repr(float(theta.phi0)))
    f0 = Decimal(repr(float(theta.f0)))
    fdot = Decimal(repr(float(theta.fdot)))
    return phi0 + 2 * PI_50 * f0 * t + PI_50 * fdot * t * t


def test_x64_is_enabled_by_importing_the_package():
    assert jax.config.read("jax_enable_x64") is True


def test_waveforms_are_float64(params, t):
    hp, hc = polarizations(params, jnp.asarray(t))
    assert hp.dtype == jnp.float64
    assert hc.dtype == jnp.float64


def test_phase_is_accurate_over_a_four_year_observation(params):
    """Absolute phase error stays far below the ~1e-3 rad that matters.

    Over four years the phase reaches ~1e6 rad.  Double precision leaves
    ~1e-9 rad of rounding; single precision would leave ~0.06 rad, which is
    enough to destroy a matched filter while remaining invisible in a plot.
    """
    for fraction in (0.25, 0.5, 1.0):
        time = 4.0 * YEAR * fraction
        computed = Decimal(repr(float(phase(params, time))))
        error = abs(computed - exact_phase(params, time))
        assert error < Decimal("1e-6"), f"phase error {error} rad at t={time}"


def test_accumulated_phase_is_large_enough_to_matter(params):
    """Sanity-check the premise of the test above."""
    total = float(phase(params, 4.0 * YEAR))
    assert total > 1e5


def test_float32_would_be_visibly_wrong(params):
    """Demonstrates why the package enables x64 on import.

    Not a test of the package so much as of the claim in its docstring: the
    same arithmetic in single precision loses the phase entirely.
    """
    time = np.float32(4.0 * YEAR)
    f0 = np.float32(params.f0)
    single = np.float32(2.0) * np.float32(np.pi) * f0 * time
    double = 2.0 * np.pi * float(params.f0) * float(4.0 * YEAR)
    assert abs(float(single) - double) > 1e-2
