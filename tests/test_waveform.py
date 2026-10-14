"""The model reproduces the analytic identities that define it."""

from __future__ import annotations

import numpy as np
import pytest

from binary_galactica import GBParams, amplitudes, frequency, phase, polarizations
from tests.conftest import reference_polarizations


def test_matches_independent_reference(params, t):
    """Agrees with a separately written NumPy implementation."""
    hp, hc = polarizations(params, t)
    hp_ref, hc_ref = reference_polarizations(params, t)
    scale = params.amp
    assert np.allclose(hp, hp_ref, rtol=0, atol=1e-12 * scale)
    assert np.allclose(hc, hc_ref, rtol=0, atol=1e-12 * scale)


def test_chirp_is_linear_in_time(params, t):
    """The instantaneous frequency sweeps as f0 + fdot * t."""
    f = frequency(params, t)
    assert np.allclose(f, params.f0 + params.fdot * t, rtol=1e-14, atol=0)
    # And the sweep is actually present at the stated rate. The two endpoint
    # frequencies, near 1.2e-3 Hz, differ by only 3e-11 Hz, so rounding limits
    # the differenced slope to ~1e-8 relative precision.
    slope = (f[-1] - f[0]) / (t[-1] - t[0])
    assert np.isclose(slope, params.fdot, rtol=1e-6, atol=0)


def test_face_on_binary_is_circularly_polarized(params, t):
    """At iota = 0 the two polarizations are equal and in quadrature."""
    p = params._replace(iota=0.0)
    hp, hc = polarizations(p, t)
    envelope = hp**2 + hc**2
    assert np.allclose(envelope, (2.0 * p.amp) ** 2, rtol=1e-12, atol=0)


def test_edge_on_binary_is_linearly_polarized(params, t):
    """At iota = pi/2 the cross polarization vanishes.

    Exactly zero is not representable -- cos(pi/2) is 6.1e-17 in float64 --
    so the meaningful statement is that the cross polarization is negligible
    beside the plus, which it is by fourteen orders of magnitude.
    """
    p = params._replace(iota=np.pi / 2.0, psi=0.0)
    hp, hc = polarizations(p, t)
    assert np.abs(hc).max() < 1e-14 * np.abs(hp).max()


def test_amplitude_enters_linearly(params, t):
    """Doubling amp doubles the strain."""
    hp, hc = polarizations(params, t)
    hp2, hc2 = polarizations(params._replace(amp=2.0 * params.amp), t)
    assert np.allclose(hp2, 2.0 * hp, rtol=1e-14, atol=0)
    assert np.allclose(hc2, 2.0 * hc, rtol=1e-14, atol=0)


def test_polarization_angle_rotation_is_orthogonal(params, t):
    """Psi rotates the pair without changing its norm."""
    hp, hc = polarizations(params._replace(psi=0.0), t)
    rotated_hp, rotated_hc = polarizations(params._replace(psi=0.9), t)
    assert np.allclose(rotated_hp**2 + rotated_hc**2, hp**2 + hc**2, rtol=1e-12, atol=0)


def test_polarization_angle_has_period_pi(params, t):
    """The wave is spin-2: psi -> psi + pi is the identity."""
    hp, hc = polarizations(params, t)
    shifted_hp, shifted_hc = polarizations(params._replace(psi=params.psi + np.pi), t)
    assert np.allclose(shifted_hp, hp, rtol=0, atol=1e-14 * params.amp)
    assert np.allclose(shifted_hc, hc, rtol=0, atol=1e-14 * params.amp)


def test_quarter_turn_in_psi_flips_the_sign(params, t):
    """Psi -> psi + pi/2 negates both polarizations."""
    hp, hc = polarizations(params, t)
    turned_hp, turned_hc = polarizations(
        params._replace(psi=params.psi + np.pi / 2.0), t
    )
    assert np.allclose(turned_hp, -hp, rtol=0, atol=1e-14 * params.amp)
    assert np.allclose(turned_hc, -hc, rtol=0, atol=1e-14 * params.amp)


def test_phase_offset_is_a_time_shift_for_a_monochromatic_source(params, t):
    """With fdot = 0, advancing phi0 is the same as shifting time."""
    p = params._replace(fdot=0.0)
    shift = 137.0
    advanced = p._replace(phi0=p.phi0 + 2.0 * np.pi * p.f0 * shift)

    hp_shifted, hc_shifted = polarizations(p, t + shift)
    hp_advanced, hc_advanced = polarizations(advanced, t)

    assert np.allclose(hp_advanced, hp_shifted, rtol=0, atol=1e-12 * p.amp)
    assert np.allclose(hc_advanced, hc_shifted, rtol=0, atol=1e-12 * p.amp)


def test_parameters_are_defined_at_t_equals_zero(params):
    """No hidden reference epoch: at t = 0 the phase is exactly phi0."""
    assert phase(params, 0.0) == pytest.approx(params.phi0, rel=1e-15)
    assert frequency(params, 0.0) == pytest.approx(params.f0, rel=1e-15)


def test_amplitudes_bracket_the_inclination_limits():
    """A+ and Ax take their known values face-on and edge-on."""
    base = dict(amp=1.0, f0=1e-3, fdot=0.0, phi0=0.0, psi=0.0, lam=0.0, beta=0.0)
    a_plus, a_cross = amplitudes(GBParams(iota=0.0, **base))
    assert a_plus == pytest.approx(2.0)
    assert a_cross == pytest.approx(-2.0)

    a_plus, a_cross = amplitudes(GBParams(iota=np.pi / 2.0, **base))
    assert a_plus == pytest.approx(1.0)
    # Bounded by 2*cos(pi/2), which float64 puts at 1.2e-16 rather than zero.
    assert a_cross == pytest.approx(0.0, abs=1e-15)


def test_sky_position_does_not_affect_the_polarizations(params, t):
    """Lam and beta belong to the response, not to the source model."""
    hp, hc = polarizations(params, t)
    moved_hp, moved_hc = polarizations(params._replace(lam=1.0, beta=1.0), t)
    assert np.array_equal(hp, moved_hp)
    assert np.array_equal(hc, moved_hc)
