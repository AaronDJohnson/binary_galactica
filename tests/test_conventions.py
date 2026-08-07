"""The sign conventions match ``ldasoft`` and ``GBGPU`` exactly.

These are the tests that make the package's output safe to hand to an existing
LISA response code.  Every other property in the suite would be satisfied just
as well by a waveform with ``psi -> -psi`` or ``Ax -> -Ax``; only this file
rules those out.
"""

from __future__ import annotations

import numpy as np

from binary_galactica import phase, polarizations


def ldasoft_coefficients(
    amp: float, iota: float, psi: float
) -> tuple[float, float, float, float]:
    """The ``DPr, DPi, DCr, DCi`` coefficients as written in ``ldasoft``.

    Transcribed from ``galactic_binary()`` in ldasoft's ``galactic.c`` (and
    identical in ``GBGPU``'s ``run_wave``, where they appear as the complex
    ``DP = Aplus*cosps - 1j*Across*sinps`` and
    ``DC = -Aplus*sinps - 1j*Across*cosps``).
    """
    cosi = np.cos(iota)
    cosps = np.cos(2.0 * psi)
    sinps = np.sin(2.0 * psi)

    a_plus = amp * (1.0 + cosi * cosi)
    a_cross = -amp * (2.0 * cosi)

    dpr = a_plus * cosps
    dpi = -a_cross * sinps
    dcr = -a_plus * sinps
    dci = -a_cross * cosps
    return dpr, dpi, dcr, dci


def test_polarizations_reproduce_ldasoft_coefficients(params, t):
    """h+ and hx equal ldasoft's plus/cross response factors.

    ldasoft contracts ``(DPr, DPi)`` and ``(DCr, DCi)`` against a wave written
    as ``e^{-i Phi}``.  Stripping the detector geometry -- unit plus and cross
    tensors, zero light-travel delay -- the factor multiplying each tensor is

        h+ = DPr cos(Phi) - DPi sin(Phi)
        hx = DCr cos(Phi) - DCi sin(Phi)

    which must be what this package returns.
    """
    dpr, dpi, dcr, dci = ldasoft_coefficients(params.amp, params.iota, params.psi)
    phi = np.asarray(phase(params, t))

    expected_hp = dpr * np.cos(phi) - dpi * np.sin(phi)
    expected_hc = dcr * np.cos(phi) - dci * np.sin(phi)

    hp, hc = polarizations(params, t)
    assert np.allclose(hp, expected_hp, rtol=0, atol=1e-14 * params.amp)
    assert np.allclose(hc, expected_hc, rtol=0, atol=1e-14 * params.amp)


def test_cross_amplitude_is_negative_for_a_face_on_binary(params):
    """Ax = -2 A cos(iota), not +2 A cos(iota).

    The overall sign of the cross polarization is a genuine convention, and
    getting it backwards flips the handedness of every circularly polarized
    source.  ldasoft and GBGPU both carry the minus sign.
    """
    p = params._replace(iota=0.0, psi=0.0, phi0=0.0, fdot=0.0)
    # A quarter period after t = 0 the plus term vanishes and hc = Ax.
    quarter_period = 0.25 / p.f0
    _, hc = polarizations(p, np.array([quarter_period]))
    assert hc[0] < 0.0
    assert np.isclose(hc[0], -2.0 * p.amp, rtol=1e-9)


def test_positive_psi_rotates_in_the_ldasoft_direction(params, t):
    """Guards the sign of the 2*psi rotation.

    ldasoft's ``DCr = -A+ sin(2 psi)`` fixes the direction: at small positive
    psi and phi0 = 0, the cross channel picks up a *negative* multiple of the
    plus amplitude.
    """
    p = params._replace(psi=0.05, phi0=0.0, fdot=0.0)
    _, hc = polarizations(p, np.array([0.0]))
    cos_iota = np.cos(p.iota)
    a_plus = p.amp * (1.0 + cos_iota**2)
    assert np.isclose(hc[0], -a_plus * np.sin(2.0 * p.psi), rtol=1e-12)


def test_phase_convention_is_quadratic_with_pi_fdot(params):
    """Phi = phi0 + 2 pi f0 t + pi fdot t^2, with pi -- not pi/2 -- on fdot.

    The factor follows from f(t) = f0 + fdot t and Phi' = 2 pi f; a stray
    factor of two here is the classic galactic-binary bug.
    """
    time = 1.0e6
    expected = (
        params.phi0
        + 2.0 * np.pi * params.f0 * time
        + np.pi * params.fdot * time**2
    )
    assert np.isclose(float(phase(params, time)), expected, rtol=1e-15)
