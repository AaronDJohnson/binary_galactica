"""The phase convention of the source-only model.

The polarization algebra, including the ldasoft/GBGPU sign and rotation
conventions, is checked against conftest.reference_polarizations in
test_waveform.py. Neither establishes current GLASS equivalence; independent
C-generated spectra in test_glass_reference.py exercise the detector waveform
and its normalization.
"""

from __future__ import annotations

import numpy as np

from binary_galactica import phase


def test_phase_convention_is_quadratic_with_pi_fdot(params):
    """Phi = phi0 + 2 pi f0 t + pi fdot t^2, with pi -- not pi/2 -- on fdot.

    The factor follows from f(t) = f0 + fdot t and Phi' = 2 pi f; a stray
    factor of two here is the classic galactic-binary bug.
    """
    time = 1.0e6
    expected = (
        params.phi0 + 2.0 * np.pi * params.f0 * time + np.pi * params.fdot * time**2
    )
    assert np.isclose(float(phase(params, time)), expected, rtol=1e-15)
