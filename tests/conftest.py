"""Shared fixtures and an independent reference implementation."""

from __future__ import annotations

import numpy as np
import pytest

from binary_galactica import GBParams

#: A generic source, deliberately with no special values: every angle is
#: off-axis and every parameter enters non-trivially.
FIDUCIAL = GBParams(
    amp=1e-22,
    f0=1.2345e-3,
    fdot=3.21e-16,
    phi0=0.7,
    iota=1.1,
    psi=0.4,
    lam=4.1,
    beta=-0.3,
)


@pytest.fixture
def params() -> GBParams:
    """Return the fiducial source."""
    return FIDUCIAL


@pytest.fixture
def t() -> np.ndarray:
    """Return a short, irregular-length time array in seconds."""
    return np.linspace(0.0, 1.0e5, 257)


def reference_polarizations(
    theta: GBParams, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an independent NumPy implementation for cross-checking.

    Written a different way from the package -- a complex exponential for the
    phase and an explicit rotation matrix for the polarization angle -- so that
    agreement is evidence about the model rather than about shared code.
    """
    amp, f0, fdot, phi0, iota, psi, _lam, _beta = (float(x) for x in theta)
    times = np.asarray(times, dtype=np.float64)

    cos_iota = np.cos(iota)
    a_plus = amp * (1.0 + cos_iota**2)
    a_cross = -2.0 * amp * cos_iota

    carrier = np.exp(1j * (phi0 + 2.0 * np.pi * f0 * times + np.pi * fdot * times**2))
    source_frame = np.stack([a_plus * carrier.real, a_cross * carrier.imag])

    rotation = np.array(
        [
            [np.cos(2.0 * psi), np.sin(2.0 * psi)],
            [-np.sin(2.0 * psi), np.cos(2.0 * psi)],
        ]
    )
    hp, hc = rotation @ source_frame
    return hp, hc
