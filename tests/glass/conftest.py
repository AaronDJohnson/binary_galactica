"""Package-local fixtures for the GLASS backend tests."""

from __future__ import annotations

import numpy as np
import pytest

INJECTION = {
    "T": 7_884_000.0,
    "f0": 1e-3,
    "fdot": 1e-18,
    "amp": 7.1562127917748725e-22,
    "phi0": 0.5,
    "colat": 1.7806669190576343,
    "lon": 0.0,
    "psi": 0.7,
    "cos_iota": 0.3,
}


@pytest.fixture(scope="session")
def injection():
    """Return the native injection used throughout the backend tests."""
    return dict(INJECTION)


@pytest.fixture(scope="session")
def glass_params():
    """Return the injection in the native GLASS sampling coordinates."""
    p = INJECTION
    return np.array(
        [
            p["f0"] * p["T"],
            np.cos(p["colat"]),
            p["lon"],
            np.log(p["amp"]),
            p["cos_iota"],
            p["psi"],
            p["phi0"],
            p["fdot"] * p["T"] ** 2,
        ],
        dtype=np.float64,
    )
