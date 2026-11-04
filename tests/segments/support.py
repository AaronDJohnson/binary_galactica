"""Shared helpers for the segment-block tests (not a conftest: import it)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "segments"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

CATALOG = Path.home() / "Documents/lisa/project-catalog"
SANGRIA = Path(
    os.environ.get(
        "BG_SANGRIA_H5",
        str(CATALOG / "data/raw/sangria/LDC2_sangria_training_v2.h5"),
    )
)


def _sangria_missing() -> str | None:
    if importlib.util.find_spec("h5py") is None:
        return "h5py is not installed"
    if not SANGRIA.exists():
        return "Sangria training file not found; set BG_SANGRIA_H5"
    return None


requires_sangria = pytest.mark.skipif(
    _sangria_missing() is not None, reason=_sangria_missing() or ""
)

#: A short grid that keeps tests fast: 2**16 samples at 15 s, about 11 days.
#: A source with ln A = -48 at 3 mHz has SNR ~38 here against (T/2)*ae_noise.
DT = 15.0
N_TIME = 2**16
T_SHORT = N_TIME * DT
N_FREQ = N_TIME // 2 + 1


def rj_impulse():
    """Import impulse with ``RJPTSampler`` or skip (needs the ``rjmcmc`` branch)."""
    impulse = pytest.importorskip("impulse")
    if not hasattr(impulse, "RJPTSampler"):
        pytest.skip("impulse lacks RJPTSampler; put the rjmcmc branch on PYTHONPATH")
    return impulse


def ldc_rows(**overrides) -> np.ndarray:
    """Return one LDC catalogue row as a structured array, with field overrides."""
    values = {
        "Amplitude": 2.0e-22,
        "EclipticLatitude": 0.3,
        "EclipticLongitude": 1.7,
        "Frequency": 3.0e-3,
        "FrequencyDerivative": 1.0e-17,
        "Inclination": 0.9,
        "InitialPhase": 2.1,
        "Polarization": 0.6,
    }
    values.update(overrides)
    dtype = [(name, np.float64) for name in values]
    return np.array([tuple(values.values())], dtype=dtype)


def native_source(q: float, lnA: float = -48.0, fdot_T2: float = 0.0) -> np.ndarray:
    """Return a GLASS-native source at ``f0*T = q`` with generic angles."""
    return np.array([q, 0.2, 1.1, lnA, 0.3, 0.7, 2.0, fdot_T2])
