"""LDC catalogue rows <-> GLASS-native 8-vectors, in host NumPy float64.

The native vector is :func:`binary_galactica.parameters.to_glass_vector`'s
``[f0*T, sin(beta), lambda, ln A, cos(iota), psi, phi0_G, fdot*T**2]``, computed
in NumPy so that it needs no JAX precision flag. The LDC ``InitialPhase`` is
GLASS's ``phi0_G`` itself, i.e. ``to_glass_vector`` applied to a package
``phi0 = -InitialPhase`` (measured against Sangria's noise-free streams in
``tools/segments/conventions_gate.py``). Longitude and ``phi0_G`` are
folded into ``[0, 2*pi)`` and ``psi`` into ``[0, pi)``: the periodic intervals
of :class:`~binary_galactica.glass.UCBPrior`. The waveform is unchanged by
``psi -> psi + pi``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Fields every start-set row must carry (the ``sky/dgb/cat`` names).
CATALOGUE_FIELDS = (
    "Amplitude",
    "EclipticLatitude",
    "EclipticLongitude",
    "Frequency",
    "FrequencyDerivative",
    "Inclination",
    "InitialPhase",
    "Polarization",
)

TWO_PI = 2.0 * np.pi


def _column(rows: Any, name: str) -> NDArray[np.float64]:
    try:
        values = rows[name]
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"catalogue rows have no {name!r} field") from exc
    return np.asarray(values, dtype=np.float64).reshape(-1)


def ldc_to_native(rows: Any, T: float) -> NDArray[np.float64]:
    """Convert LDC catalogue rows to ``(n, 8)`` GLASS-native vectors.

    ``rows`` is a NumPy structured array of any shape (h5py reads Sangria
    catalogues as ``(n, 1)``) or a mapping from field name to values.

    >>> row = {"Amplitude": 1e-22, "EclipticLatitude": 0.3,
    ...        "EclipticLongitude": 1.0, "Frequency": 3e-3,
    ...        "FrequencyDerivative": 1e-17, "Inclination": 0.5,
    ...        "InitialPhase": 1.0, "Polarization": 4.0}
    >>> ldc_to_native(row, 1.0e7).shape
    (1, 8)
    """
    if not (np.isfinite(T) and T > 0):
        raise ValueError("T must be finite and positive")
    if not isinstance(rows, Mapping) and getattr(rows, "dtype", None) is None:
        raise ValueError("rows must be a structured array or a mapping of fields")
    cols = {name: _column(rows, name) for name in CATALOGUE_FIELDS}
    n = cols["Frequency"].size
    if any(col.size != n for col in cols.values()):
        raise ValueError("catalogue fields differ in length")
    if not all(np.all(np.isfinite(col)) for col in cols.values()):
        raise ValueError("catalogue rows must be finite")
    if np.any(cols["Amplitude"] <= 0) or np.any(cols["Frequency"] <= 0):
        raise ValueError("Amplitude and Frequency must be positive")
    native = np.empty((n, 8), dtype=np.float64)
    native[:, 0] = cols["Frequency"] * T
    native[:, 1] = np.sin(cols["EclipticLatitude"])
    native[:, 2] = np.mod(cols["EclipticLongitude"], TWO_PI)
    native[:, 3] = np.log(cols["Amplitude"])
    native[:, 4] = np.cos(cols["Inclination"])
    native[:, 5] = np.mod(cols["Polarization"], np.pi)
    native[:, 6] = np.mod(cols["InitialPhase"], TWO_PI)
    native[:, 7] = cols["FrequencyDerivative"] * T * T
    return native


def native_to_ldc(native: ArrayLike, T: float) -> dict[str, NDArray[np.float64]]:
    """Convert ``(n, 8)`` native vectors to LDC catalogue fields."""
    x = np.asarray(native, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError(f"native vectors must have shape (n, 8), got {x.shape}")
    return {
        "Amplitude": np.exp(x[:, 3]),
        "EclipticLatitude": np.arcsin(np.clip(x[:, 1], -1.0, 1.0)),
        "EclipticLongitude": x[:, 2].copy(),
        "Frequency": x[:, 0] / T,
        "FrequencyDerivative": x[:, 7] / (T * T),
        "Inclination": np.arccos(np.clip(x[:, 4], -1.0, 1.0)),
        "InitialPhase": np.mod(x[:, 6], TWO_PI),
        "Polarization": x[:, 5].copy(),
    }


__all__ = ["CATALOGUE_FIELDS", "ldc_to_native", "native_to_ldc"]
