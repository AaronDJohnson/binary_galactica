"""Read the LDC2a Sangria training file (application side, not package API).

The block does not build data. These helpers do it for the tools and tests:
- time-domain XYZ streams;
- their ``dt * rfft`` A/E/T spectra, in GLASS's AET convention;
- catalogue rows.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from binary_galactica.glass import xyz_to_aet  # noqa: E402
from binary_galactica.segments import CATALOGUE_FIELDS  # noqa: E402

DEFAULT_PATH = Path(
    os.environ.get(
        "BG_SANGRIA_H5",
        "/Users/aaron/Documents/lisa/project-catalog/LDC2_sangria_training_v2.h5",
    )
)
POPULATIONS = ("dgb", "igb", "vgb", "mbhb")


def _key(stream: str) -> str:
    if stream == "obs":
        return "obs/tdi"
    if stream in POPULATIONS:
        return f"sky/{stream}/tdi"
    raise ValueError(f"unknown stream {stream!r}")


def grid(path: Path = DEFAULT_PATH) -> tuple[int, float]:
    """Return ``(number of samples, dt)`` of the observation."""
    with h5py.File(path, "r") as f:
        ds = f["obs/tdi"]
        t = ds[:2, 0]["t"]
        return int(ds.shape[0]), float(t[1] - t[0])


def load_xyz(stream: str, path: Path = DEFAULT_PATH) -> tuple[dict[str, np.ndarray], float]:
    """Return ``({"X", "Y", "Z"} -> samples, dt)`` for one stream."""
    with h5py.File(path, "r") as f:
        data = f[_key(stream)][:, 0]
    dt = float(data["t"][1] - data["t"][0])
    return {ch: np.array(data[ch], dtype=np.float64) for ch in "XYZ"}, dt


def combine(
    plus: Sequence[str], minus: Iterable[str] = (), path: Path = DEFAULT_PATH
) -> tuple[dict[str, np.ndarray], float]:
    """Sum the ``plus`` streams and subtract the ``minus`` streams."""
    total, dt = load_xyz(plus[0], path)
    signed = [(name, 1.0) for name in plus[1:]] + [(name, -1.0) for name in minus]
    for name, sign in signed:
        other, other_dt = load_xyz(name, path)
        if other_dt != dt:
            raise ValueError("streams have different cadences")
        for ch in "XYZ":
            total[ch] += sign * other[ch]
    return total, dt


def aet_series(xyz: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return GLASS's A, E and T combinations of an XYZ stream."""
    A, E, T = (np.asarray(c) for c in xyz_to_aet(xyz["X"], xyz["Y"], xyz["Z"]))
    return {"A": A, "E": E, "T": T}


def aet_spectra(xyz: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    """Return ``dt * rfft`` of the A, E and T combinations of an XYZ stream."""
    return {ch: dt * np.fft.rfft(series) for ch, series in aet_series(xyz).items()}


def load_catalogue(
    population: str,
    fields: Sequence[str] = CATALOGUE_FIELDS,
    path: Path = DEFAULT_PATH,
    index: Sequence[int] | None = None,
) -> np.ndarray:
    """Return catalogue rows of one population as a 1-D structured array.

    ``index`` selects rows by position and returns them in increasing index
    order (h5py fancy indexing); ``None`` reads every row.
    """
    with h5py.File(path, "r") as f:
        view = f[f"sky/{population}/cat"].fields(list(fields))
        if index is None:
            return view[:, 0]
        order = sorted({int(i) for i in index})
        if not order:
            return view[0:0, 0]
        return view[order][:, 0]


__all__ = [
    "DEFAULT_PATH",
    "aet_series",
    "aet_spectra",
    "combine",
    "grid",
    "load_catalogue",
    "load_xyz",
]
