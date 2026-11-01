"""Owned sources rendered over their full band on the rfft grid.

Templates are in ``dt * rfft`` units: GLASS's raw output times ``sqrt(T)``.
They are given in A and E, and in GLASS's ``T = (X + Y + Z) / sqrt(3)`` when
the data has a third channel. As in
:class:`~binary_galactica.segments.SegmentLikelihood`, an automatic (``None``)
on-fly response grid is resolved from the concrete ``T`` before compiling.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..glass import GLASS_TO_CONTINUOUS, ucb_waveform
from ..glass.constants import INV_SQ3
from .likelihood import _response_samples, quantize_bins


@dataclass(frozen=True)
class Contribution:
    """A sparse block of the published template on bins ``bin_lo .. bin_hi - 1``."""

    bin_lo: int
    values: NDArray[np.complex128]

    @property
    def bin_hi(self) -> int:
        """Return one past the last bin."""
        return self.bin_lo + self.values.shape[1]

    @classmethod
    def empty(cls, n_channels: int) -> Contribution:
        """Return a contribution with no bins."""
        return cls(0, np.zeros((n_channels, 0), dtype=np.complex128))

    def add_to(self, model: NDArray[np.complex128], sign: float = 1.0) -> None:
        """Add ``sign`` times this contribution into a full ``(channels, n_freq)`` model."""
        if self.values.shape[1]:
            model[:, self.bin_lo : self.bin_hi] += sign * self.values

    def window(self, bin_lo: int, bin_hi: int) -> NDArray[np.complex128]:
        """Return the contribution on ``[bin_lo, bin_hi)``, zero elsewhere."""
        out = np.zeros((self.values.shape[0], bin_hi - bin_lo), dtype=np.complex128)
        lo, hi = max(bin_lo, self.bin_lo), min(bin_hi, self.bin_hi)
        if hi > lo:
            out[:, lo - bin_lo : hi - bin_lo] = self.values[:, lo - self.bin_lo : hi - self.bin_lo]
        return out


@functools.lru_cache(maxsize=None)
def _renderer(BW: int, backend: str, fmt: str, response_samples: int | None) -> Callable[..., Any]:
    def one(p, T, t0):
        w = ucb_waveform(p, T=T, BW=BW, t0=t0, fmt=fmt, backend=backend,
                         response_samples=response_samples)
        c = GLASS_TO_CONTINUOUS(T)
        t = (w["X"] + w["Y"] + w["Z"]) * INV_SQ3
        return w["freq_bin_indices"], c * jnp.stack([w["A"], w["E"], t])

    return jax.jit(jax.vmap(one, in_axes=(0, None, None)))


def render_sources(
    sources: ArrayLike,
    *,
    T: float,
    t0: float,
    BW: int,
    n_channels: int,
    n_freq: int,
    backend: str = "onfly",
    fmt: str = "sangria",
    response_samples: int | None = None,
) -> Contribution:
    """Render ``(n, 8)`` sources in A, E (and T if ``n_channels == 3``).

    Bins outside ``[1, n_freq - 1)`` are dropped, so DC and Nyquist stay zero.
    """
    if n_channels not in (2, 3):
        raise ValueError("n_channels must be 2 (A, E) or 3 (A, E, T)")
    x = np.asarray(sources, dtype=np.float64).reshape(-1, 8)
    n = x.shape[0]
    if n == 0:
        return Contribution.empty(n_channels)
    resolved = _response_samples(float(T), backend, response_samples)
    batch = np.concatenate([x, np.repeat(x[:1], quantize_bins(n) - n, axis=0)])
    idx, vals = _renderer(int(BW), backend, fmt, None if resolved is None else int(resolved))(
        jnp.asarray(batch), jnp.float64(T), jnp.float64(t0)
    )
    idx = np.asarray(idx)[:n]
    vals = np.asarray(vals)[:n, :n_channels]
    if not (np.all(np.isfinite(idx)) and np.all(np.isfinite(vals))):
        raise ValueError("cannot render sources with invalid parameters")
    idx = idx.astype(np.int64)
    lo = max(int(idx.min()), 1)
    hi = min(int(idx.max()) + 1, n_freq - 1)
    out = np.zeros((n_channels, max(hi - lo, 0)), dtype=np.complex128)
    for i in range(n):
        ok = (idx[i] >= lo) & (idx[i] < hi)
        np.add.at(out.T, idx[i][ok] - lo, vals[i][:, ok].T)  # out.T is a view of out
    return Contribution(lo, out)


__all__ = ["Contribution", "render_sources"]
