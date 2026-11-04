"""Fisher decompositions held fixed for one Gibbs turn, keyed by frequency."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def floor_eigenvalues(
    evalues: ArrayLike, evectors: ArrayLike, widths: ArrayLike
) -> NDArray[np.float64]:
    """Raise eigenvalues so that no unit step leaves the prior widths.

    Along eigenvector ``v_i`` a unit step moves coordinate ``j`` by
    ``v_ji / sqrt(lambda_i)``, so ``lambda_i >= max_j (v_ji / w_j)**2``.
    Non-finite and non-positive eigenvalues take that floor.
    """
    ev = np.asarray(evalues, dtype=np.float64)
    vec = np.asarray(evectors, dtype=np.float64)
    w = np.asarray(widths, dtype=np.float64)
    floor = np.max((vec / w[:, None]) ** 2, axis=0)
    ev = np.where(np.isfinite(ev), ev, 0.0)
    return np.maximum(ev, floor)


def fisher_decomposition(
    F: ArrayLike, widths: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return floored ``(evalues, evectors)`` with ``evectors[:, i]``, or ``None``."""
    F = np.asarray(F, dtype=np.float64)
    if not np.all(np.isfinite(F)):
        return None
    evalues, evectors = np.linalg.eigh(0.5 * (F + F.T))
    return floor_eigenvalues(evalues, evectors, widths), evectors


class FisherBank:
    """Per-chain Fisher decompositions, each stored with its source's ``f0*T``.

    A source uses the stored decomposition of the same chain whose frequency is
    nearest its own, within ``match_bins``. The assignment depends only on the
    source's frequency and on matrices fixed for the turn, so a Fisher jump
    that keeps the assignment is symmetric, whatever births, deaths and swaps
    do to the slots.
    """

    def __init__(self, n_chains: int, match_bins: float) -> None:
        self.match_bins = float(match_bins)
        self._q = [np.empty(0) for _ in range(n_chains)]
        self._evalues = [np.empty((0, 8)) for _ in range(n_chains)]
        self._evectors = [np.empty((0, 8, 8)) for _ in range(n_chains)]

    def clear(self) -> None:
        """Forget every stored decomposition."""
        n = len(self._q)
        self._q = [np.empty(0) for _ in range(n)]
        self._evalues = [np.empty((0, 8)) for _ in range(n)]
        self._evectors = [np.empty((0, 8, 8)) for _ in range(n)]

    def store(
        self, chain: int, q: ArrayLike, evalues: ArrayLike, evectors: ArrayLike
    ) -> None:
        """Replace one chain's decompositions."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        self._q[chain] = q
        self._evalues[chain] = np.asarray(evalues, dtype=np.float64).reshape(q.size, 8)
        self._evectors[chain] = np.asarray(evectors, dtype=np.float64).reshape(
            q.size, 8, 8
        )

    def count(self, chain: int) -> int:
        """Return the number of decompositions stored for a chain."""
        return int(self._q[chain].size)

    def lookup(self, chain: int, q: float) -> int | None:
        """Return the index of the nearest stored source within range, else ``None``."""
        qs = self._q[chain]
        if qs.size == 0:
            return None
        d = np.abs(qs - q)
        i = int(np.argmin(d))
        return i if d[i] <= self.match_bins else None

    def decomposition(
        self, chain: int, i: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the stored ``(evalues, evectors)``."""
        return self._evalues[chain][i], self._evectors[chain][i]


__all__ = ["FisherBank", "fisher_decomposition", "floor_eigenvalues"]
