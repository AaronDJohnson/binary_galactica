"""Joint prior of one segment: uniform count times GLASS's box per source."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..glass import UCBPrior


class SegmentPrior:
    """``log p(k) + sum_{i<k} UCBPrior.log_prior(x_i)`` for padded impulse rows.

    ``p(k)`` is uniform on ``k_min .. k_max``; slots past ``k`` are ignored.
    Periodic angles are wrapped before the box test, as in
    :meth:`UCBPrior.log_prior`. Rows are ``(n, 8 * k_max)`` with ``k`` of
    shape ``(n,)``, impulse's vectorized convention.
    """

    def __init__(self, prior: UCBPrior, k_max: int, k_min: int = 0) -> None:
        """Store the box prior and the allowed range of source counts."""
        if not 0 <= int(k_min) <= int(k_max):
            raise ValueError("need 0 <= k_min <= k_max")
        self.prior = prior
        self.k_max = int(k_max)
        self.k_min = int(k_min)
        self.log_pk = -math.log(self.k_max - self.k_min + 1)
        self.log_component = -prior.log_volume

    def __call__(self, x: ArrayLike, k: ArrayLike) -> NDArray[np.float64]:
        """Return the log prior of each row."""
        rows = np.atleast_2d(np.asarray(x, dtype=np.float64))
        ks = np.atleast_1d(np.asarray(k)).astype(np.int64)
        comps = rows.reshape(rows.shape[0], self.k_max, 8)
        wrapped = self.prior.wrap(comps)
        with np.errstate(invalid="ignore"):
            inside = np.all(
                (wrapped >= self.prior.lo) & (wrapped <= self.prior.hi), axis=-1
            )
        active = np.arange(self.k_max)[None, :] < ks[:, None]
        ok = np.all(inside | ~active, axis=1) & (ks >= self.k_min) & (ks <= self.k_max)
        return np.where(ok, self.log_pk + ks * self.log_component, -np.inf)


__all__ = ["SegmentPrior"]
