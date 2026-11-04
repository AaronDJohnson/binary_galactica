"""GLASS's single-source moves as impulse custom jumps (all symmetric).

Each move picks one active source uniformly and proposes a new value for it.
It returns the current row at ``k = 0`` or when the move declines. impulse
wraps the periodic coordinates afterwards, and the prior rejects anything
outside the box.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..glass import UCBPrior, draw_from_fisher, fm_shift, psi_phi_jump
from .fisher import FisherBank


class SlotMove:
    """Base class: uniform choice of an active slot, symmetric proposal."""

    def __call__(self, chain_stats: Any) -> tuple[NDArray[np.float64], float]:
        """Return ``(new row, 0.0)`` for impulse's ``add_custom_jump``."""
        row = np.array(chain_stats.current_sample, dtype=np.float64)
        layout = chain_stats.layout
        k = layout.k(row)
        if k == 0:
            return row, 0.0
        rng = chain_stats.rng
        sl = layout.slot(int(rng.integers(k)))
        new = self.propose(row[sl].copy(), rng, int(chain_stats.chain_index))
        if new is not None:
            row[sl] = new
        return row, 0.0

    def propose(
        self, source: NDArray[np.float64], rng: np.random.Generator, chain: int
    ) -> NDArray[np.float64] | None:
        """Return the proposed source, or ``None`` to stay."""
        raise NotImplementedError


class FisherMove(SlotMove):
    """One eigendirection of the turn-fixed Fisher matrix nearest in frequency."""

    def __init__(self, bank: FisherBank) -> None:
        """Store the Fisher bank the proposal reads."""
        self.bank = bank

    def propose(self, source, rng, chain):
        """Jump along the matched decomposition; decline if the match would change."""
        i = self.bank.lookup(chain, float(source[0]))
        if i is None:
            return None
        evalues, evectors = self.bank.decomposition(chain, i)
        new, _ = draw_from_fisher(source, rng, evalues, evectors)
        if self.bank.lookup(chain, float(new[0])) != i:
            return None
        return new


class PriorRedrawMove(SlotMove):
    """Redraw all eight parameters of the source from the segment prior.

    Returning ``qxy = 0`` is valid only because ``prior`` is the same uniform
    box as the per-slot prior of the target :class:`SegmentPrior`; callers must
    pass that same prior.
    """

    def __init__(self, prior: UCBPrior) -> None:
        """Store the prior the redraw samples from."""
        self.prior = prior

    def propose(self, source, rng, chain):
        """Return a fresh prior draw."""
        return self.prior.sample(rng)


class FmShiftMove(SlotMove):
    """GLASS's annual-modulation sideband hop, with longitude jitter."""

    def __init__(self, T: float) -> None:
        """Store the period ``T`` used to shift frequencies."""
        self.T = float(T)

    def propose(self, source, rng, chain):
        """Shift ``f0*T`` by ``s*T/yr`` and jitter the longitude."""
        return fm_shift(source, rng, self.T)[0]


class PsiPhiMove(SlotMove):
    """GLASS's continuous psi-phi0 correlation jump."""

    def propose(self, source, rng, chain):
        """Shift ``psi`` by ``+-d/2`` and ``phi0`` by ``+-d``."""
        return psi_phi_jump(source, rng)[0]


__all__ = ["FisherMove", "FmShiftMove", "PriorRedrawMove", "PsiPhiMove", "SlotMove"]
