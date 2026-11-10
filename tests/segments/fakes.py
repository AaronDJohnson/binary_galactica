"""Picklable fake runners for the executor tests.

Spawned worker processes import this module by name
(``tests.segments.fakes``), so the factories must be module-level functions.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from binary_galactica.segments import ChainArchive


class ShiftRunner:
    """Brightens every source by 0.1 in ln A per turn; archives the residual norm."""

    def __init__(self, spec):
        """Start from ``spec.start``; archive to ``spec.outdir / 'cold'``."""
        self.spec = spec
        self.x = np.array(spec.start, dtype=np.float64).reshape(-1, 8)
        self.archive = ChainArchive(Path(spec.outdir) / "cold")
        self.turn = 0

    def advance(self, residual, c):
        """Brighten the sources and archive one turn."""
        self.x = self.x.copy()
        self.x[:, 3] += 0.1
        self.archive.append(
            self.turn,
            samples=np.zeros((1, 8)),
            k=[len(self.x)],
            lnlike=[float(np.sum(np.abs(residual)) + np.sum(c))],
        )
        self.turn += 1

    def cold_sources(self):
        """Return the current sources."""
        return self.x.copy()


class RaisingRunner(ShiftRunner):
    """Raises in segment 2's advance."""

    def advance(self, residual, c):
        """Raise for segment 2; otherwise advance as usual."""
        if self.spec.segment.index == 2:
            raise ValueError("injected failure in segment 2")
        super().advance(residual, c)


class DyingRunner(ShiftRunner):
    """Kills its worker process in segment 2's advance."""

    def advance(self, residual, c):
        """Exit the process for segment 2; otherwise advance as usual."""
        if self.spec.segment.index == 2:
            os._exit(3)
        super().advance(residual, c)


class StallingRunner(RaisingRunner):
    """Raises in segment 2's advance and stalls for a minute in segment 4's."""

    def advance(self, residual, c):
        """Sleep for segment 4; otherwise advance as a :class:`RaisingRunner`."""
        if self.spec.segment.index == 4:
            time.sleep(60.0)
        super().advance(residual, c)


def shift_runner(spec):
    """Return a :class:`ShiftRunner`."""
    return ShiftRunner(spec)


def raising_runner(spec):
    """Return a :class:`RaisingRunner`."""
    return RaisingRunner(spec)


def dying_runner(spec):
    """Return a :class:`DyingRunner`."""
    return DyingRunner(spec)


def stalling_runner(spec):
    """Return a :class:`StallingRunner`."""
    return StallingRunner(spec)


def failing_build(spec):
    """Raise instead of building a runner."""
    raise ValueError(f"cannot build segment {spec.segment.index}")
