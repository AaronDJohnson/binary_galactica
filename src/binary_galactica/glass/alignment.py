# Copyright 2019 Tyson B. Littenberg & Neil J. Cornish
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``ucb_alignment``: choosing a source's band the way GLASS does.

In GLASS you never pick ``BW`` by hand. ``ucb_alignment``
(``glass/ucb/src/glass_ucb_waveform.c``) derives it from the source and fixes
all the bin bookkeeping in one place:

.. code-block:: c

    source->BW   = 2*ucb_bandwidth(...);
    source->qmin = (int)(source->f0*data->T) - source->BW/2;
    source->qmax = source->qmin + source->BW;
    source->imin = source->qmin - data->qmin;
    source->imax = source->imin + source->BW;

Three details are easy to lose and all three are reproduced here:

* the **factor of two**. ``ucb_bandwidth`` returns a half-bandwidth; the
  doubling happens here, far from the heuristic, and it is what makes
  ``BW`` a power of two in ``[32, 2*Nmax]``.
* ``(int)(f0*T)`` is the **round-tripped** truncation, with ``f0`` already
  formed as ``params[0]/T``. That is not ``trunc(params[0])`` --- see the
  carrier-bin note in :mod:`binary_galactica.glass.waveform`.
* ``imin``/``imax`` are offsets into the *segment*, i.e. relative to the
  segment's own first bin, while ``qmin``/``qmax`` are absolute. Confusing
  the two silently shifts every source by the segment offset.

This is a **host-side** function: ``BW`` sets array shapes, so it has to be a
concrete Python ``int`` before
:func:`binary_galactica.glass.ucb_waveform` is traced. Call it once per
source, then jit the generator at the ``BW`` it returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _validation
from .bandwidth import ucb_bandwidth
from .constants import FSTAR, LARM

__all__ = ["Alignment", "ucb_alignment"]


@dataclass(frozen=True)
class Alignment:
    """Where a source sits on the frequency grid.

    Attributes
    ----------
    BW
        Band width in bins, and simultaneously the slow-time sample count.
    qmin, qmax
        Absolute Fourier-bin bounds, ``qmax`` exclusive.
    imin, imax
        The same bounds as offsets into the segment, i.e. relative to the
        segment's first bin.
    q
        The carrier bin, ``trunc(f0*T)``.

    """

    BW: int
    qmin: int
    qmax: int
    imin: int
    imax: int
    q: int

    @property
    def bins(self) -> np.ndarray:
        """Absolute bin index of every sample the generator will return."""
        return self.qmin + np.arange(self.BW)


def ucb_alignment(
    params,
    T: float,
    N: int,
    *,
    segment_qmin: int = 0,
    L: float = LARM,
    fstar: float = FSTAR,
    safety: float = 1.0,
) -> Alignment:
    """Derive a source's band, GLASS-style.

    Parameters
    ----------
    params
        GLASS UCB parameter vector (length >= 8). Only elements 0, 1, 3 and
        7 are read: ``f0*T``, ``cos(theta)``, ``ln(amp)`` and ``fdot*T^2``.
    T
        Observation time [s].
    N
        Positive integer length of the full frequency grid --- GLASS's
        ``data->NFFT``. Sets the ``Nmax`` clamp inside the heuristic.
    L
        Arm length used by the bandwidth noise estimate [m].
    fstar
        Transfer frequency used by the bandwidth noise estimate [Hz].
    segment_qmin
        Integer index of the segment's first bin, GLASS's ``data->qmin``.
        Only affects ``imin``/``imax``.
    safety
        Passed through to
        :func:`binary_galactica.glass.bandwidth.ucb_bandwidth`. ``1.0``
        is GLASS's exact behavior.

    Returns
    -------
    :class:`Alignment`.

    Examples
    --------
    The GLASS usage pattern, end to end::

        a = ucb_alignment(p, T, N)
        orb = binary_galactica.glass.orbit_cache(T, 0.0, a.BW)
        w = binary_galactica.glass.ucb_waveform(p, T=T, BW=a.BW, positions=orb)
        assert int(w["qmin"]) == a.qmin

    """
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 1:
        raise ValueError(
            f"params must be a one-dimensional vector; got shape {params.shape}"
        )
    if params.shape[-1] < 8:
        raise ValueError(
            f"ucb_alignment needs at least 8 parameters (fdot at index 7); "
            f"got {params.shape[-1]}"
        )
    segment_qmin = _validation.integer(segment_qmin, "segment_qmin")

    f0 = params[0] / T
    costheta = params[1]
    amp = np.exp(params[3])
    fdot = params[7] / (T * T)

    BW = 2 * ucb_bandwidth(
        f0, fdot, costheta, amp, T, N, L=L, fstar=fstar, safety=safety
    )
    # GLASS: (int)(source->f0*data->T), i.e. the round-tripped product, not
    # the raw params[0]. NumPy's float64 division matches the C directly;
    # the waveform kernel protects the same operations from JIT rewrites.
    q = int(np.trunc(f0 * T))
    qmin = q - BW // 2
    return Alignment(
        BW=BW,
        qmin=qmin,
        qmax=qmin + BW,
        imin=qmin - segment_qmin,
        imax=qmin - segment_qmin + BW,
        q=q,
    )
