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
r"""GLASS's bandwidth heuristic, ``ucb_bandwidth``.

``glass/ucb/src/glass_ucb_waveform.c``. GLASS sizes the band --- and
therefore, because the two are locked, the slow-time grid --- as the larger
of two power-of-two estimates:

**Doppler spread.**
    :math:`\mathrm{bw} = 2T\big[(4 + 2\pi f\,\mathrm{AU}\sin\theta/c)/
    \mathrm{yr} + |\dot f| T\big]` bins. Note the ``/YEAR``: this is
    Carson's rule with the modulation frequency in *bins*, so it scales
    with how many years were observed. A heuristic that instead adds the
    modulation index :math:`\beta` to a bin count silently assumes
    :math:`T \approx 1` yr.

**Sinc spread.**
    the next power of two above the analytic SNR.

The second term is the interesting one. It makes the bandwidth --- and so
the slow-grid refinement --- scale with :math:`\rho`, which is the right
law: the reduced-DFT amplitude error falls as :math:`1/BW`, so the mismatch
falls as :math:`1/BW^2`, and the indistinguishability requirement
:math:`1-\mathrm{FF} < 1/(2\rho^2)` is met when :math:`BW \propto \rho`.
GLASS's constant is roughly a factor two short of that (see
``README.md``); :func:`ucb_bandwidth` therefore takes a ``safety`` multiplier,
defaulting to GLASS's exact behaviour at ``safety=1``.
"""

from __future__ import annotations

import numpy as np

from .constants import AU, CLIGHT, FSTAR, LARM, PI2, SACC, SLOC, SPS, YEAR


def rednoise(f):
    """``rednoise``, glass_lisa.c:1183."""
    return 1.0 + 16.0 * ((2.0e-5 / f) ** 10.0 + (1.0e-4 / f) ** 2)


def ae_noise(f, L: float = LARM, fstar: float = FSTAR):
    """``AEnoise``, glass_lisa.c. One-sided A/E noise PSD.

    Used only to size the band; the analysis PSD lives elsewhere.
    """
    x = f / fstar
    trans = np.sin(x) ** 2
    return (
        (16.0 / 3.0)
        * trans
        * (
            (2.0 + np.cos(x)) * (SPS + SLOC)
            + 2.0
            * (3.0 + 2.0 * np.cos(x) + np.cos(2.0 * x))
            * (SLOC / 2.0 + SACC / (PI2 * f) ** 4 * rednoise(f))
        )
        / (2.0 * L) ** 2
    )


def analytic_snr(A, Sn, Sf, sqT):
    """``analytic_snr``: ``A*sqrt(T)*sin(f/fstar)/sqrt(Sn)``.

    GLASS's own comment: *"not exactly what's in paper--calibrated against
    (h|h)"*. It is an order-of-magnitude estimator, and the ``next_pow2``
    that consumes it rounds away the calibration anyway.
    """
    return A * sqT * Sf / np.sqrt(Sn)


def _next_pow2(x: float) -> int:
    """GLASS's ``(int)pow(2,(int)log2(x-1)+1)``.

    Rounds ``x`` up to a power of two, with two edge cases worth naming:
    it returns 1 for ``x <= 1`` (``log2`` of a non-positive number), and for
    ``x`` exactly a power of two it returns ``x`` itself, not ``2x``,
    because the ``-1`` drops it below the next exponent.
    """
    if x <= 1.0:
        return 1
    return int(2 ** (int(np.log2(x - 1.0)) + 1))


def ucb_bandwidth(
    f,
    fdot,
    costheta,
    amp,
    T,
    N,
    *,
    L: float = LARM,
    fstar: float = FSTAR,
    safety: float = 1.0,
) -> int:
    """``ucb_bandwidth``: half-bandwidth in bins.

    Parameters
    ----------
    f, fdot, costheta, amp
        Source frequency [Hz], drift [Hz/s], cosine colatitude, amplitude.
    T
        Observation time [s].
    N
        Length of the full frequency grid; sets the ``Nmax`` clamp.
    L
        Arm length used by the analytic noise curve [m].
    fstar
        Transfer frequency used by the analytic noise curve [Hz].
    safety
        Multiplier applied to the SNR term before rounding up to a power of
        two. ``1.0`` reproduces GLASS exactly. Values above 1 buy
        reduced-DFT accuracy at proportional cost --- see the discussion
        above and in ``README.md``.

    Returns
    -------
    The half-bandwidth. GLASS's ``ucb_alignment`` **doubles** this to get
    ``source->BW``, which is the ``BW`` passed to ``ucb_waveform``; use
    :func:`ucb_bw` if you want that number directly.

    """
    Nmin = 16
    Nmax = int(2 ** int(np.log2(N / 2)))
    sqT = np.sqrt(T)

    sintheta = np.sin(np.arccos(costheta))
    bw = 2.0 * T * ((4.0 + PI2 * f * (AU / CLIGHT) * sintheta) / YEAR + abs(fdot) * T)
    # Clamp ORDER matters. GLASS writes
    #     if(DS > Nmax) DS = Nmax;
    #     if(DS < Nmin) DS = Nmin;
    # i.e. the Nmax ceiling first and the Nmin floor LAST, so the floor wins
    # unconditionally. That is max(min(x, Nmax), Nmin), not the more natural
    # min(max(x, Nmin), Nmax) -- the two disagree whenever Nmax < Nmin, which
    # a short grid (N < 64) produces.
    DS = max(min(_next_pow2(bw), Nmax), Nmin)

    snrm = analytic_snr(amp, ae_noise(f, L, fstar), np.sin(f / fstar), sqT)
    SS = max(min(_next_pow2(snrm * safety), Nmax), Nmin)

    return max(DS, SS)


def ucb_bw(f, fdot, costheta, amp, T, N, **kw) -> int:
    """Return the generator bandwidth, ``2 * ucb_bandwidth``.

    GLASS applies this factor of two in ``ucb_alignment``
    (``source->BW = 2*ucb_bandwidth(...)``), far from the heuristic itself.
    Missing it halves the band and quarters the slow grid.
    """
    return 2 * ucb_bandwidth(f, fdot, costheta, amp, T, N, **kw)


__all__ = ["ae_noise", "analytic_snr", "rednoise", "ucb_bandwidth", "ucb_bw"]
