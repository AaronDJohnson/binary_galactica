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
r"""GLASS's UCB prior --- uniform in every sampled variable.

``glass/ucb/src/glass_ucb_prior.c``. GLASS samples the eight-vector directly
and puts a **flat** prior on each component, so the log prior is a constant
inside the box and :math:`-\infty` outside:

======  ============  ==========================  ===================
index   variable      prior                       bounds
======  ============  ==========================  ===================
0       :math:`f_0T`  uniform                     ``[fmin*T, fmax*T]``
1       :math:`\cos\theta`  uniform               ``[-1, 1]``
2       :math:`\lambda`     uniform               ``[0, 2*pi]``
3       :math:`\ln A`       uniform               ``[-54, -46]``
4       :math:`\cos\iota`   uniform               ``[-1, 1]``
5       :math:`\psi`        uniform               ``[0, pi]``
6       :math:`\phi_0`      uniform               ``[0, 2*pi]``
7       :math:`\dot f T^2`  uniform               empirical envelope
======  ============  ==========================  ===================

Two of these deserve to be stated rather than absorbed.

**The sky prior is already isotropic.** Uniform in :math:`\cos\theta` *is*
:math:`\sin\theta\,\mathrm{d}\theta`, so sampling this variable natively needs
no Jacobian and no sigmoid: the parameter is bounded on :math:`[-1,1]` by
construction. That is the main reason GLASS can sample natively at all.

**The amplitude prior is log-uniform.** ``p(lnA) = U[-54,-46]`` means
:math:`p(A)\propto 1/A` over
:math:`A\in[3.5\times10^{-24},\,1.0\times10^{-20}]`. A waveform generator
does not itself prescribe this prior; callers choose whether to use it.

The :math:`\dot f` bounds are GLASS's empirical MLDC envelope,

.. math::
    \dot f_{\min} = -5\times10^{-6} f_{\min}^{13/3}, \qquad
    \dot f_{\max} = 8\times10^{-7} f_{\max}^{11/3},

replaced, under ``--detached``, by the GR chirp relation over
:math:`\mathcal{M}\in[0.15,1.0]\,M_\odot`.

Longitude is **wrapped, not rejected** --- GLASS folds ``params[2]`` back into
:math:`[0,2\pi)` with ``atan2`` inside its range check. ``psi`` wraps with
period :math:`\pi` and ``phi0`` with :math:`2\pi`. Wrapping expresses the
periodic support and permits moves across each interval endpoint.

The part of the prior that is not in the prior file
---------------------------------------------------
GLASS additionally rejects any proposed source with :math:`\rho < 5`:

.. code-block:: c

    //rejection sample on SNR?
    if(sn<5.0) logPy = -INFINITY;

That lives in ``glass_ucb_sampler.c:145`` --- in the Metropolis update, not in
``glass_ucb_prior.c`` --- and it repeats in every reversible-jump birth, split
and merge move. It is nonetheless part of the prior in every sense that
matters: it truncates the posterior at :math:`\rho = 5`. Anyone porting
"GLASS's priors" by reading the prior file alone will miss it, so
:func:`snr_cut` is provided here beside the box.

It is not folded into :meth:`UCBPrior.log_prior`, because unlike the box it
cannot be evaluated from the parameters alone --- it needs the waveform and
the noise. Apply it where you have those.

This threshold is a search choice that truncates the low-amplitude posterior.
A fixed-source parameter-estimation run should apply it only when that
truncation is intended.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bandwidth import ucb_bandwidth  # noqa: F401  (keeps the module cohesive)
from .constants import TSUN

__all__ = ["SNR_MIN", "UCBPrior", "snr_cut", "ucb_fdot", "ucb_prior"]

#: ``glass_ucb_prior.c``: default ln-amplitude box.
LNA_MIN, LNA_MAX = -54.0, -46.0

#: GLASS's sampler-level SNR rejection threshold (glass_ucb_sampler.c:145).
SNR_MIN = 5.0

#: Chirp-mass box used by ``--detached`` [solar masses].
MC_MIN, MC_MAX = 0.15, 1.00

TWO_PI = 2.0 * np.pi


def ucb_fdot(Mc: float, f0: float) -> float:
    r"""``ucb_fdot``: GR chirp rate from chirp mass [Hz/s].

    :math:`\dot f = (96/5)\,\pi^{8/3}(\mathcal{M}T_\odot)^{5/3}f_0^{11/3}`.
    """
    M = Mc * TSUN
    return 19.2 * (np.pi**8 * M**5 * f0**11) ** (1.0 / 3.0)


@dataclass(frozen=True)
class UCBPrior:
    """GLASS's uniform UCB prior over the eight sampled variables.

    Build with :func:`ucb_prior`, which derives the frequency and ``fdot``
    bounds from the segment the way GLASS does. Direct construction requires
    finite, strictly ordered bounds of shape ``(8,)``. Each periodic interval
    must span its specified period. Bounds and periods are copied on input.
    """

    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    #: Indices that are wrapped rather than rejected, with their periods.
    periodic: Mapping[int, float] = field(
        default_factory=lambda: {2: TWO_PI, 5: np.pi, 6: TWO_PI}
    )

    def __post_init__(self) -> None:
        """Validate and snapshot the eight-dimensional uniform distribution."""
        lo = np.array(self.lo, dtype=np.float64, copy=True)
        hi = np.array(self.hi, dtype=np.float64, copy=True)
        if lo.shape != (8,) or hi.shape != (8,):
            raise ValueError("prior bounds must each have shape (8,)")
        with np.errstate(over="ignore", invalid="ignore"):
            widths = hi - lo
        if (
            not np.all(np.isfinite(lo))
            or not np.all(np.isfinite(hi))
            or not np.all(np.isfinite(widths))
            or np.any(widths <= 0)
        ):
            raise ValueError("prior bounds must be finite and strictly ordered")
        periodic = dict(self.periodic)
        for index, period in periodic.items():
            if (
                not isinstance(index, (int, np.integer))
                or not 0 <= index < 8
                or not np.isfinite(period)
                or period <= 0
                or not np.isclose(widths[index], period, rtol=1e-14, atol=0.0)
            ):
                raise ValueError(
                    "periodic coordinates need valid indices and finite positive "
                    "periods matching their bound widths"
                )
        lo.setflags(write=False)
        hi.setflags(write=False)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "periodic", MappingProxyType(periodic))

    @property
    def ndim(self) -> int:
        """Return the number of sampled coordinates."""
        return int(self.lo.size)

    @property
    def log_volume(self) -> float:
        """``sum(log(hi-lo))`` --- GLASS's ``logPriorVolume``, summed."""
        return float(np.sum(np.log(self.hi - self.lo)))

    def wrap(self, params: ArrayLike) -> NDArray[np.float64]:
        """Fold periodic angles into range for parameters of shape ``(..., 8)``.

        Returns a copy; the input is not mutated (GLASS mutates in place,
        which is fine in C and a trap in Python).
        """
        p = np.array(params, dtype=np.float64, copy=True)
        if p.ndim == 0 or p.shape[-1] != self.ndim:
            raise ValueError("params must have shape (..., 8)")
        for i, period in self.periodic.items():
            with np.errstate(invalid="ignore"):
                p[..., i] = self.lo[i] + np.mod(p[..., i] - self.lo[i], period)
        return p

    def log_prior(self, params: ArrayLike) -> float:
        """Evaluate one ``(8,)`` vector: ``-inf`` outside, ``-log(volume)`` inside.

        Angles are wrapped first, so a proposal that steps past ``2*pi`` is
        accepted at its folded value rather than rejected.
        """
        if np.shape(params) != (self.ndim,):
            raise ValueError("params must have shape (8,)")
        p = self.wrap(params)
        if not np.all(np.isfinite(p)):
            return -np.inf
        if np.any(p < self.lo) or np.any(p > self.hi):
            return -np.inf
        return -self.log_volume

    def contains(self, params: ArrayLike) -> bool:
        """Return whether the parameter vector lies inside the wrapped prior."""
        return bool(np.isfinite(self.log_prior(params)))

    def sample(
        self, rng: np.random.Generator, size: int | None = None
    ) -> NDArray[np.float64]:
        """Draw from the prior. Uniform in every coordinate, by construction."""
        shape = (self.ndim,) if size is None else (size, self.ndim)
        return rng.uniform(self.lo, self.hi, size=shape)


def ucb_prior(
    *,
    T: float,
    fmin: float,
    fmax: float,
    detached: bool = False,
    lnA_bounds: tuple[float, float] = (LNA_MIN, LNA_MAX),
) -> UCBPrior:
    """Build GLASS's prior for a segment spanning ``[fmin, fmax]`` Hz.

    Parameters
    ----------
    T
        Finite positive observation time [s]. Sampled variables are ``f0*T`` and
        ``fdot*T^2``, so the bounds are scaled the same way.
    fmin, fmax
        Finite analysis band [Hz], with ``0 < fmin < fmax``.
    detached
        Use the GR chirp-mass relation for the ``fdot`` bounds instead of the
        empirical MLDC envelope, over ``Mc = [0.15, 1.0]`` solar masses.
        ``--detached`` in GLASS.
    lnA_bounds
        Finite, strictly ordered ln-amplitude bounds; default ``(-54, -46)``.

    """
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and positive")
    if not (np.isfinite(fmin) and np.isfinite(fmax) and 0 < fmin < fmax):
        raise ValueError(f"need finite 0 < fmin < fmax; got {fmin}, {fmax}")
    amplitude_bounds = np.asarray(lnA_bounds, dtype=np.float64)
    if (
        amplitude_bounds.shape != (2,)
        or not np.all(np.isfinite(amplitude_bounds))
        or amplitude_bounds[0] >= amplitude_bounds[1]
    ):
        raise ValueError("lnA_bounds must be a finite, strictly ordered pair")

    if detached:
        fdotmin = ucb_fdot(MC_MIN, fmin)
        fdotmax = ucb_fdot(MC_MAX, fmax)
    else:
        # Empirical envelopes from the MLDC catalogue; note fdotmin is
        # NEGATIVE, so the box admits outspiralling (mass-transferring)
        # systems. --detached removes that by construction.
        fdotmin = -0.000005 * fmin ** (13.0 / 3.0)
        fdotmax = 0.0000008 * fmax ** (11.0 / 3.0)

    lo = np.array(
        [fmin * T, -1.0, 0.0, amplitude_bounds[0], -1.0, 0.0, 0.0, fdotmin * T * T]
    )
    hi = np.array(
        [
            fmax * T,
            1.0,
            TWO_PI,
            amplitude_bounds[1],
            1.0,
            np.pi,
            TWO_PI,
            fdotmax * T * T,
        ]
    )
    return UCBPrior(lo=lo, hi=hi)


def snr_cut(snr: float, *, snr_min: float = SNR_MIN) -> bool:
    """GLASS's sampler-level rejection: ``True`` if the source is acceptable.

    ``glass_ucb_sampler.c:145`` sets ``logPy = -INFINITY`` for any proposal
    whose optimal SNR falls below 5, and repeats the test in the birth, split
    and merge moves. It is a prior truncation implemented in the sampler.

    Kept separate from :meth:`UCBPrior.log_prior` because it needs the
    waveform and the noise, not just the parameters.

    Parameters
    ----------
    snr
        Optimal SNR of the proposed source, under whatever inner product the
        analysis uses.
    snr_min
        Threshold. ``0`` disables the cut, which is the right choice for a
        fixed-source parameter-estimation run -- see the module docstring.

    """
    return bool(np.asarray(snr) >= snr_min)
