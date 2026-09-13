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
r"""Host-side UCB proposals adapted from ``glass_ucb_proposal.c``.

These NumPy helpers operate on the eight native GLASS sampling coordinates.
Apply the prior and wrap periodic angles before accepting a proposal.

The division of labour
----------------------
========================  =====  =====  ====================================
proposal                  GLASS  here   what it covers
========================  =====  =====  ====================================
:func:`draw_from_fisher`    0.3    0.7  local correlations
F-statistic draw/jump       0.4     --  global search; see below
:func:`fm_shift`            0.1    0.1  annual-modulation sidebands in f0
:func:`psi_phi_jump`        0.1    0.1  phase-polarization correlations
:class:`PriorDraw`          0.1    0.1  independent prior draws
========================  =====  =====  ====================================

The "here" column is :func:`single_source_mix`. Its fixed source count omits
GLASS's F-statistic search proposals and gives their mixture weight to the
Fisher component. Extra search proposals can be supplied explicitly.

The Fisher proposal draws along **one randomly chosen eigenvector** of a
supplied information matrix. A decomposition fixed independently of the chain
is a symmetric coordinate rotation. Re-linearizing at the current state is an
adaptive proposal and requires the qualification in the calling convention
below.

``fm_shift`` hops between annual-modulation sidebands. ``psi_phi_jump`` makes
continuous phase-polarization shifts, which are generally not exact waveform
symmetries; the discrete ``(delta phi0, delta psi) = (pi, pi/2)`` shift is.

Calling convention
------------------
Each proposal takes the current parameter vector and returns
``(proposed, log_q_ratio)`` where ``log_q_ratio`` is
:math:`\log q(x|y) - \log q(y|x)`, to be **added** to the Metropolis
log-acceptance ratio. The prior, sideband, and phase-polarization proposals
are symmetric or independence draws from a uniform density. The Fisher
proposal returns ``0.0`` only for a decomposition held fixed independently of
the chain state. Recomputing it from the current state requires a justified
adaptive-sampling scheme or an explicit reverse-density correction.

Non-finite Fisher jumps and correlated jumps that reach either cosine boundary
become whole-vector self-proposals. Every bundled proposal returns a new array.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import YEAR
from .prior import UCBPrior

ProposalResult = tuple[NDArray[np.float64], float]
Proposal = Callable[[ArrayLike, np.random.Generator], ProposalResult]
FisherDecomposition = tuple[NDArray[np.float64], NDArray[np.float64]]
FisherOracle = Callable[[ArrayLike], FisherDecomposition]

__all__ = [
    "FISHER_REFRESH_STEPS",
    "FisherCache",
    "FisherProposal",
    "GLASS_MCMC_WEIGHTS",
    "ProposalMix",
    "PriorDraw",
    "ExtrinsicPriorDraw",
    "draw_from_fisher",
    "fm_shift",
    "psi_phi_jump",
    "single_source_mix",
]

#: GLASS's fixed-dimension MCMC weights (``initialize_ucb_proposal``), for the
#: proposals a single-source run uses. ``fisher`` is deliberately absent: GLASS
#: gives it **the remainder**, ``proposal[4]->weight -= check``, so it is
#: derived rather than set. See :func:`single_source_mix`.
GLASS_MCMC_WEIGHTS = {
    "prior": 0.1,
    "fm_shift": 0.1,
    "psi_phi": 0.1,
    # "fstat_draw": 0.2,   # omitted for a single source -- see single_source_mix
    # "fstat_jump": 0.2,
}

#: Back-compat alias with ``fisher`` resolved, for direct :class:`ProposalMix`
#: construction.
DEFAULT_WEIGHTS = dict(
    GLASS_MCMC_WEIGHTS, fisher=1.0 - sum(GLASS_MCMC_WEIGHTS.values())
)

#: Default refresh interval in calls to :class:`FisherCache`. The value comes
#: from GLASS's ``10 * UCB_MODEL_NP`` inner steps per outer iteration, but this
#: helper counts Fisher calls, not total steps of a mixed sampler.
FISHER_REFRESH_STEPS = 80

#: Indices whose absolute value must not exceed 1 (the two cosines).
_COSINE_IDX = (1, 4)


def draw_from_fisher(
    params: ArrayLike,
    rng: np.random.Generator,
    evalues: ArrayLike,
    evectors: ArrayLike,
) -> ProposalResult:
    """``draw_from_fisher``: one random eigendirection of the local Fisher.

    Parameters
    ----------
    params
        Current 8-vector.
    rng
        ``numpy.random.Generator``.
    evalues, evectors
        Fisher eigen-decomposition, with
        ``evectors[:, i]`` the eigenvector for ``evalues[i]`` --- the same
        layout as GLASS's ``fisher_evectr[j][i]``. Keep this decomposition
        fixed for a zero-correction production proposal.

    Notes
    -----
    Two details are transcribed directly:

    * the step is ``N(0,1)/sqrt(lambda_i)`` along **one** eigenvector chosen
      uniformly. GLASS draws a full vector of amplitudes and then uses only
      the chosen component, which is the same thing;
    * there is **no** ``1/sqrt(NP)`` scaling. The line is present in the C and
      commented out, precisely because only one direction moves per jump.

    This helper intentionally departs from GLASS's guards. A non-finite jump
    or output becomes a whole-vector self-proposal instead of GLASS's
    deterministic one-percent replacement. If either cosine reaches a closed
    boundary, the whole correlated jump is likewise rejected instead of
    reverting only that coordinate. These self-proposals preserve the support
    symmetry of a fixed Fisher decomposition, so their log proposal correction
    is zero.

    """
    params = np.asarray(params, dtype=np.float64)
    n = params.size

    # Invalid eigenvalues are allowed to propagate to a non-finite jump and
    # are handled by the whole-vector rejection below.
    with np.errstate(divide="ignore", invalid="ignore"):
        # All n normals are consumed before the direction is chosen. Keep the
        # order to keep the RNG stream comparable with the C.
        amps = rng.standard_normal(n) / np.sqrt(evalues)
        # GLASS: (int)(rand_r_U_0_1(seed)*NP), where its U(0,1) is inclusive
        # of 1.0, can index one past the end with probability ~1/RAND_MAX.
        # NumPy's half-open uniform avoids that out-of-bounds index.
        i = int(rng.uniform() * n)
        jump = amps[i] * np.asarray(evectors)[:, i]

    if not np.all(np.isfinite(jump)):
        return params.copy(), 0.0

    with np.errstate(over="ignore", invalid="ignore"):
        out = params + jump
    if not np.all(np.isfinite(out)):
        return params.copy(), 0.0
    if any(k < n and (out[k] >= 1.0 or out[k] <= -1.0) for k in _COSINE_IDX):
        return params.copy(), 0.0
    return out, 0.0


def fm_shift(params: ArrayLike, rng: np.random.Generator, T: float) -> ProposalResult:
    """``fm_shift``: hop to an annual-modulation sideband.

    The Doppler modulation frequency in **bins** is ``fm = T/YEAR``, and the
    proposal moves ``f0*T`` by ``scale*fm`` for an integer
    ``scale`` drawn uniformly from ``{-3,-2,-1,1,2,3}`` (GLASS draws
    ``floor(-3 + 7*U)`` and rejects 0). Longitude is jittered by
    ``U(-0.2, 0.2)`` rad at the same time, because the sideband ambiguity is
    partly a sky ambiguity.

    The sideband spacing is fixed by the observation time.
    """
    params = np.asarray(params, dtype=np.float64)
    fm = T / YEAR

    scale = 0
    while scale == 0:
        scale = int(np.floor(-3.0 + 7.0 * rng.uniform()))

    out = params.copy()
    out[0] += scale * fm
    out[2] += -0.2 + rng.uniform() * 0.4
    return out, 0.0


def psi_phi_jump(params: ArrayLike, rng: np.random.Generator) -> ProposalResult:
    r"""Draw a continuous :math:`\psi`--:math:`\phi_0` correlation proposal.

    The discrete shift :math:`(\delta\phi_0,\delta\psi)=(\pi,\pi/2)` is an
    exact waveform symmetry at general inclination. Continuous shifts along
    :math:`|\delta\psi|=|\delta\phi_0|/2` are not generally invariant. Strong
    continuous correlations arise near face-on or face-off orientations,
    becoming exact at those special orientations for the appropriate sign.

    GLASS draws ``scale ~ U(0, 2*pi)``, sets ``|d_phi| = scale`` and
    ``|d_psi| = scale/2``, then chooses each sign independently. This symmetric
    proposal needs the usual likelihood acceptance test.
    """
    params = np.asarray(params, dtype=np.float64)
    scale = rng.uniform() * 2.0 * np.pi
    d_phi = scale if rng.uniform() >= 0.5 else -scale
    d_psi = 0.5 * scale if rng.uniform() >= 0.5 else -0.5 * scale

    out = params.copy()
    out[5] += d_psi
    out[6] += d_phi
    return out, 0.0


@dataclass
class PriorDraw:
    """``draw_from_prior``: an independence draw from the uniform box.

    The forward and reverse densities are the same constant, so the ratio is
    zero. GLASS returns the density itself; the difference never reaches the
    acceptance ratio.
    """

    prior: UCBPrior

    def __call__(self, params: ArrayLike, rng: np.random.Generator) -> ProposalResult:
        """Draw and return a fresh parameter vector from the prior."""
        return self.prior.sample(rng), 0.0


@dataclass
class ExtrinsicPriorDraw:
    """``draw_from_extrinsic_prior``: redraw the extrinsic block only.

    Indices 1, 2 (sky) and 4, 5, 6 (``cos iota``, ``psi``, ``phi0``) are
    redrawn from their uniform priors; ``f0``, ``ln A`` and ``fdot`` are held.
    Same constant-density argument as :class:`PriorDraw`.
    """

    prior: UCBPrior
    indices: tuple[int, ...] = (1, 2, 4, 5, 6)

    def __call__(self, params: ArrayLike, rng: np.random.Generator) -> ProposalResult:
        """Redraw the configured extrinsic coordinates from the prior."""
        out = np.asarray(params, dtype=np.float64).copy()
        for n in self.indices:
            out[n] = rng.uniform(self.prior.lo[n], self.prior.hi[n])
        return out, 0.0


@dataclass
class FisherProposal:
    """:func:`draw_from_fisher` bound to a Fisher oracle.

    Parameters
    ----------
    fisher
        ``fisher(params) -> (evalues, evectors)``, with ``evectors[:, i]``
        belonging to ``evalues[i]``. A state-dependent oracle is adaptive;
        this wrapper does not compute its reverse-density correction.

    Zero-correction constraint
    --------------------------
    :func:`draw_from_fisher` is symmetric for a fixed decomposition in the
    interior, including its whole-vector self-proposals at invalid draws and
    cosine boundaries. If this oracle recomputes the decomposition from
    ``params``, the resulting proposal is state-dependent and its zero
    correction is not generally valid. Freeze the decomposition during
    production sampling, supply the correct forward/reverse proposal ratio,
    or use a separately justified adaptive-sampling scheme.

    :class:`FisherCache` can reduce repeated decomposition work. It counts
    calls to this Fisher component, so a mixture does not reproduce GLASS's
    refresh cadence of 80 total inner sampler steps.

    """

    fisher: FisherOracle

    def __call__(self, params: ArrayLike, rng: np.random.Generator) -> ProposalResult:
        """Draw one proposal using the Fisher decomposition at ``params``."""
        evalues, evectors = self.fisher(params)
        return draw_from_fisher(params, rng, evalues, evectors)


class ProposalMix:
    """Weighted mixture, selected the way GLASS selects.

    GLASS picks a uniform index and accepts it with probability equal to that
    proposal's weight, looping until one is accepted --- which is a categorical
    draw with probabilities proportional to the weights. We do the categorical
    directly; it is the same distribution without the rejection loop.
    """

    def __init__(
        self,
        proposals: Mapping[str, Proposal],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Build a categorical mixture from named proposal callables."""
        weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        missing = set(proposals) - set(weights)
        if missing:
            raise ValueError(f"no weight given for {sorted(missing)}")
        self.names = sorted(proposals)
        self.proposals = dict(proposals)
        w = np.array([weights[n] for n in self.names], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            total = w.sum()
        if (
            not np.all(np.isfinite(w))
            or np.any(w < 0)
            or not np.isfinite(total)
            or total <= 0
        ):
            raise ValueError(
                "weights must be finite and non-negative with a finite positive sum"
            )
        self.p = w / total

    def __call__(
        self, params: ArrayLike, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], float, str]:
        """Draw a proposal and apply it. Returns ``(proposed, log_q, name)``."""
        name = self.names[int(rng.choice(len(self.names), p=self.p))]
        out, logq = self.proposals[name](params, rng)
        return out, logq, name


class FisherCache:
    """Cache a Fisher decomposition for a configured number of Fisher calls.

    GLASS refreshes its decomposition once per outer iteration, every
    ``10*NP = 80`` total inner sampler steps. This donor-compatible Python
    helper instead refreshes on the first call and every ``every`` calls
    thereafter: with ``every=80``, calls 1, 81, 161, etc. refresh the cache.
    Other components selected by :class:`ProposalMix` do not advance its
    counter. It therefore differs from GLASS's total-step refresh cadence.

    A refresh based on the current chain position remains state-dependent
    adaptation. Caching it for several jumps does not by itself make the
    zero-correction Fisher proposal stationary or prove detailed balance.
    Freeze a production decomposition independently of chain state, compute
    the reverse-density correction, or separately justify the adaptive
    sampling scheme.

    Parameters
    ----------
    fisher
        ``fisher(params) -> (evalues, evectors)``.
    every
        Refresh interval in calls to this cache.

    """

    def __init__(self, fisher: FisherOracle, every: int = FISHER_REFRESH_STEPS) -> None:
        """Initialize a Fisher cache with a positive refresh interval."""
        if every < 1:
            raise ValueError("refresh interval must be >= 1 Fisher call")
        self._fisher = fisher
        self.every = int(every)
        self._cached: FisherDecomposition | None = None
        self._since = 0
        self.refreshes = 0

    def __call__(self, params: ArrayLike) -> FisherDecomposition:
        """Return the cached decomposition, refreshing on schedule."""
        if self._cached is None or self._since >= self.every:
            self._cached = self._fisher(params)
            self._since = 0
            self.refreshes += 1
        self._since += 1
        return self._cached

    def invalidate(self) -> None:
        """Force a refresh on the next call."""
        self._since = self.every


def single_source_mix(
    *,
    prior: UCBPrior,
    fisher: FisherOracle,
    T: float,
    every: int = FISHER_REFRESH_STEPS,
    extra: Mapping[str, Proposal] | None = None,
    weights: Mapping[str, float] | None = None,
) -> ProposalMix:
    """GLASS's cocktail, for a run that samples exactly one binary.

    The returned Fisher component reports zero proposal correction. For
    production sampling, pass an oracle that returns a fixed decomposition or
    separately justify adaptation/correct the forward and reverse densities.
    The automatic :class:`FisherCache` wrapper counts Fisher proposals, not
    total mixture draws. Its refresh schedule is not a correctness guarantee.

    What a single source changes
    ----------------------------
    * **No reversible jump.** GLASS's ``rjweight`` set (prior 0.2, F-statistic
      draw 0.8) exists for birth and death moves. With a fixed source count
      there are none, so only the fixed-dimension MCMC weights apply.
    * **The F-statistic proposal is omitted.** It carries 0.4 of GLASS's MCMC
      weight and 0.8 of its RJ weight. Dropping it removes a global relocation
      move that ``prior`` partly covers. Add a search proposal through
      ``extra`` when needed. Every bundled proposal reports
      ``log q = 0``; for the Fisher component that is valid only under the
      fixed-decomposition or separately justified adaptive use described
      above.

    * **The SNR >= 5 cut is not applied.** See
      :func:`binary_galactica.glass.prior.snr_cut`:
      this search threshold changes the prior support and must be applied
      explicitly when intended.

    Weights follow GLASS's own rule: the named proposals keep their literal
    weights and ``fisher`` takes **the remainder**, because that is what
    ``proposal[4]->weight -= check`` does. Dropping the F-statistic therefore
    raises the Fisher share from 0.3 to 0.7.

    Parameters
    ----------
    prior
        A :class:`binary_galactica.glass.prior.UCBPrior`.
    fisher
        ``fisher(params) -> (evalues, evectors)``. Wrapped in a
        :class:`FisherCache` automatically unless it already is one. A
        state-dependent result makes the zero-correction Fisher move adaptive.
    T
        Observation time [s], for :func:`fm_shift`'s bin quantum.
    every
        Number of Fisher proposals between adaptive decomposition refreshes.
        This controls cadence, not validity of the sampling scheme.
    extra
        Additional ``{name: callable}`` proposals, e.g. an F-statistic draw.
        Their weights must be supplied via ``weights``.
    weights
        Override the literal weights. ``fisher`` is still derived as the
        remainder unless given explicitly.

    """
    if not isinstance(fisher, FisherCache):
        fisher = FisherCache(fisher, every=every)

    proposals = {
        "fisher": FisherProposal(fisher),
        "fm_shift": lambda p, rng: fm_shift(p, rng, T),
        "psi_phi": psi_phi_jump,
        "prior": PriorDraw(prior),
    }
    if extra:
        proposals.update(extra)

    w = dict(GLASS_MCMC_WEIGHTS)
    if weights:
        w.update(weights)
    missing = set(proposals) - set(w) - {"fisher"}
    if missing:
        raise ValueError(
            f"no weight given for {sorted(missing)}; pass weights={{...}} "
            "alongside extra proposals"
        )

    if "fisher" not in w:
        rest = sum(v for k, v in w.items() if k in proposals)
        if rest >= 1.0:
            raise ValueError(
                f"non-Fisher weights sum to {rest:.3f}; nothing left for the "
                "Fisher proposal, which GLASS defines as the remainder"
            )
        w["fisher"] = 1.0 - rest

    return ProposalMix(proposals, {k: w[k] for k in proposals})
