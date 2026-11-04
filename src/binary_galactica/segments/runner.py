"""One segment's reversible-jump sampler, continued across Gibbs turns.

impulse has no call that continues a run after the likelihood changes. Each
turn therefore calls ``RJPTSampler.sample`` again on the *same* sampler,
starting from the previous turn's final state. These survive across turns:
- the temperature ladder and its adaptation (``ptstate``, so the hottest chain
  stays fixed and the intermediate ones keep adapting);
- the RNG streams;
- the proposal bundle and the per-chain statistics.

``sample`` recomputes every chain's likelihood for the new residual and PSD
and raises if any value is not finite. impulse restarts its iteration count
and truncates its chain files on every call, so the runner copies the cold
chain into a :class:`ChainArchive` after each turn and deletes impulse's
files. ``num_adapt`` stays ``None``, because it would count per turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import UCBBlockConfig
from .fisher import FisherBank, fisher_decomposition
from .likelihood import SegmentLikelihood
from .moves import FisherMove, FmShiftMove, PriorRedrawMove, PsiPhiMove
from .prior import SegmentPrior
from .table import Segment

#: Periodic coordinates of each source and their intervals (UCBPrior's).
COMPONENT_PERIODIC = {2: (0.0, 2.0 * np.pi), 5: (0.0, np.pi), 6: (0.0, 2.0 * np.pi)}


def _impulse() -> Any:
    try:
        import impulse
    except ImportError as exc:  # pragma: no cover - exercised without impulse
        raise ImportError(
            "the segment block needs impulse-mcmc with RJPTSampler, which currently "
            "needs impulse's rjmcmc branch "
            "(pip install 'binary-galactica[segments]')"
        ) from exc
    if not hasattr(impulse, "RJPTSampler"):
        raise ImportError("impulse lacks RJPTSampler; install its rjmcmc branch")
    return impulse


def glass_ladder(
    ntemps: int, step: float = 1.2, hottest: float = 1.0e12
) -> NDArray[np.float64]:
    """Return GLASS's ladder ``step**i`` with the hottest chain at ``hottest``."""
    ladder = step ** np.arange(int(ntemps), dtype=np.float64)
    if ntemps > 1:
        ladder[-1] = hottest
    return ladder


class ChainArchive:
    """Cold-chain samples of one segment, one compressed file per turn.

    The directory must not already hold archived turns, so that one archive
    never mixes two runs; existing turns raise ``ValueError``.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.files():
            raise ValueError(
                f"{self.directory} already holds archived turns; use a new directory"
            )

    def append(
        self, turn: int, *, samples: ArrayLike, k: ArrayLike, lnlike: ArrayLike
    ) -> None:
        """Write one turn's cold samples ``(n, D)``, counts and likelihoods."""
        np.savez_compressed(
            self.directory / f"turn_{int(turn):05d}.npz",
            samples=np.asarray(samples, dtype=np.float64),
            k=np.asarray(k, dtype=np.int64),
            lnlike=np.asarray(lnlike, dtype=np.float64),
        )

    def files(self) -> list[Path]:
        """Return the per-turn files in turn order."""
        return sorted(self.directory.glob("turn_*.npz"))

    def load(self) -> dict[str, NDArray[Any]]:
        """Concatenate every turn; ``turn`` gives each row's turn."""
        parts, turns = [], []
        for p in self.files():
            # Close each file after reading: an open NpzFile per turn would
            # exhaust the descriptor limit on long runs.
            with np.load(p) as z:
                parts.append({name: z[name] for name in ("samples", "k", "lnlike")})
            turns.append(int(p.stem.split("_")[1]))
        if not parts:
            raise FileNotFoundError(f"no archived turns in {self.directory}")
        return {
            "samples": np.concatenate([z["samples"] for z in parts]),
            "k": np.concatenate([z["k"] for z in parts]),
            "lnlike": np.concatenate([z["lnlike"] for z in parts]),
            "turn": np.concatenate(
                [np.full(z["k"].shape, t, dtype=np.int64) for z, t in zip(parts, turns)]
            ),
        }


class SegmentRunner:
    """Sampler, likelihood and Fisher bank of one segment.

    ``start`` holds the segment's ``(n, 8)`` owned start sources; every chain
    starts there. With ``config.birth_death`` off, the count is fixed at
    ``n``. Otherwise it ranges over ``0 .. segment.k_max``. ``reference`` is
    the run-fixed ``(2, n_bins)`` covariance of the SNR cut. ``prior_only``
    replaces the likelihood with zero (no data, no cut, no Fisher matrices).
    """

    def __init__(
        self,
        segment: Segment,
        start: ArrayLike,
        *,
        T: float,
        t0: float,
        config: UCBBlockConfig,
        outdir: str | Path,
        seed: int,
        reference: ArrayLike | None = None,
        prior_only: bool = False,
    ) -> None:
        start = np.asarray(start, dtype=np.float64).reshape(-1, 8)
        self.segment = segment
        self.config = config
        self.T = float(T)
        self.fixed_count = not config.birth_death
        if self.fixed_count and start.shape[0] == 0:
            raise ValueError("a fixed-count segment needs at least one start source")
        impulse = _impulse()
        self.k_max = start.shape[0] if self.fixed_count else segment.k_max
        if start.shape[0] > self.k_max:
            raise ValueError(
                f"{start.shape[0]} start sources exceed k_max={self.k_max}"
            )
        k_min = self.k_max if self.fixed_count else 0
        self.prior = segment.prior(T)
        self.log_prior = SegmentPrior(self.prior, self.k_max, k_min)
        self.likelihood: SegmentLikelihood | None = None
        if not prior_only:
            if reference is None:
                raise ValueError(
                    "a runner with a likelihood needs the reference covariance"
                )
            self.likelihood = SegmentLikelihood.for_segment(
                segment, T=T, t0=t0, config=config, k_max=self.k_max
            )
            self.likelihood.set_reference(reference)
        self.bank = FisherBank(config.ntemps, config.fisher_match_bins)
        self.outdir = Path(outdir)
        self.archive = ChainArchive(self.outdir / "cold")
        self.turn = 0
        adaptive = (config.am_weight + config.scam_weight + config.de_weight) > 0
        self.sampler = impulse.RJPTSampler(
            n_shared=0,
            component_dim=8,
            max_components=self.k_max,
            min_components=k_min,
            lnlike=self._lnlike,
            lnprior=self.log_prior,
            component_proposal=impulse.IndependentComponentProposal(
                draw=self._draw, logpdf=self.prior.log_prior
            ),
            birth_death_weight=config.birth_death_weight(),
            component_redraw_weight=0.0,
            component_periodic=COMPONENT_PERIODIC,
            vectorized=True,
            jax=True,
            ntemps=config.ntemps,
            ladder=glass_ladder(
                config.ntemps, config.ladder_step, config.hottest_temperature
            ),
            am_weight=config.am_weight,
            scam_weight=config.scam_weight,
            de_weight=config.de_weight,
            buffer_size=2_000 if adaptive else 16,
            cov_update=100 if adaptive else config.steps_per_turn,
            save_freq=config.steps_per_turn,
            seed=int(seed),
            outdir=str(self.outdir / "impulse"),
            verbose=False,
        )
        moves = {
            "fisher": FisherMove(self.bank),
            "prior": PriorRedrawMove(self.prior),
            "fm_shift": FmShiftMove(self.T),
            "psi_phi": PsiPhiMove(),
        }
        for name, weight in config.within_model_weights().items():
            if weight > 0:
                self.sampler.add_custom_jump(moves[name], weight)
        x0 = np.full(8 * self.k_max, np.nan)
        x0[: start.size] = start.reshape(-1)
        self._x = np.tile(x0, (config.ntemps, 1))
        self._k = np.full(config.ntemps, start.shape[0], dtype=np.int64)

    def _draw(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return self.prior.sample(rng)

    def _lnlike(
        self, x: NDArray[np.float64], k: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        if self.likelihood is None:
            return np.zeros(len(x))
        return self.likelihood(x, k)

    @property
    def positions(self) -> NDArray[np.float64]:
        """Return every chain's padded parameters ``(ntemps, 8 * k_max)``."""
        return self._x.copy()

    @property
    def ks(self) -> NDArray[np.int64]:
        """Return every chain's source count."""
        return self._k.copy()

    def cold_sources(self) -> NDArray[np.float64]:
        """Return the cold chain's active sources ``(k, 8)``."""
        k = int(self._k[0])
        return self._x[0, : 8 * k].reshape(k, 8).copy()

    def refresh_fisher(self) -> None:
        """Recompute every chain's Fisher decompositions at its current sources.

        The bank stays fixed for the whole turn, and :class:`FisherMove`
        declines any jump that would change a source's bank cell, so within a
        turn the Fisher jump is exactly symmetric. Rebuilding the bank from
        the current state between turns is an approximation: it is
        non-diminishing adaptation of the proposal, as in GLASS.
        """
        self.bank.clear()
        if self.likelihood is None:
            return
        rows = [self._x[c, : 8 * int(k)].reshape(-1, 8) for c, k in enumerate(self._k)]
        owners = np.concatenate([np.full(len(r), c) for c, r in enumerate(rows)])
        if owners.size == 0:
            return
        stack = np.concatenate(rows)
        matrices = self.likelihood.fisher_matrices(stack)
        widths = self.prior.hi - self.prior.lo
        for c in range(len(rows)):
            qs, evalues, evectors = [], [], []
            for src, F in zip(stack[owners == c], matrices[owners == c]):
                dec = fisher_decomposition(F, widths)
                if dec is not None:
                    qs.append(src[0])
                    evalues.append(dec[0])
                    evectors.append(dec[1])
            self.bank.store(
                c,
                np.array(qs),
                np.array(evalues).reshape(-1, 8),
                np.array(evectors).reshape(-1, 8, 8),
            )

    def advance(
        self, residual: ArrayLike | None = None, c: ArrayLike | None = None
    ) -> None:
        """Run one Gibbs turn on this turn's ``(2, n_bins)`` residual and variance."""
        if self.likelihood is not None:
            if residual is None or c is None:
                raise ValueError("advance needs the residual and its covariance")
            self.likelihood.set_data(residual, c)
        self.refresh_fisher()
        self.sampler.sample(
            self._x,
            self.config.steps_per_turn,
            thin=self.config.thin,
            initial_k=self._k,
        )
        width = 8 * self.k_max
        state = np.asarray(self.sampler.state.positions)
        self._x = np.array(state[:, :width])
        self._k = state[:, width].astype(np.int64)
        chain = self.sampler.load_chain()
        self.archive.append(
            self.turn,
            samples=chain["samples"][0],
            k=chain["k"][0],
            lnlike=chain["lnlike"][0],
        )
        for path in (self.outdir / "impulse").glob("chain_*"):
            path.unlink()
        # Free impulse's (ntemps, steps, D) ring buffer; every sample() call rebuilds it.
        self.sampler.short_chain = None
        self.turn += 1


__all__ = ["COMPONENT_PERIODIC", "ChainArchive", "SegmentRunner", "glass_ladder"]
