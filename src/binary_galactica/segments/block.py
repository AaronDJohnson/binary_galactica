"""Every segment of the band in one enchilada block; one Gibbs turn per call.

One :meth:`UCBBlock.sample` call:
1. checks that the residual and covariance are frequency-domain and on the
   construction grid;
2. takes the A and E coefficient variances ``C = (T/2) S`` from the
   covariance;
3. sweeps even segments, then odd. Each segment's sampler sees the conditional
   residual minus the other segments' current owned sources, on its padded
   window;
4. publishes the owned sources over their full band in every data channel.

The block implements enchilada's ``Block`` protocol structurally and never
imports enchilada. States are built with ``conditional_residual.block_state``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .catalogue import ldc_to_native, native_to_ldc
from .config import UCBBlockConfig
from .likelihood import SegmentLikelihood
from .render import Contribution, render_sources
from .runner import SegmentRunner
from .table import Segment, build_segment_table

#: Channel layouts the block accepts; A and E are always rows 0 and 1.
CHANNEL_SETS = (("A", "E"), ("A", "E", "T"))


def _names(rows: Any, n: int) -> list[str]:
    try:
        raw = np.asarray(rows["Name"]).reshape(-1)
    except (KeyError, ValueError, IndexError, TypeError):
        return [f"source_{i}" for i in range(n)]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


class UCBBlock:
    """GLASS-style UCB segments as one enchilada block (frequency domain).

    Parameters
    ----------
    grid
        An L1Data in any domain. It fixes ``num_time_samples``,
        ``sample_rate_hz``, ``start_time_gps`` and ``channel_names``, which must
        be ``("A", "E")`` or ``("A", "E", "T")``.
    start_sources
        LDC catalogue rows with the ``sky/dgb/cat`` fields; ``Name`` is optional.
    outdir
        Run directory for the configuration, segment table, dropped sources and
        chains.
    config
        :class:`UCBBlockConfig`; the default holds the spec's settings.
    reference_covariance
        Optional frequency ``DataCovariance`` defining the SNR cut for the run.
        ``None`` takes the first turn's covariance.
    name
        Registration name.

    Register with ``Wheel.add(..., data_domain="frequency")``.
    """

    def __init__(
        self,
        *,
        grid: Any,
        start_sources: Any,
        outdir: str | Path,
        config: UCBBlockConfig | None = None,
        reference_covariance: Any = None,
        name: str = "ucb",
    ) -> None:
        self.name = name
        self.config = config if config is not None else UCBBlockConfig()
        channels = tuple(grid.channel_names)
        if channels not in CHANNEL_SETS:
            raise ValueError(f"channels must be one of {CHANNEL_SETS}, got {channels}")
        self.channels = channels
        self.num_time_samples = int(grid.num_time_samples)
        self.sample_rate_hz = float(grid.sample_rate_hz)
        self.T = self.num_time_samples / self.sample_rate_hz
        self.t0 = float(grid.start_time_gps)
        self.n_freq = self.num_time_samples // 2 + 1
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.start_native = ldc_to_native(start_sources, self.T)
        self.start_names = _names(start_sources, self.start_native.shape[0])
        self.table = build_segment_table(self.start_native, T=self.T, n_freq=self.n_freq,
                                         config=self.config)
        self.config.write(self.outdir / "config.json")
        self.table.write(self.outdir / "segment_table.json")
        self.dropped: list[dict[str, Any]] = [
            self._record(i, None, "outside band") for i in self.table.outside
        ]
        self.owned_start = {seg.index: list(seg.owned) for seg in self.table.segments}
        self.c_ref: NDArray[np.float64] | None = None
        self.turn = 0
        self._runners: dict[int, SegmentRunner | None] | None = None
        self._sources: dict[int, NDArray[np.float64]] = {}
        self._contrib: dict[int, Contribution] = {}
        self._model: NDArray[np.complex128] | None = None
        if reference_covariance is not None:
            self._set_reference(self._diagonals(reference_covariance))
        self._write_dropped()

    # ---- inputs -----------------------------------------------------------

    def _record(self, i: int, snr: float | None, reason: str) -> dict[str, Any]:
        ldc = native_to_ldc(self.start_native[i], self.T)
        return {
            "index": int(i),
            "name": self.start_names[i],
            "reason": reason,
            "snr": snr,
            "native": self.start_native[i].tolist(),
            "ldc": {key: float(value[0]) for key, value in ldc.items()},
        }

    def _write_dropped(self) -> None:
        text = json.dumps(self.dropped, indent=1)
        (self.outdir / "dropped_start_sources.json").write_text(text + "\n")

    def _check_data(self, data: Any) -> None:
        if getattr(data, "data_domain", None) != "frequency":
            raise ValueError(
                "the UCB block needs frequency-domain data; register it with "
                "data_domain='frequency'"
            )
        if (
            tuple(data.channel_names) != self.channels
            or int(data.num_time_samples) != self.num_time_samples
            or float(data.sample_rate_hz) != self.sample_rate_hz
        ):
            raise ValueError("data grid or channels differ from the construction grid")

    def _diagonals(self, cov: Any) -> NDArray[np.float64]:
        """Return ``(2, n_freq)`` A/E coefficient variances; inactive bins are inf."""
        if getattr(cov, "data_domain", None) != "frequency" or not hasattr(
            cov, "covariance_matrix"
        ):
            raise TypeError("noise_covariance must be a frequency-domain DataCovariance")
        if tuple(cov.channel_names) != self.channels:
            raise ValueError("covariance channels differ from the data channels")
        m = np.asarray(cov.covariance_matrix)
        n = len(self.channels)
        if m.shape != (self.n_freq, n, n):
            raise ValueError(f"covariance must have shape {(self.n_freq, n, n)}")
        c = np.stack([m[:, 0, 0].real, m[:, 1, 1].real]).astype(np.float64)
        mask = getattr(cov, "active_mask", None)
        if mask is not None:
            c[:, ~np.asarray(mask, dtype=bool)] = np.inf
        lo, hi = self.table.segments[0].bin_lo, self.table.segments[-1].bin_hi
        scale = np.sqrt(np.abs(c[0, lo:hi] * c[1, lo:hi]))
        if np.any(np.abs(m[lo:hi, 0, 1]) > 1e-12 * scale):
            raise ValueError("the UCB block needs uncorrelated A and E noise")
        return c

    def _set_reference(self, c_ref: NDArray[np.float64]) -> None:
        """Fix the SNR-cut covariance and drop start sources below the cut."""
        self.c_ref = c_ref
        for seg in self.table.segments:
            own = self.owned_start[seg.index]
            if not own:
                continue
            lik = SegmentLikelihood.for_segment(seg, T=self.T, t0=self.t0, config=self.config)
            lik.set_reference(c_ref[:, seg.bin_lo : seg.bin_hi])
            keep = []
            for i, rho in zip(own, lik.snr(self.start_native[own])):
                if rho >= self.config.snr_min:
                    keep.append(i)
                else:
                    self.dropped.append(self._record(i, float(rho), "snr below cut"))
            self.owned_start[seg.index] = keep
        self._write_dropped()

    # ---- model ------------------------------------------------------------

    def _render(self, seg: Segment, sources: NDArray[np.float64]) -> Contribution:
        return render_sources(sources, T=self.T, t0=self.t0, BW=seg.bw,
                              n_channels=len(self.channels), n_freq=self.n_freq,
                              backend=self.config.backend,
                              response_samples=self.config.response_samples)

    def _start_sources(self) -> dict[int, NDArray[np.float64]]:
        return {
            seg.index: self.start_native[self.owned_start[seg.index]].reshape(-1, 8)
            for seg in self.table.segments
        }

    def _model_of(self, contrib: dict[int, Contribution]) -> NDArray[np.complex128]:
        model = np.zeros((len(self.channels), self.n_freq), dtype=np.complex128)
        for part in contrib.values():
            part.add_to(model)
        return model

    def _template(self, model: NDArray[np.complex128]) -> dict[str, NDArray[np.complex128]]:
        return {ch: model[i].copy() for i, ch in enumerate(self.channels)}

    def _parameters(self, sources: dict[int, NDArray[np.float64]]) -> dict[str, Any]:
        segments = []
        for seg in self.table.segments:
            x = sources[seg.index].reshape(-1, 8)
            segments.append({
                "index": seg.index,
                "lo": seg.lo,
                "hi": seg.hi,
                "count": int(x.shape[0]),
                "native": x.copy(),
                "ldc": native_to_ldc(x, self.T),
            })
        return {"turn": self.turn, "segments": segments}

    def initial_block_state(self, data: Any) -> Any:
        """Return the start set's published state on ``data``'s grid."""
        sources = self._start_sources()
        contrib = {seg.index: self._render(seg, sources[seg.index]) for seg in self.table.segments}
        return data.block_state(self._template(self._model_of(contrib)),
                                parameters=self._parameters(sources))

    def _start(self, rng: np.random.Generator) -> None:
        assert self.c_ref is not None
        self._sources = self._start_sources()
        self._runners = {}
        for seg in self.table.segments:
            start = self._sources[seg.index]
            if not self.config.birth_death and start.shape[0] == 0:
                self._runners[seg.index] = None
                continue
            self._runners[seg.index] = SegmentRunner(
                seg,
                start,
                T=self.T,
                t0=self.t0,
                config=self.config,
                outdir=self.outdir / f"segment_{seg.index:05d}",
                seed=int(rng.integers(2**32)),
                reference=self.c_ref[:, seg.bin_lo : seg.bin_hi],
            )
        self._contrib = {seg.index: self._render(seg, self._sources[seg.index])
                         for seg in self.table.segments}
        self._model = self._model_of(self._contrib)

    # ---- Block protocol ---------------------------------------------------

    def sample(
        self,
        conditional_residual: Any,
        noise_covariance: Any,
        current_block_state: Any,
        *,
        rng: np.random.Generator,
    ) -> Any:
        """Run one Gibbs turn and return the published block state."""
        data = conditional_residual
        self._check_data(data)
        c = self._diagonals(noise_covariance)
        if self.c_ref is None:
            self._set_reference(c)
        if self._runners is None:
            self._start(rng)
        assert self._runners is not None and self._model is not None
        r = np.stack([np.asarray(data.channel_data[ch], dtype=np.complex128) for ch in ("A", "E")])
        for parity in (0, 1):
            for seg in self.table.segments[parity::2]:
                runner = self._runners[seg.index]
                if runner is None:
                    continue
                lo, hi = seg.bin_lo, seg.bin_hi
                own = self._contrib[seg.index].window(lo, hi)[:2]
                runner.advance(r[:, lo:hi] - (self._model[:2, lo:hi] - own), c[:, lo:hi])
                cold = runner.cold_sources()
                keep = cold[(cold[:, 0] >= seg.lo) & (cold[:, 0] < seg.hi)]
                new = self._render(seg, keep)
                self._contrib[seg.index].add_to(self._model, -1.0)
                new.add_to(self._model, 1.0)
                self._contrib[seg.index] = new
                self._sources[seg.index] = keep
        self._model = self._model_of(self._contrib)
        self.turn += 1
        return data.block_state(self._template(self._model),
                                parameters=self._parameters(self._sources))


__all__ = ["CHANNEL_SETS", "UCBBlock"]
