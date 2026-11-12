"""Every segment of the band in one enchilada block; one Gibbs turn per call.

One :meth:`UCBBlock.sample` call:
1. checks that the residual is frequency-domain and on the construction grid
   (channels, length, sample rate and start time), that the covariance is an
   enchilada ``Covariance`` with the data's channels, and that the ledger's
   turn matches the block's;
2. takes the A and E coefficient variances ``C = (T/2) S`` from
   ``covariance.pair``;
3. sweeps even segments, then odd. Each segment's sampler sees the conditional
   residual minus the other segments' current owned sources, on its padded
   window. Every job of a parity is built from the model as it stood when the
   parity started, so a parity can run in worker processes (``workers``);
4. publishes the owned sources over their full band in every data channel.

The block implements enchilada's ``Block`` protocol structurally and never
imports enchilada. States are built with ``conditional_residual.block_state``.
"""

from __future__ import annotations

import json
import weakref
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .catalogue import ldc_to_native, native_to_ldc
from .config import UCBBlockConfig
from .executor import ProcessExecutor, RunnerSpec, SegmentExecutor, SerialExecutor
from .likelihood import SegmentLikelihood
from .render import Contribution, render_sources
from .runner import ChainArchive
from .table import Segment, build_segment_table

#: Channel layouts the block accepts; A and E are always rows 0 and 1.
CHANNEL_SETS = (("A", "E"), ("A", "E", "T"))


def _make_executor(workers: int) -> SegmentExecutor:
    """Return the sweep's executor: in-process for one worker, else processes."""
    return SerialExecutor() if workers == 1 else ProcessExecutor(workers)


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
        be ``("A", "E")`` or ``("A", "E", "T")``. ``start_time_gps`` is used as
        the waveform and orbit ``t0``, so it must be in the catalogue's time
        frame (LDC t=0 for Sangria).
    start_sources
        LDC catalogue rows with the ``sky/dgb/cat`` fields; ``Name`` is optional.
    outdir
        Run directory for the configuration, segment table, dropped sources and
        chains. It may be missing or empty, but must not already hold a run
        (``config.json`` or ``segment_*/cold/turn_*.npz``), else ``ValueError``.
    config
        :class:`UCBBlockConfig`; the default holds the spec's settings.
    reference_covariance
        enchilada ``Covariance`` on the data's frequency grid defining the SNR
        cut for the run; start sources below the cut are dropped at
        construction. It is required by :meth:`initial_block_state`, so that
        the published turn-0 template never includes a source the block then
        drops. ``None`` defers the cut to the first turn's covariance, for
        callers that seed the ledger another way.
    name
        Registration name.
    workers
        Processes for the sweep. ``1`` (the default) runs every segment in the
        main process. More starts up to that many ``spawn`` worker processes at
        the first turn (one per non-empty shard), each owning a fixed shard of
        segments for the whole run
        (:class:`~binary_galactica.segments.executor.ProcessExecutor`). The
        chains do not depend on ``workers``. Stop the workers with
        :meth:`close`, or use the block as a context manager; a script that
        uses workers needs an ``if __name__ == "__main__":`` guard.

    The block holds its own sampler state (every segment's chains) and does
    not restart from ``current_block_state``. It only checks that the ledger's
    ``parameters["turn"]``, when present, equals :attr:`turn`, so a ledger out
    of step with the block raises.

    A turn is not atomic. If an exception escapes :meth:`sample` after its
    inputs are validated, some segments may already have advanced and
    archived the turn while :attr:`turn` has not moved. This covers applying
    the cut, starting the runners and the sweep itself. A retry would advance
    those segments twice, so every later call raises ``RuntimeError``; build a
    fresh ``UCBBlock`` on a fresh ``outdir``: a failed turn leaves archived
    chains behind, and the constructor refuses an ``outdir`` that holds
    ``config.json`` or any archived turn, so a run is never mislabelled.
    Rejected inputs (a wrong domain, grid or covariance, or a ledger turn out
    of step) raise before anything changes and leave the block usable.

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
        workers: int = 1,
    ) -> None:
        """Validate the inputs, build the segment table and write the run files."""
        if isinstance(workers, bool) or int(workers) != workers or workers < 1:
            raise ValueError(f"workers must be a positive integer, got {workers!r}")
        self.workers = int(workers)
        self.outdir = Path(outdir)
        self._check_outdir_unused()
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
        self.start_native = ldc_to_native(start_sources, self.T)
        self.start_names = _names(start_sources, self.start_native.shape[0])
        self.table = build_segment_table(
            self.start_native, T=self.T, n_freq=self.n_freq, config=self.config
        )
        # Validate the reference before anything is written, so a rejected one
        # leaves no half-written run directory behind.
        c_ref = (
            None
            if reference_covariance is None
            else self._diagonals(reference_covariance)
        )
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.config.write(self.outdir / "config.json")
        self.table.write(self.outdir / "segment_table.json")
        self.dropped: list[dict[str, Any]] = [
            self._record(i, None, "outside band") for i in self.table.outside
        ]
        self.owned_start = {seg.index: list(seg.owned) for seg in self.table.segments}
        self.c_ref: NDArray[np.float64] | None = None
        self.turn = 0
        self._active: set[int] | None = None
        self._executor: SegmentExecutor | None = None
        self._finalizer: weakref.finalize | None = None
        self._closed = False
        self._sources: dict[int, NDArray[np.float64]] = {}
        self._contrib: dict[int, Contribution] = {}
        self._model: NDArray[np.complex128] | None = None
        self._failed = False
        if c_ref is not None:
            self._set_reference(c_ref)
        self._write_dropped()

    # ---- inputs -----------------------------------------------------------

    def _check_outdir_unused(self) -> None:
        """Refuse an outdir that already holds a run, before any file is written."""
        if (self.outdir / "config.json").exists() or any(
            self.outdir.glob("segment_*/cold/turn_*.npz")
        ):
            raise ValueError(
                f"{self.outdir} already holds a run (config.json or archived turns); "
                "use a new directory"
            )

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
            or float(data.start_time_gps) != self.t0
        ):
            raise ValueError(
                "data grid, start time or channels differ from the construction grid"
            )

    def _diagonals(self, cov: Any) -> NDArray[np.float64]:
        """Return ``(2, n_freq)`` A/E coefficient variances from a ``Covariance``.

        Only ``channel_names``, ``pair`` and an optional ``form`` are read, so
        any enchilada ``Covariance`` from 0.3.0 on works. Variances are checked
        inside the table's span only; the segment windows never read DC.
        """
        if cov is None or not callable(getattr(cov, "pair", None)):
            raise TypeError(
                "covariance must be an enchilada Covariance (0.3.0 or later); "
                "register the block with data_domain='frequency' and give the "
                "Wheel an initial_covariance"
            )
        if getattr(cov, "form", "diagonal") != "diagonal":
            raise ValueError("the UCB block needs a diagonal-form Covariance")
        if tuple(cov.channel_names) != self.channels:
            raise ValueError("covariance channels differ from the data channels")
        diag = [np.asarray(cov.pair(ch, ch)) for ch in ("A", "E")]
        if any(d.shape != (self.n_freq,) for d in diag):
            raise ValueError(f"covariance arrays must have shape {(self.n_freq,)}")
        c = np.stack([d.real for d in diag]).astype(np.float64)
        lo, hi = self.table.segments[0].bin_lo, self.table.segments[-1].bin_hi
        band = c[:, lo:hi]
        if not np.all(np.isfinite(band) & (band > 0)):
            raise ValueError(
                "covariance variances must be finite and positive in the block's band"
            )
        cross = np.asarray(cov.pair("A", "E"))[lo:hi]
        if np.any(np.abs(cross) > 1e-12 * np.sqrt(band[0] * band[1])):
            raise ValueError("the UCB block needs uncorrelated A and E noise")
        return c

    def _set_reference(self, c_ref: NDArray[np.float64]) -> None:
        """Fix the SNR-cut covariance and drop start sources below the cut."""
        self.c_ref = c_ref
        for seg in self.table.segments:
            own = self.owned_start[seg.index]
            if not own:
                continue
            lik = SegmentLikelihood.for_segment(
                seg, T=self.T, t0=self.t0, config=self.config
            )
            lik.set_reference(c_ref[:, seg.bin_lo : seg.bin_hi])
            keep = []
            for i, rho in zip(own, lik.snr(self.start_native[own]), strict=True):
                if rho >= self.config.snr_min:
                    keep.append(i)
                else:
                    self.dropped.append(self._record(i, float(rho), "snr below cut"))
            self.owned_start[seg.index] = keep
        self._write_dropped()

    # ---- model ------------------------------------------------------------

    def _render(self, seg: Segment, sources: NDArray[np.float64]) -> Contribution:
        return render_sources(
            sources,
            T=self.T,
            t0=self.t0,
            BW=seg.bw,
            n_channels=len(self.channels),
            n_freq=self.n_freq,
            backend=self.config.backend,
            response_samples=self.config.response_samples,
        )

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

    def _template(
        self, model: NDArray[np.complex128]
    ) -> dict[str, NDArray[np.complex128]]:
        return {ch: model[i].copy() for i, ch in enumerate(self.channels)}

    def _parameters(self, sources: dict[int, NDArray[np.float64]]) -> dict[str, Any]:
        segments = []
        for seg in self.table.segments:
            x = sources[seg.index].reshape(-1, 8)
            segments.append(
                {
                    "index": seg.index,
                    "lo": seg.lo,
                    "hi": seg.hi,
                    "count": int(x.shape[0]),
                    "native": x.copy(),
                    "ldc": native_to_ldc(x, self.T),
                }
            )
        return {"turn": self.turn, "segments": segments}

    def initial_block_state(self, data: Any) -> Any:
        """Return the start set's published state on ``data``'s grid.

        It renders the start sources that survive the SNR cut and publishes
        ``parameters["turn"] == 0`` (the current :attr:`turn` before any
        sample). It raises ``ValueError`` if the block was built without
        ``reference_covariance``, because the cut is not yet applied.
        """
        self._check_data(data)
        if self.c_ref is None:
            raise ValueError(
                "initial_block_state needs the SNR cut applied first; build the "
                "UCBBlock with reference_covariance= (e.g. the initial noise "
                "covariance)"
            )
        sources = self._start_sources()
        contrib = {
            seg.index: self._render(seg, sources[seg.index])
            for seg in self.table.segments
        }
        return data.block_state(
            self._template(self._model_of(contrib)),
            parameters=self._parameters(sources),
        )

    def _start(self, rng: np.random.Generator) -> None:
        assert self.c_ref is not None
        self._sources = self._start_sources()
        specs = []
        for seg in self.table.segments:
            start = self._sources[seg.index]
            if not self.config.birth_death and start.shape[0] == 0:
                continue
            specs.append(
                RunnerSpec(
                    segment=seg,
                    start=start,
                    T=self.T,
                    t0=self.t0,
                    config=self.config,
                    outdir=self.outdir / f"segment_{seg.index:05d}",
                    seed=int(rng.integers(2**32)),
                    reference=self.c_ref[:, seg.bin_lo : seg.bin_hi].copy(),
                    n_channels=len(self.channels),
                    n_freq=self.n_freq,
                )
            )
        executor = _make_executor(self.workers)
        self._executor = executor
        self._finalizer = weakref.finalize(self, executor.close)
        executor.start(specs)
        self._active = {spec.segment.index for spec in specs}
        self._contrib = {
            seg.index: self._render(seg, self._sources[seg.index])
            for seg in self.table.segments
        }
        self._model = self._model_of(self._contrib)

    def cold_chain(self, index: int) -> dict[str, NDArray[Any]] | None:
        """Return segment ``index``'s archived cold chain.

        This is :meth:`ChainArchive.read` of the segment's archive directory, so
        it works whichever process runs the segment, and after :meth:`close`.

        ``None`` when the segment has no runner (fixed count with no start sources).
        Raises ``RuntimeError`` before the first ``sample()`` call and ``KeyError`` for
        an unknown index.
        """
        if self._active is None:
            raise RuntimeError("no chains yet: call sample() first")
        if not 0 <= index < len(self.table.segments):
            raise KeyError(index)
        if index not in self._active:
            return None
        return ChainArchive.read(self.outdir / f"segment_{index:05d}" / "cold")

    def peak_rss_bytes(self) -> list[int]:
        """Return the peak resident memory of each process running segments.

        One entry per worker process, or the main process's with one worker.
        Raises ``RuntimeError`` before the first ``sample()`` call.
        """
        if self._executor is None:
            raise RuntimeError("no executor yet: call sample() first")
        return self._executor.peak_rss_bytes()

    def close(self) -> None:
        """Stop the sweep's worker processes; later :meth:`sample` calls raise.

        Archived chains stay readable through :meth:`cold_chain`. Closing twice,
        or before the first turn, is harmless.
        """
        self._closed = True
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self) -> UCBBlock:
        """Return the block; leaving the ``with`` block calls :meth:`close`."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the block."""
        self.close()

    # ---- Block protocol ---------------------------------------------------

    def sample(
        self,
        conditional_residual: Any,
        covariance: Any,
        current_block_state: Any,
        *,
        rng: np.random.Generator,
    ) -> Any:
        """Run one Gibbs turn and return the published block state.

        Every job of a parity is built from the model as it stood when the
        parity started; the parity's updates are applied in index order once
        it finishes. The segment table guarantees that no same-parity update
        reaches another same-parity window
        (:func:`~binary_galactica.segments.table.same_parity_disjoint`), so
        this equals advancing the segments one by one in index order, whether
        they run in the main process or in ``workers`` processes.

        ``current_block_state`` is not used to restart the sampler. Its
        ``parameters["turn"]``, if present, must equal :attr:`turn`. After a
        failed turn every call raises ``RuntimeError`` (see the class notes).
        """
        if self._failed:
            raise RuntimeError(
                "a previous sample() failed mid-sweep; build a fresh UCBBlock "
                "(with a new outdir)"
            )
        if self._closed:
            raise RuntimeError("the block is closed; build a fresh UCBBlock")
        params = getattr(current_block_state, "parameters", None)
        ledger_turn = params.get("turn") if isinstance(params, dict) else None
        if ledger_turn is not None and ledger_turn != self.turn:
            raise ValueError(
                f"current_block_state is from turn {ledger_turn} but the block is at "
                f"turn {self.turn}; the ledger and the block's sampler state are "
                "out of step"
            )
        data = conditional_residual
        self._check_data(data)
        c = self._diagonals(covariance)
        r = np.stack(
            [
                np.asarray(data.channel_data[ch], dtype=np.complex128)
                for ch in ("A", "E")
            ]
        )
        # Everything above only reads its inputs. From here on segments change
        # state one by one, so a failure leaves the block unusable.
        try:
            model = self._turn(r, c, rng)
        except BaseException:
            self._failed = True
            if self._finalizer is not None:
                self._finalizer()  # stop the workers; the block cannot continue
            raise
        return data.block_state(
            self._template(model), parameters=self._parameters(self._sources)
        )

    def _turn(
        self,
        r: NDArray[np.complex128],
        c: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.complex128]:
        """Apply the cut, start the runners if needed, sweep, and return the model."""
        if self.c_ref is None:
            self._set_reference(c)
        if self._active is None:
            self._start(rng)
        assert self._executor is not None and self._active is not None
        assert self._model is not None
        for parity in (0, 1):
            jobs = {}
            for seg in self.table.segments[parity::2]:
                if seg.index not in self._active:
                    continue
                lo, hi = seg.bin_lo, seg.bin_hi
                own = self._contrib[seg.index].window(lo, hi)[:2]
                jobs[seg.index] = (
                    r[:, lo:hi] - (self._model[:2, lo:hi] - own),
                    c[:, lo:hi],
                )
            results = self._executor.advance(jobs)
            for index in sorted(results):
                result = results[index]
                self._contrib[index].add_to(self._model, -1.0)
                result.contribution.add_to(self._model, 1.0)
                self._contrib[index] = result.contribution
                self._sources[index] = result.owned
        model = self._model_of(self._contrib)
        self._model = model
        self.turn += 1
        return model


__all__ = ["CHANNEL_SETS", "UCBBlock"]
