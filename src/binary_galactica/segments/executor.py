"""Run segment samplers in the main process or in worker processes.

The block builds every job of a parity from the model as it stood when the
parity started, and the segment table guarantees that no same-parity update
touches another same-parity window (:func:`~.table.same_parity_disjoint`). An
executor therefore only has to run each job; the order does not change the
result.

- :class:`SerialExecutor` holds the runners in the main process and runs the
  jobs in index order.
- :class:`ProcessExecutor` gives each worker process a fixed shard of segments
  for the whole run (:func:`assign_shards`), so samplers never cross a process
  boundary. Workers use the ``spawn`` start method, because forking a process
  that has initialised JAX is unsafe, and they inherit the parent's
  environment (``XLA_FLAGS`` included) and its JAX x64 setting.

Other JAX settings changed with ``jax.config.update`` in the parent do not
reach the workers; set them through environment variables, which the
workers inherit. Workers ignore SIGINT: an interrupt reaches the parent,
which then terminates them.

If any worker fails, the executor terminates every worker at once and
refuses further work; build a new one.

A script that uses :class:`ProcessExecutor` must guard its entry point with
``if __name__ == "__main__":``, as ``spawn`` re-imports the main module.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import resource
import signal
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .config import UCBBlockConfig
from .render import Contribution, render_sources
from .runner import SegmentRunner
from .table import Segment

#: One segment's job: its ``(2, n_bins)`` residual window and A/E variances.
Job = tuple[NDArray[np.complex128], NDArray[np.float64]]

#: A step's cost from the dry run's fit at commit 6dc3c4d
#: (``tools/segments/dry_run.py``): a fixed part and parts per template bin,
#: per slot-bin and per window bin, in microseconds. Fits vary between runs,
#: the per-bin term most, so :func:`assign_shards` uses these only as
#: relative weights for load balance.
STEP_COST_US = (1252.0, 3.33, 0.036, 0.022)


@dataclass(frozen=True)
class RunnerSpec:
    """Everything needed to build one segment's runner; picklable."""

    segment: Segment
    start: NDArray[np.float64]
    T: float
    t0: float
    config: UCBBlockConfig
    outdir: Path
    seed: int
    reference: NDArray[np.float64]
    n_channels: int
    n_freq: int


@dataclass(frozen=True)
class AdvanceResult:
    """One segment's turn: the cold sources it owns and their rendered template."""

    owned: NDArray[np.float64]
    contribution: Contribution


class SegmentExecutor(Protocol):
    """Runs the segments' samplers; see the module notes."""

    def start(self, specs: Sequence[RunnerSpec]) -> None:
        """Build one runner per spec."""

    def advance(self, jobs: Mapping[int, Job]) -> dict[int, AdvanceResult]:
        """Advance each job's segment by one turn and return the results."""

    def peak_rss_bytes(self) -> list[int]:
        """Return the peak resident memory of each process running segments."""

    def close(self) -> None:
        """Release the runners (and any worker processes)."""


RunnerFactory = Callable[[RunnerSpec], Any]


def build_runner(spec: RunnerSpec) -> SegmentRunner:
    """Return the :class:`SegmentRunner` a spec describes."""
    return SegmentRunner(
        spec.segment,
        spec.start,
        T=spec.T,
        t0=spec.t0,
        config=spec.config,
        outdir=spec.outdir,
        seed=spec.seed,
        reference=spec.reference,
    )


def advance_runner(runner: Any, spec: RunnerSpec, job: Job) -> AdvanceResult:
    """Run one turn and render the cold sources the segment owns."""
    residual, c = job
    runner.advance(residual, c)
    cold = runner.cold_sources()
    owned = cold[spec.segment.owns(cold[:, 0])]
    contribution = render_sources(
        owned,
        T=spec.T,
        t0=spec.t0,
        BW=spec.segment.bw,
        n_channels=spec.n_channels,
        n_freq=spec.n_freq,
        backend=spec.config.backend,
        response_samples=spec.config.response_samples,
    )
    return AdvanceResult(owned, contribution)


def segment_cost(segment: Segment) -> float:
    """Return the relative cost of one step of ``segment`` (see ``STEP_COST_US``)."""
    fixed, per_bw, per_slot_bw, per_bin = STEP_COST_US
    return (
        fixed
        + per_bw * segment.bw
        + per_slot_bw * segment.k_max * segment.bw
        + per_bin * segment.n_bins
    )


def assign_shards(segments: Sequence[Segment], workers: int) -> list[list[int]]:
    """Split segment indices among ``workers``, balancing each parity separately.

    Within each parity the segments go, most expensive first, to the worker
    with the least load of that parity so far (ties to the lower worker). Each
    shard is returned in index order; a shard may be empty.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    shards: list[list[int]] = [[] for _ in range(workers)]
    for parity in (0, 1):
        load = [0.0] * workers
        group = sorted(
            (s for s in segments if s.index % 2 == parity),
            key=lambda s: (-segment_cost(s), s.index),
        )
        for seg in group:
            w = min(range(workers), key=lambda k: (load[k], k))
            shards[w].append(seg.index)
            load[w] += segment_cost(seg)
    return [sorted(shard) for shard in shards]


def peak_rss_bytes() -> int:
    """Return this process's peak resident memory in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class SerialExecutor:
    """Runners in the main process; jobs run in index order."""

    def __init__(self, runner_factory: RunnerFactory = build_runner) -> None:
        """Use ``runner_factory`` to build each runner (``build_runner`` by default)."""
        self._factory = runner_factory
        self._runners: dict[int, tuple[Any, RunnerSpec]] = {}

    def start(self, specs: Sequence[RunnerSpec]) -> None:
        """Build one runner per spec, in order."""
        for spec in specs:
            self._runners[spec.segment.index] = (self._factory(spec), spec)

    def advance(self, jobs: Mapping[int, Job]) -> dict[int, AdvanceResult]:
        """Advance the jobs' segments in index order; exceptions propagate unchanged."""
        results = {}
        for index in sorted(jobs):
            runner, spec = self._runners[index]
            results[index] = advance_runner(runner, spec, jobs[index])
        return results

    def peak_rss_bytes(self) -> list[int]:
        """Return the main process's peak resident memory, as a one-item list."""
        return [peak_rss_bytes()]

    def close(self) -> None:
        """Drop the runners."""
        self._runners.clear()


def _worker(
    conn: Any, factory: RunnerFactory, specs: list[RunnerSpec], x64: bool
) -> None:
    """Build a shard's runners, then serve jobs until told to stop."""
    # A terminal Ctrl-C reaches the whole process group; the parent handles it
    # and stops the workers (see ProcessExecutor.advance).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import jax

    jax.config.update("jax_enable_x64", x64)
    try:
        runners = {spec.segment.index: (factory(spec), spec) for spec in specs}
    except BaseException:
        conn.send(("error", None, traceback.format_exc()))
        return
    conn.send(("ok", None, None))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        if message == "rss":
            conn.send(("ok", peak_rss_bytes(), None))
            continue
        results = {}
        for index in sorted(message):
            runner, spec = runners[index]
            try:
                results[index] = advance_runner(runner, spec, message[index])
            except BaseException:
                # The shard's state is now inconsistent; report and stop.
                conn.send(("error", index, traceback.format_exc()))
                return
        conn.send(("ok", results, None))


class ProcessExecutor:
    """Persistent worker processes, each owning a fixed shard of segments."""

    def __init__(
        self, workers: int, *, runner_factory: RunnerFactory = build_runner
    ) -> None:
        """Start nothing yet; :meth:`start` launches up to ``workers`` processes.

        ``runner_factory`` must be picklable (a module-level function), because
        each worker receives it.
        """
        if int(workers) < 1:
            raise ValueError("workers must be at least 1")
        self.workers = int(workers)
        self._factory = runner_factory
        self._context = mp.get_context("spawn")
        self._procs: list[Any] = []
        self._conns: list[Any] = []
        self._shards: list[list[int]] = []
        self._shard_of: dict[int, int] = {}
        self._closed = False
        self._failed = False

    @property
    def shards(self) -> list[list[int]]:
        """Return each running worker's segment indices."""
        return [list(shard) for shard in self._shards]

    def start(self, specs: Sequence[RunnerSpec]) -> None:
        """Launch one worker per non-empty shard and wait until all are built."""
        import jax

        if self._closed:
            raise RuntimeError("the executor is closed")
        if self._procs:
            raise RuntimeError("the executor is already started")
        x64 = bool(jax.config.read("jax_enable_x64"))
        by_index = {spec.segment.index: spec for spec in specs}
        try:
            for shard in assign_shards([s.segment for s in specs], self.workers):
                if not shard:
                    continue
                parent, child = self._context.Pipe()
                proc = self._context.Process(
                    target=_worker,
                    args=(child, self._factory, [by_index[i] for i in shard], x64),
                    name=f"ucb-shard-{len(self._procs)}",
                    daemon=True,
                )
                proc.start()
                child.close()
                k = len(self._procs)
                self._procs.append(proc)
                self._conns.append(parent)
                self._shards.append(shard)
                for i in shard:
                    self._shard_of[i] = k
            for k in range(len(self._procs)):
                self._receive(k)
        except BaseException:
            if not self._failed:
                self._fail()
            raise

    def _send(self, k: int, message: Any) -> None:
        try:
            self._conns[k].send(message)
        except (BrokenPipeError, OSError) as exc:
            raise self._died(k) from exc

    def _fail(self) -> None:
        """Stop every worker at once: after a failure the shards are inconsistent.

        A worker may still be busy, or blocked sending a result nobody will
        read, so waiting for it to see the stop message could take a turn.
        """
        self._failed = True
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs:
            proc.join(timeout=5.0)

    def _died(self, k: int) -> RuntimeError:
        self._procs[k].join(timeout=1.0)
        error = RuntimeError(
            f"worker {k} (segments {self._shards[k]}) died with exit code "
            f"{self._procs[k].exitcode}"
        )
        self._fail()
        return error

    def _receive(self, k: int) -> Any:
        try:
            status, payload, text = self._conns[k].recv()
        except (EOFError, OSError) as exc:
            raise self._died(k) from exc
        if status == "error":
            where = "building its runners" if payload is None else f"segment {payload}"
            self._fail()
            raise RuntimeError(f"worker {k} failed in {where}:\n{text}")
        return payload

    def _check_running(self) -> None:
        if self._failed:
            raise RuntimeError("the executor has failed; start a new one")
        if self._closed:
            raise RuntimeError("the executor is closed")

    def advance(self, jobs: Mapping[int, Job]) -> dict[int, AdvanceResult]:
        """Send each worker its shard's jobs, then gather every result."""
        self._check_running()
        parts: dict[int, dict[int, Job]] = {}
        for index, job in jobs.items():
            parts.setdefault(self._shard_of[index], {})[index] = job
        try:
            for k, part in parts.items():
                self._send(k, part)
            results: dict[int, AdvanceResult] = {}
            for k in parts:
                results.update(self._receive(k))
        except BaseException:
            if not self._failed:
                self._fail()
            raise
        return results

    def peak_rss_bytes(self) -> list[int]:
        """Return each worker's peak resident memory in bytes."""
        self._check_running()
        try:
            for k in range(len(self._procs)):
                self._send(k, "rss")
            return [int(self._receive(k)) for k in range(len(self._procs))]
        except BaseException:
            if not self._failed:
                self._fail()
            raise

    def close(self) -> None:
        """Ask every worker to stop, then terminate any that does not."""
        if self._closed:
            return
        self._closed = True
        if not self._failed:  # after a failure the workers are already stopped
            for conn in self._conns:
                with contextlib.suppress(BrokenPipeError, OSError):
                    conn.send(None)
            for proc in self._procs:
                proc.join(timeout=10.0)
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
        for conn in self._conns:
            conn.close()


__all__ = [
    "STEP_COST_US",
    "AdvanceResult",
    "Job",
    "ProcessExecutor",
    "RunnerSpec",
    "SegmentExecutor",
    "SerialExecutor",
    "advance_runner",
    "assign_shards",
    "build_runner",
    "peak_rss_bytes",
    "segment_cost",
]
