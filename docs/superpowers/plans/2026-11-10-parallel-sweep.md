# Parallel Even/Odd Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run each parity of the UCB segment sweep in worker processes and get exactly the chains of a serial sweep.

**Architecture:** A template-aware merge rule makes same-parity segments share no bins, so every job of a parity can be built from the model as it stood when the parity started. A small executor interface runs those jobs either in the main process (`SerialExecutor`) or in persistent `spawn` worker processes that each own a fixed shard of segments (`ProcessExecutor`). The block applies the results in index order.

**Tech Stack:** Python 3.10+, NumPy, JAX (x64), `multiprocessing` (spawn), impulse `RJPTSampler` (rjmcmc branch), pytest.

**User decisions (already made):**
- Order of sub-projects: "Sweep, TDI2, WDM, search".
- Exactness: "A. Template-aware merge" (the table changes; the dry runs and validation reports are re-run).
- Backend: "Local processes, pluggable".
- Spec approved ("Approve, write the spec"); work on a new branch ("Sure let's make a new branch to make these changes") → branch `parallel-sweep`, based on `ucb-segment-block` at `6dc3c4d`. PR #11 must not change.
- Spec: `docs/superpowers/specs/2026-11-09-parallel-sweep-design.md`.

---

## Environment (read first)

- Work on branch `parallel-sweep` in `/Users/aaron/Documents/lisa/binary_galactica`. Do not push.
- Python is `/Users/aaron/miniconda3` (3.11). The package is not installed; tests and tools use `PYTHONPATH`. For everything in this plan use:

  ```sh
  export PYTHONPATH=src:$HOME/Documents/GitHub/impulse_mcmc:$HOME/Documents/lisa/enchilada/src
  ```

  `$HOME/Documents/GitHub/impulse_mcmc` is impulse's `rjmcmc` branch. The installed impulse lacks `RJPTSampler`, and tests that need real runners skip without it. `enchilada/src` is needed by `check_block` tests and the validation tools.
- `ruff 0.17.0` is installed: `ruff check . && ruff format --check .` must pass. `python -m mypy src` has a pre-existing baseline of 41 errors, 2 of them in `segments/` (impulse is untyped). Don't add new ones.
- pytest-xdist is **not** installed; run pytest without `-n`.
- Spawned workers print `OMP: Warning #179: Function Can't set size of /tmp file failed` under the Claude Code sandbox. It is harmless; filter it with `grep -v "OMP: Warning"`.
- A spawned worker re-imports the main module, so a script that uses `workers > 1` needs an `if __name__ == "__main__":` guard (every tool under `tools/segments` has one).
- End every commit message with:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

## File structure

| File | Responsibility |
|---|---|
| `src/binary_galactica/segments/executor.py` (new) | `RunnerSpec`, `AdvanceResult`, `SegmentExecutor` protocol, `build_runner`, `advance_runner`, `segment_cost`, `assign_shards`, `peak_rss_bytes`, `SerialExecutor`, `ProcessExecutor` |
| `src/binary_galactica/segments/runner.py` | `ChainArchive.read` (reads a used archive directory) |
| `src/binary_galactica/segments/table.py` | template-aware merge rule, cross-cluster merging, `same_parity_disjoint` |
| `src/binary_galactica/segments/block.py` | `workers` argument; parity jobs from the parity-start model; executor lifecycle (`close`, context manager, finalizer); `cold_chain` from disk; `peak_rss_bytes` |
| `src/binary_galactica/segments/__init__.py` | exports |
| `tests/segments/fakes.py` (new) | picklable fake runners that spawned workers can import |
| `tests/segments/test_executor.py` (new) | shard assignment, executor plumbing and failures, bitwise determinism with real runners |
| `tests/segments/test_table.py`, `test_block.py`, `test_runner.py` | new and updated tests |
| `tools/segments/parallel_timing.py` (new) | worker-count timing on a Sangria sub-band → `tools/performance/segments-parallel.json` |
| `tools/segments/segment767.py`, `tools/segments/compare_glass.py` | `--workers` option |
| `tools/performance/segments-*.json` | re-run reports |
| `README.md` | parallel sweep, merge rule, new numbers |

---

### Task 1: Executors and the determinism check

**Goal:** Add `segments/executor.py` with a serial and a process executor, and prove with real runners that spawned workers reproduce the in-process runner bit for bit.

**Files:**
- Create: `src/binary_galactica/segments/executor.py`
- Modify: `src/binary_galactica/segments/runner.py` (add `ChainArchive.read` after `append`)
- Modify: `src/binary_galactica/segments/__init__.py`
- Create: `tests/segments/fakes.py`
- Create: `tests/segments/test_executor.py`
- Modify: `tests/segments/test_runner.py`

**Acceptance Criteria:**
- [ ] `pytest tests/segments/test_executor.py` passes, including the slow `test_process_workers_reproduce_the_serial_runner`. That test compares `SerialExecutor`, `ProcessExecutor(1)` and `ProcessExecutor(3)` with `np.testing.assert_array_equal`: bitwise, with no tolerance.
- [ ] A worker exception raises `RuntimeError` containing `failed in segment 2` and the worker's traceback text. A worker killed with `os._exit(3)` raises `RuntimeError` containing `died with exit code 3`. A failing runner build raises `RuntimeError` containing `building its runners`.
- [ ] After `close()`, no worker process is alive, and a second `close()` is harmless.
- [ ] `ChainArchive.read(directory)` loads a directory that already holds turns, and raises `FileNotFoundError` on a directory with none.
- [ ] `ruff check .` and `ruff format --check .` pass.

**Verify:** `python -m pytest tests/segments/test_executor.py tests/segments/test_runner.py -q 2>&1 | grep -v "OMP: Warning" | tail -3` → `... passed` with no failures (impulse's rjmcmc branch must be on `PYTHONPATH`, or the real-runner tests skip and the gate is not met).

**Gate:** if the slow determinism test fails (any array differs), STOP. Report `BLOCKED` with the first differing array and its maximum difference. Do not loosen the comparison: the spec makes bitwise equality the claim everything else builds on.

**Steps:**

- [ ] **Step 1: Write the fake runners** — create `tests/segments/fakes.py`:

```python
"""Picklable fake runners for the executor tests.

Spawned worker processes import this module by name
(``tests.segments.fakes``), so the factories must be module-level functions.
"""

from __future__ import annotations

import os
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


def shift_runner(spec):
    """Return a :class:`ShiftRunner`."""
    return ShiftRunner(spec)


def raising_runner(spec):
    """Return a :class:`RaisingRunner`."""
    return RaisingRunner(spec)


def dying_runner(spec):
    """Return a :class:`DyingRunner`."""
    return DyingRunner(spec)


def failing_build(spec):
    """Raise instead of building a runner."""
    raise ValueError(f"cannot build segment {spec.segment.index}")
```

- [ ] **Step 2: Write the executor tests** — create `tests/segments/test_executor.py`:

```python
"""Executors run segment turns in the main process or in worker processes."""

import numpy as np
import pytest

from binary_galactica.glass import ae_noise
from binary_galactica.segments import (
    ChainArchive,
    ProcessExecutor,
    RunnerSpec,
    Segment,
    SerialExecutor,
    UCBBlockConfig,
    assign_shards,
    build_segment_table,
    render_sources,
)
from binary_galactica.segments.executor import segment_cost

from . import fakes
from .support import N_FREQ, T_SHORT, native_source, rj_impulse

QS = [2934.3 + 12.0 * k for k in range(5)]


def _config(**kw):
    base = {
        "band_bins": (2920, 3000),
        "n_target": 1,
        "min_interior_bins": 8,
        "steps_per_turn": 50,
        "ntemps": 2,
        "hottest_temperature": 10.0,
    }
    base.update(kw)
    return UCBBlockConfig(**base)


def _problem(config):
    """Return five one-source segments, data with those sources, and A/E variances."""
    x = np.array([native_source(q) for q in QS])
    table = build_segment_table(x, T=T_SHORT, n_freq=N_FREQ, config=config)
    assert len(table.segments) == 5
    f = np.arange(N_FREQ) / T_SHORT
    f[0] = f[1]
    c = np.stack([0.5 * T_SHORT * ae_noise(f)] * 2)
    rng = np.random.default_rng(0)
    noise = np.sqrt(c / 2) * (
        rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)
    )
    model = np.zeros((2, N_FREQ), dtype=np.complex128)
    for seg in table.segments:
        render_sources(
            x[list(seg.owned)],
            T=T_SHORT,
            t0=0.0,
            BW=seg.bw,
            n_channels=2,
            n_freq=N_FREQ,
        ).add_to(model)
    return x, table, noise + model, c


def _specs(x, table, c, config, root):
    return [
        RunnerSpec(
            segment=seg,
            start=x[list(seg.owned)],
            T=T_SHORT,
            t0=0.0,
            config=config,
            outdir=root / f"segment_{seg.index:05d}",
            seed=100 + seg.index,
            reference=c[:, seg.bin_lo : seg.bin_hi].copy(),
            n_channels=3,
            n_freq=N_FREQ,
        )
        for seg in table.segments
    ]


def _run(executor, root, config, turns=2):
    """Run ``turns`` turns of both parities; return the results and the archives."""
    x, table, r, c = _problem(config)
    executor.start(_specs(x, table, c, config, root))
    rounds = []
    try:
        for _ in range(turns):
            for parity in (0, 1):
                jobs = {
                    seg.index: (
                        r[:, seg.bin_lo : seg.bin_hi],
                        c[:, seg.bin_lo : seg.bin_hi],
                    )
                    for seg in table.segments[parity::2]
                }
                rounds.append(executor.advance(jobs))
    finally:
        executor.close()
    archives = {
        seg.index: ChainArchive.read(root / f"segment_{seg.index:05d}" / "cold")
        for seg in table.segments
    }
    return rounds, archives


def _assert_identical(a, b):
    rounds_a, archives_a = a
    rounds_b, archives_b = b
    assert len(rounds_a) == len(rounds_b)
    for ra, rb in zip(rounds_a, rounds_b, strict=True):
        assert sorted(ra) == sorted(rb)
        for i in ra:
            np.testing.assert_array_equal(ra[i].owned, rb[i].owned)
            assert ra[i].contribution.bin_lo == rb[i].contribution.bin_lo
            np.testing.assert_array_equal(
                ra[i].contribution.values, rb[i].contribution.values
            )
    assert sorted(archives_a) == sorted(archives_b)
    for i, chain in archives_a.items():
        for key, value in chain.items():
            np.testing.assert_array_equal(value, archives_b[i][key])


def _segment(index, bw, k_max, width=100, pad=10):
    lo = 1000 + 200 * index
    return Segment(
        index=index, lo=lo, hi=lo + width, pad=pad, bw=bw, k_max=k_max, owned=()
    )


def test_assign_shards_balances_each_parity():
    segs = [
        _segment(i, bw=bw, k_max=k)
        for i, (bw, k) in enumerate(
            [(64, 20), (256, 80), (128, 20), (512, 40), (64, 10), (128, 60), (32, 10)]
        )
    ]
    shards = assign_shards(segs, 3)
    assert sorted(i for shard in shards for i in shard) == list(range(7))
    assert all(shard == sorted(shard) for shard in shards)
    for parity in (0, 1):
        group = [s for s in segs if s.index % 2 == parity]
        loads = [
            sum(segment_cost(s) for s in group if s.index in shard) for shard in shards
        ]
        assert max(loads) - min(loads) <= max(segment_cost(s) for s in group)
    # Even, most expensive first: 2, 0, 4 to workers 0, 1, 2, then 6 to worker 2
    # (the least loaded). Odd: 3, 1, 5 to workers 0, 1, 2.
    assert shards == [[2, 3], [0, 1], [4, 5, 6]]


def test_assign_shards_needs_a_worker():
    with pytest.raises(ValueError, match="at least 1"):
        assign_shards([_segment(0, 64, 10)], 0)
    with pytest.raises(ValueError, match="at least 1"):
        ProcessExecutor(0)


def test_more_workers_than_segments_leaves_shards_empty():
    assert assign_shards([_segment(0, 64, 10)], 3) == [[0], [], []]


def test_process_workers_match_the_serial_executor(tmp_path):
    config = _config()
    serial = _run(SerialExecutor(fakes.shift_runner), tmp_path / "serial", config)
    executor = ProcessExecutor(2, runner_factory=fakes.shift_runner)
    parallel = _run(executor, tmp_path / "parallel", config)
    assert executor.shards == [[1, 4], [0, 2, 3]]
    _assert_identical(serial, parallel)


def test_worker_exception_names_the_segment(tmp_path):
    executor = ProcessExecutor(2, runner_factory=fakes.raising_runner)
    with pytest.raises(RuntimeError, match=r"failed in segment 2:(.|\n)*injected"):
        _run(executor, tmp_path, _config())


def test_dead_worker_is_reported(tmp_path):
    executor = ProcessExecutor(2, runner_factory=fakes.dying_runner)
    with pytest.raises(RuntimeError, match="died with exit code 3"):
        _run(executor, tmp_path, _config())


def test_build_failure_is_reported(tmp_path):
    executor = ProcessExecutor(2, runner_factory=fakes.failing_build)
    with pytest.raises(RuntimeError, match="building its runners(.|\n)*cannot build"):
        _run(executor, tmp_path, _config())


def test_close_stops_every_worker(tmp_path):
    config = _config()
    x, table, _, c = _problem(config)
    executor = ProcessExecutor(3, runner_factory=fakes.shift_runner)
    executor.start(_specs(x, table, c, config, tmp_path))
    procs = list(executor._procs)
    assert len(procs) == 3 and all(p.is_alive() for p in procs)
    memory = executor.peak_rss_bytes()
    assert len(memory) == 3 and all(m > 0 for m in memory)
    executor.close()
    executor.close()
    assert not any(p.is_alive() for p in procs)


@pytest.mark.slow
def test_process_workers_reproduce_the_serial_runner(tmp_path):
    # The real runners, bit for bit: in-process, one worker, three workers.
    rj_impulse()
    config = _config()
    serial = _run(SerialExecutor(), tmp_path / "serial", config)
    one = _run(ProcessExecutor(1), tmp_path / "one", config)
    three = _run(ProcessExecutor(3), tmp_path / "three", config)
    _assert_identical(serial, one)
    _assert_identical(one, three)
    lnlike = serial[1][0]["lnlike"]
    assert np.unique(lnlike).size > 1  # the chains moved
```

Append to `tests/segments/test_runner.py`, after `test_archive_refuses_a_used_directory`:

```python
def test_archive_read_accepts_a_used_directory(tmp_path):
    archive = ChainArchive(tmp_path)
    archive.append(0, samples=np.ones((2, 8)), k=[1, 1], lnlike=[0.0, 1.0])
    chain = ChainArchive.read(tmp_path)
    assert chain["lnlike"].tolist() == [0.0, 1.0]
    assert chain["turn"].tolist() == [0, 0]
    with pytest.raises(FileNotFoundError):
        ChainArchive.read(tmp_path / "missing")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_executor.py tests/segments/test_runner.py -q 2>&1 | tail -3`
Expected: collection error `ImportError: cannot import name 'ProcessExecutor'`.

- [ ] **Step 4: Add `ChainArchive.read`** — in `src/binary_galactica/segments/runner.py`, insert this classmethod between `append` and `files`:

```python
    @classmethod
    def read(cls, directory: str | Path) -> dict[str, NDArray[Any]]:
        """Return :meth:`load` of an archive another process may be writing.

        Unlike the constructor, this accepts a directory that holds turns.
        """
        archive = cls.__new__(cls)
        archive.directory = Path(directory)
        return archive.load()
```

- [ ] **Step 5: Create `src/binary_galactica/segments/executor.py`**:

```python
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

A script that uses :class:`ProcessExecutor` must guard its entry point with
``if __name__ == "__main__":``, as ``spawn`` re-imports the main module.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import resource
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

#: The dry run's fitted cost of one step (``tools/segments/dry_run.py``): a fixed
#: part and parts per template bin, per slot-bin and per window bin, in
#: microseconds. :func:`assign_shards` uses it only as relative weights.
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

    @property
    def shards(self) -> list[list[int]]:
        """Return each running worker's segment indices."""
        return [list(shard) for shard in self._shards]

    def start(self, specs: Sequence[RunnerSpec]) -> None:
        """Launch one worker per non-empty shard and wait until all are built."""
        import jax

        if self._procs:
            raise RuntimeError("the executor is already started")
        x64 = bool(jax.config.read("jax_enable_x64"))
        by_index = {spec.segment.index: spec for spec in specs}
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

    def _send(self, k: int, message: Any) -> None:
        try:
            self._conns[k].send(message)
        except (BrokenPipeError, OSError) as exc:
            raise self._died(k) from exc

    def _died(self, k: int) -> RuntimeError:
        self._procs[k].join(timeout=1.0)
        return RuntimeError(
            f"worker {k} (segments {self._shards[k]}) died with exit code "
            f"{self._procs[k].exitcode}"
        )

    def _receive(self, k: int) -> Any:
        try:
            status, payload, text = self._conns[k].recv()
        except (EOFError, OSError) as exc:
            raise self._died(k) from exc
        if status == "error":
            where = "building its runners" if payload is None else f"segment {payload}"
            raise RuntimeError(f"worker {k} failed in {where}:\n{text}")
        return payload

    def advance(self, jobs: Mapping[int, Job]) -> dict[int, AdvanceResult]:
        """Send each worker its shard's jobs, then gather every result."""
        parts: dict[int, dict[int, Job]] = {}
        for index, job in jobs.items():
            parts.setdefault(self._shard_of[index], {})[index] = job
        for k, part in parts.items():
            self._send(k, part)
        results: dict[int, AdvanceResult] = {}
        for k in parts:
            results.update(self._receive(k))
        return results

    def peak_rss_bytes(self) -> list[int]:
        """Return each worker's peak resident memory in bytes."""
        for k in range(len(self._procs)):
            self._send(k, "rss")
        return [int(self._receive(k)) for k in range(len(self._procs))]

    def close(self) -> None:
        """Ask every worker to stop, then terminate any that does not."""
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                conn.send(None)
        for proc in self._procs:
            proc.join(timeout=10.0)
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
```

- [ ] **Step 6: Export the executor names** — in `src/binary_galactica/segments/__init__.py`, add after `from .config import BACKENDS, UCBBlockConfig`:

```python
from .executor import (
    AdvanceResult,
    ProcessExecutor,
    RunnerSpec,
    SerialExecutor,
    assign_shards,
)
```

and add `"AdvanceResult"`, `"ProcessExecutor"`, `"RunnerSpec"`, `"SerialExecutor"` and `"assign_shards"` to `__all__`, keeping it sorted the way it is now: capitalised names first in ASCII order, then lower-case names (`"AdvanceResult"` before `"BACKENDS"`, `"ProcessExecutor"` after `"PriorRedrawMove"`, `"RunnerSpec"` after `"PsiPhiMove"`, `"SerialExecutor"` after `"SegmentTable"`, `"assign_shards"` after `"UCBBlockConfig"`).

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/segments/test_executor.py tests/segments/test_runner.py -q 2>&1 | grep -v "OMP: Warning" | tail -3`
Expected: all passed, 0 skipped among the executor tests. A prototype of this code took about 11 s for the executor file, including the slow test.

- [ ] **Step 8: Lint and the fast segment suite**

Run: `ruff check . && ruff format --check . && python -m pytest tests/segments -m "not slow" -q 2>&1 | grep -v "OMP: Warning" | tail -2`
Expected: ruff clean; all passed (the block tests still use the in-process loop and are untouched by this task).

- [ ] **Step 9: Commit**

```bash
git add src/binary_galactica/segments/executor.py src/binary_galactica/segments/runner.py \
  src/binary_galactica/segments/__init__.py tests/segments/fakes.py \
  tests/segments/test_executor.py tests/segments/test_runner.py
git commit -m "Add serial and process executors for the segment sweep

Workers own fixed shards of segments, built from picklable specs, and
reproduce the in-process runner bit for bit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

```json:metadata
{"files": ["src/binary_galactica/segments/executor.py", "src/binary_galactica/segments/runner.py", "src/binary_galactica/segments/__init__.py", "tests/segments/fakes.py", "tests/segments/test_executor.py", "tests/segments/test_runner.py"], "verifyCommand": "python -m pytest tests/segments/test_executor.py tests/segments/test_runner.py -q", "acceptanceCriteria": ["test_process_workers_reproduce_the_serial_runner passes with bitwise comparisons", "worker exception, dead worker and build failure raise RuntimeError with the stated messages", "no worker alive after close(); second close harmless", "ChainArchive.read loads a used directory and raises FileNotFoundError on an empty one", "ruff check and ruff format --check pass"], "modelTier": "standard"}
```

---

### Task 2: Template-aware merge rule

**Goal:** Change the segment table so that same-parity segments share no bins (windows disjoint, no owned template reaching the other's window), including across `skip_gaps` clusters, and check it table-wide.

**Files:**
- Modify: `src/binary_galactica/segments/table.py` (module docstring; `_first_violation`; new `_reach`, `_extent`, `_parity_gap`, `_bws`, `_cluster_conflict`, `same_parity_disjoint`; `_Tiling`; `_tiling`, `_cluster_tiling`, `_clustered`, `build_segment_table` thread `n_freq` and use the tiling's `bws`)
- Modify: `src/binary_galactica/segments/__init__.py`
- Test: `tests/segments/test_table.py`

**Acceptance Criteria:**
- [ ] For an interior with both neighbours, the merge pass enforces `w_s >= max(pad_s, e_{s-1} + pad_{s+1}, pad_{s-1} + e_{s+1})`, with `e = max(pad, bw//2 + 1)`. At the ends of a tiling the old rule `w_s >= max(pad_s, pad of the one neighbour)` applies.
- [ ] `build_segment_table` calls `same_parity_disjoint(table)`, which raises `ValueError` matching `segments 0 and 2 share a parity but are 30 bins apart; their windows and templates need 43` for the hand-built table in the test.
- [ ] Under `skip_gaps`, `_cluster_conflict` merges two clusters when a same-parity pair that straddles them is too close, as well as when their windows overlap.
- [ ] The 80-source crowded set (15 bins apart, ln A = -50, T = 1 yr) tiles into exactly `[(120000, 120293, 64, 256, 20), (120293, 120593, 64, 256, 20), (120593, 120893, 65, 256, 20), (120893, 121186, 65, 256, 20)]` as `(lo, hi, pad, bw, len(owned))`. The old rule gave 8 segments, all six same-parity pairs within reach.
- [ ] Rendering a source at each interior edge gives bins outside every same-parity window.
- [ ] Every existing test in `tests/segments/test_table.py` still passes unchanged (a prototype showed the pinned tilings do not move).
- [ ] On Sangria the table has 770 segments (862 with `skip_gaps`, in 133 clusters).

**Verify:** `python -m pytest tests/segments/test_table.py -q` → all passed.

**Steps:**

- [ ] **Step 1: Write the failing tests** — in `tests/segments/test_table.py`, change the imports to:

```python
from binary_galactica.segments import (
    Segment,
    SegmentTable,
    UCBBlockConfig,
    build_segment_table,
    doppler_bins,
    padding_bins,
    render_sources,
    same_parity_disjoint,
)
from binary_galactica.segments.table import _cluster_conflict, _Tiling
```

Add a parity check to `_check_rules`, after its last loop:

```python
    _check_parity(table)
```

and define, just above `_check_rules`:

```python
def _check_parity(table):
    """Same-parity segments: windows disjoint, no owned template reaching across."""
    segs = table.segments
    for a, b in zip(segs[:-2], segs[2:], strict=True):
        assert a.hi + max(a.pad, a.bw // 2 + 1) <= b.lo - b.pad
        assert a.hi + a.pad <= b.lo - max(b.pad, b.bw // 2 + 1)
```

Also call `_check_parity(table)` at the end of `_check_cluster_rules`. Then append these tests:

```python
def _crowded():
    """80 sources 15 bins apart: their 256-bin templates outreach 64-bin pads."""
    return _sources(120_000.37 + 15.0 * np.arange(80))


def test_owned_templates_never_reach_a_same_parity_window():
    # The pads-only rule left this set in 8 segments, with every same-parity
    # pair within reach of a template; the reach term merges them to 4.
    table = build_segment_table(_crowded(), T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    _check_rules(table)
    assert [(s.lo, s.hi, s.pad, s.bw, len(s.owned)) for s in table.segments] == [
        (120_000, 120_293, 64, 256, 20),
        (120_293, 120_593, 64, 256, 20),
        (120_593, 120_893, 65, 256, 20),
        (120_893, 121_186, 65, 256, 20),
    ]
    assert all(s.bw // 2 + 1 > s.pad for s in table.segments)


def test_rendered_edge_sources_stay_out_of_same_parity_windows():
    table = build_segment_table(_crowded(), T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    segs = table.segments
    for s, seg in enumerate(segs):
        for q in (seg.lo + 1e-6, seg.hi - 1e-6):
            part = render_sources(
                _sources([q]), T=T, t0=0.0, BW=seg.bw, n_channels=2, n_freq=N_FREQ
            )
            assert part.values.shape[1] == seg.bw
            for j in (s - 2, s + 2):
                if 0 <= j < len(segs):
                    other = segs[j]
                    assert part.bin_hi <= other.bin_lo or part.bin_lo >= other.bin_hi


def _hand_table(middle_width):
    segs = (
        Segment(index=0, lo=1000, hi=1100, pad=10, bw=64, k_max=10, owned=()),
        Segment(
            index=1, lo=1100, hi=1100 + middle_width, pad=10, bw=64, k_max=10, owned=()
        ),
        Segment(
            index=2,
            lo=1100 + middle_width,
            hi=1300,
            pad=10,
            bw=64,
            k_max=10,
            owned=(),
        ),
    )
    return SegmentTable(T=T, n_freq=N_FREQ, segments=segs)


def test_same_parity_disjoint_rejects_a_reaching_template():
    # need = max(max(10, 64//2 + 1) + 10, 10 + max(10, 33)) = 43
    same_parity_disjoint(_hand_table(43))
    with pytest.raises(
        ValueError,
        match="segments 0 and 2 share a parity but are 30 bins apart; "
        "their windows and templates need 43",
    ):
        same_parity_disjoint(_hand_table(30))


def test_clusters_merge_when_a_template_reaches_across_the_gap():
    # The windows are disjoint (1140 + 10 <= 1160 - 10), but segment 0's
    # 256-bin templates reach 129 bins past 1100, into segment 2's window.
    a = _Tiling(edges=[1000, 1100, 1140], owned=[(), ()], pads=[10, 10], bws=[256, 64])
    b = _Tiling(edges=[1160, 1300], owned=[()], pads=[10], bws=[64])
    assert _cluster_conflict([a, b]) == (0, 1)
    narrow = _Tiling(
        edges=[1000, 1100, 1140], owned=[(), ()], pads=[10, 10], bws=[64, 64]
    )
    assert _cluster_conflict([narrow, b]) is None
    touching = _Tiling(edges=[1150, 1300], owned=[()], pads=[10], bws=[64])
    assert _cluster_conflict([narrow, touching]) == (0, 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_table.py -q 2>&1 | tail -3`
Expected: collection error `ImportError: cannot import name 'same_parity_disjoint'`.

- [ ] **Step 3: Implement the rule in `src/binary_galactica/segments/table.py`.** Make each replacement exactly once.

(a) In the module docstring, replace

```
``|fdot|`` among the owned start sources. A merge pass then enforces
``w_s >= max(pad_s, pad_{s-1} + pad_{s+1})``. Padding then reaches only
adjacent interiors, and windows of the same parity are disjoint.
```

with

```
``|fdot|`` among the owned start sources. Each interior's template bandwidth
is GLASS's, capped by its window: ``bw = min(ucb_bw, floor_pow2(w + 2 pad))``.
An owned template reaches ``r = bw//2 + 1`` bins past its interior (its
carrier bin ``trunc(f0*T)`` lies inside, plus one bin of margin). With
``e = max(pad, r)``, a merge pass enforces

    w_s >= max(pad_s, e_{s-1} + pad_{s+1}, pad_{s-1} + e_{s+1})

and, at an end of the tiling, ``w_s >= max(pad_s, pad of the one neighbour)``.
Padding then reaches only adjacent interiors; windows of the same parity are
disjoint; and no owned template reaches a same-parity window. A parity of the
sweep can therefore run in parallel (:func:`same_parity_disjoint`).
```

and replace

```
Two neighbouring clusters whose padded windows overlap are merged into one and
re-tiled, until every gap separates the windows.
```

with

```
Two neighbouring clusters are merged into one and re-tiled while their padded
windows overlap, or while two same-parity segments that straddle them are
closer than the merge rule allows.
```

(b) Replace the whole `_first_violation` function with:

```python
def _reach(bw: int) -> int:
    """Return how many bins an owned template reaches past its interior."""
    return int(bw) // 2 + 1


def _extent(pad: int, bw: int) -> int:
    """Return how far a segment's window or owned templates reach past its interior."""
    return max(int(pad), _reach(bw))


def _parity_gap(a_pad: int, a_bw: int, b_pad: int, b_bw: int) -> int:
    """Return the bins two same-parity interiors need between them."""
    return max(_extent(a_pad, a_bw) + b_pad, a_pad + _extent(b_pad, b_bw))


def _bws(edges, owned, native, T, pads, n_freq) -> list[int]:
    """Return each interior's template bandwidth: GLASS's, capped by its window."""
    bws = []
    for (a, b), own, pad in zip(
        zip(edges[:-1], edges[1:], strict=True), owned, pads, strict=True
    ):
        if own:
            fdot = float(np.max(np.abs(native[list(own), 7]))) / T**2
            amp = float(np.max(np.exp(native[list(own), 3])))
        else:
            fdot, amp = 0.0, 0.0
        glass = ucb_bw(b / T, fdot, 0.0, amp, T, n_freq)
        bws.append(int(min(glass, _floor_pow2((b - a) + 2 * pad))))
    return bws


def _first_violation(edges: list[int], pads: list[int], bws: list[int]) -> int | None:
    n = len(pads)
    for s in range(n):
        need = pads[s]
        if 0 < s < n - 1:
            need = max(
                need, _parity_gap(pads[s - 1], bws[s - 1], pads[s + 1], bws[s + 1])
            )
        elif s > 0:
            need = max(need, pads[s - 1])
        elif s < n - 1:
            need = max(need, pads[s + 1])
        if edges[s + 1] - edges[s] < need:
            return s
    return None
```

(c) Give `_Tiling` a `bws: list[int]` field after `pads`, and make `narrow` call `_first_violation(self.edges, self.pads, self.bws)`.

(d) `_tiling` gains an `n_freq: int` parameter between `T` and `config`. In its loop, compute `bws = _bws(edges, owned, x, T, pads, n_freq)` after `pads`, call `_first_violation(edges, pads, bws)`, and return `_Tiling(edges, owned, pads, bws)`.

(e) `_cluster_tiling` gains `n_freq: int` between `T` and `config` and passes it to `_tiling(lo, hi, q, members, x, T, n_freq, config)`.

(f) Replace the whole `_clustered` function with:

```python
def _cluster_conflict(tilings: list[_Tiling]) -> tuple[int, int] | None:
    """Return the first two clusters that must merge, or ``None``.

    Neighbouring clusters must merge if their padded windows overlap. Two
    clusters must also merge if same-parity segments ``s`` and ``s + 2`` (in
    the global order) lie in them and are closer than the merge rule allows.
    """
    rows = [
        (c, a, b, pad, bw)
        for c, t in enumerate(tilings)
        for a, b, pad, bw in zip(t.edges[:-1], t.edges[1:], t.pads, t.bws, strict=True)
    ]
    for (c, _, hi, pad, _), (d, lo, _, pad2, _) in zip(
        rows[:-1], rows[1:], strict=True
    ):
        if c != d and hi + pad > lo - pad2:
            return c, d
    for (c, _, hi, pad, bw), (d, lo, _, pad2, bw2) in zip(
        rows[:-2], rows[2:], strict=True
    ):
        if c != d and lo - hi < _parity_gap(pad, bw, pad2, bw2):
            return c, d
    return None


def _clustered(
    band: tuple[int, int],
    q: NDArray[np.float64],
    members: NDArray[np.int64],
    x: NDArray[np.float64],
    T: float,
    n_freq: int,
    config: UCBBlockConfig,
) -> list[_Tiling]:
    """Return the tilings of the clusters, merging clusters that conflict."""
    tilings = [
        _cluster_tiling(lo, hi, band, q, members, x, T, n_freq, config)
        for lo, hi in _cluster_spans(q, members, x, T, band)
    ]
    while (pair := _cluster_conflict(tilings)) is not None:
        first, last = pair
        group = tilings[first : last + 1]
        lo = min(t.edges[0] for t in group)
        hi = max(t.edges[-1] for t in group)
        tilings[first : last + 1] = [
            _cluster_tiling(lo, hi, band, q, members, x, T, n_freq, config)
        ]
    return tilings
```

(g) In `build_segment_table`, pass `n_freq` to `_clustered((band_lo, band_hi), q, members, x, T, n_freq, config)` and to `_tiling(band_lo, band_hi, q, members, x, T, n_freq, config)`. Build the interiors with each tiling's `bws`:

```python
    interiors = [
        (a, b, own, pad, bw)
        for t in tilings
        for a, b, own, pad, bw in zip(
            t.edges[:-1], t.edges[1:], t.owned, t.pads, t.bws, strict=True
        )
    ]
    owned_all = sorted(i for _, _, own, _, _ in interiors for i in own)
```

Unpack `for s, (a, b, own, pad, bw) in enumerate(interiors):` and delete the lines that recomputed `n_bins`, `fdot`, `amp` and `bw` there (the `Segment(...)` call keeps `bw=int(bw)`). Replace the final `return SegmentTable(...)` with:

```python
    table = SegmentTable(
        T=float(T), n_freq=int(n_freq), segments=tuple(segments), outside=outside
    )
    same_parity_disjoint(table)
    return table


def same_parity_disjoint(table: SegmentTable) -> None:
    """Raise ``ValueError`` if two segments of the same parity can interact.

    Segments ``s`` and ``s + 2`` must be far enough apart that their padded
    windows are disjoint and neither one's owned templates (``bw//2 + 1`` bins
    past its interior) reach the other's window. A parity of the sweep can
    then run in any order, or in parallel, with the same result.
    """
    segs = table.segments
    for a, b in zip(segs[:-2], segs[2:], strict=True):
        need = _parity_gap(a.pad, a.bw, b.pad, b.bw)
        if b.lo - a.hi < need:
            raise ValueError(
                f"segments {a.index} and {b.index} share a parity but are "
                f"{b.lo - a.hi} bins apart; their windows and templates need {need}"
            )
```

Add `"same_parity_disjoint"` to the module's `__all__`.

(h) In `src/binary_galactica/segments/__init__.py`, import `same_parity_disjoint` from `.table` (after `padding_bins`) and add `"same_parity_disjoint"` at the end of `__all__`.

- [ ] **Step 4: Run the table tests**

Run: `python -m pytest tests/segments/test_table.py -q 2>&1 | tail -2`
Expected: all passed (the existing pinned tilings are unchanged).

- [ ] **Step 5: Check Sangria's counts** (needs the Sangria file; skip with a note if it is missing):

```sh
python - <<'EOF'
import sys
sys.path.insert(0, "tools/segments")
import numpy as np, runs, sangria
from binary_galactica.segments import UCBBlockConfig, build_segment_table, ldc_to_native
p = sangria.DEFAULT_PATH; n, dt = sangria.grid(p); T = n * dt
x = ldc_to_native(runs.catalogue_rows(p, T=T, snr_min=7.0), T)
for skip in (False, True):
    s = build_segment_table(x, T=T, n_freq=n // 2 + 1, config=UCBBlockConfig(skip_gaps=skip)).segments
    clusters = 1 + sum(a.hi != b.lo for a, b in zip(s[:-1], s[1:]))
    print(skip, len(s), clusters, max(len(a.owned) for a in s))
EOF
```

Expected: `False 770 1 42` and `True 862 133 42`.

- [ ] **Step 6: Lint and the fast segment suite**

Run: `ruff check . && ruff format --check . && python -m pytest tests/segments -m "not slow" -q 2>&1 | grep -v "OMP: Warning" | tail -2`
Expected: ruff clean; all passed.

- [ ] **Step 7: Commit**

```bash
git add src/binary_galactica/segments/table.py src/binary_galactica/segments/__init__.py tests/segments/test_table.py
git commit -m "Keep same-parity segments from sharing any bin

The merge rule now counts how far an owned template reaches past its
interior, clusters merge when a same-parity pair straddling them is too
close, and the table checks the property as a whole. Sangria tiles into
770 segments (862 with skip_gaps).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

```json:metadata
{"files": ["src/binary_galactica/segments/table.py", "src/binary_galactica/segments/__init__.py", "tests/segments/test_table.py"], "verifyCommand": "python -m pytest tests/segments/test_table.py -q", "acceptanceCriteria": ["merge pass enforces w_s >= max(pad_s, e_{s-1} + pad_{s+1}, pad_{s-1} + e_{s+1}) with e = max(pad, bw//2 + 1); end rule unchanged", "same_parity_disjoint raises the stated ValueError for the hand table", "_cluster_conflict merges clusters on a cross-gap template reach and on window overlap", "crowded 80-source set tiles into the four pinned segments", "edge renders stay outside same-parity windows", "existing test_table tests pass unchanged", "Sangria: 770 segments, 862 with skip_gaps in 133 clusters"], "modelTier": "standard"}
```

---

### Task 3: Block sweep through the executors

**Goal:** Make `UCBBlock` build each parity's jobs from the parity-start model, run them through an executor chosen by a new `workers` argument, and manage the executor's lifecycle.

**Files:**
- Modify: `src/binary_galactica/segments/block.py`
- Test: `tests/segments/test_block.py`

**Acceptance Criteria:**
- [ ] `UCBBlock(..., workers=N)` validates `N` as a positive integer before writing any file. `workers=1` uses `SerialExecutor`; more uses `ProcessExecutor(N)`.
- [ ] Every existing block test passes after the fixture patches `binary_galactica.segments.executor.SegmentRunner`. The sweep order stays `[0, 2, 1]`, and segment 1 still sees the *updated* neighbours.
- [ ] `test_worker_processes_publish_what_the_main_process_does` gives bitwise equal templates, published parameters and archives with `ProcessExecutor(2)` and the fake runner. The slow `test_parallel_block_reproduces_the_serial_block` does the same with real runners and `ProcessExecutor(3)`.
- [ ] `test_snapshot_sweep_matches_the_sequential_loop` replays a turn one segment at a time. Every job equals, bit for bit, the job the old sequential loop would have built, and the updates do change the model.
- [ ] A worker failure raises `RuntimeError` (`failed in segment 2`), marks the block failed, and leaves no worker alive. Garbage-collecting a block whose workers are running stops them (`weakref.finalize`).
- [ ] `close()` (also via `with`) makes `sample()` raise `RuntimeError("the block is closed; ...")`, keeps `cold_chain` working, and a second `close()` is harmless.
- [ ] `peak_rss_bytes()` returns one entry per worker (one for the main process with `workers=1`).

**Verify:** `python -m pytest tests/segments/test_block.py -q 2>&1 | grep -v "OMP: Warning" | tail -2` → all passed (with `enchilada/src` on `PYTHONPATH` the two `check_block` tests run too).

**Steps:**

- [ ] **Step 1: Update the fixture and write the failing tests** — in `tests/segments/test_block.py`:

Add `import gc` before `import importlib` and `from pathlib import Path` after `import json`. Then replace the package import and the module handle with:

```python
from binary_galactica.segments import (
    ChainArchive,
    ProcessExecutor,
    SerialExecutor,
    UCBBlock,
    UCBBlockConfig,
    render_sources,
)

from . import fakes
from .support import DT, N_FREQ, N_TIME, T_SHORT, ldc_rows, rj_impulse

block_module = importlib.import_module("binary_galactica.segments.block")
executor_module = importlib.import_module("binary_galactica.segments.executor")
```

Replace `_FakeRunner.__init__`, its `archive` property and `advance` with:

```python
    def __init__(self, segment, start, **kwargs):
        self.segment = segment
        self.start = np.array(start).reshape(-1, 8)
        self.archive = ChainArchive(Path(kwargs["outdir"]) / "cold")
        self.turn = 0

    def advance(self, residual, c):
        if _FakeRunner.fail is not None and len(_FakeRunner.calls) == _FakeRunner.fail:
            raise _InjectedFailureError(f"segment {self.segment.index}")
        _FakeRunner.calls.append((self.segment.index, np.array(residual)))
        self.archive.append(
            self.turn,
            samples=np.zeros((1, 8)),
            k=[1],
            lnlike=[float(self.segment.index)],
        )
        self.turn += 1
```

In the `fake_runner` fixture, patch the executor module instead of the block module:

```python
    monkeypatch.setattr(executor_module, "SegmentRunner", _FakeRunner)
```

In `test_cold_chain_accessor`, replace the lines from `chain = block.cold_chain(1)` to the end with:

```python
    chain = block.cold_chain(1)
    assert chain["k"].tolist() == [1] and chain["lnlike"].tolist() == [1.0]
    with pytest.raises(KeyError):
        block.cold_chain(99)
    block._active.discard(1)  # a segment with no start sources and a fixed count
    assert block.cold_chain(1) is None
```

Append these tests:

```python
def test_workers_must_be_a_positive_integer(tmp_path):
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="workers must be a positive integer"):
            UCBBlock(
                grid=_grid(),
                start_sources=_rows(),
                outdir=tmp_path / str(bad),
                config=_config(),
                workers=bad,
            )
        assert not (tmp_path / str(bad)).exists()


def test_close_ends_sampling_but_keeps_the_chains(tmp_path, fake_runner):
    with UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    ) as block:
        block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))
        assert len(block.peak_rss_bytes()) == 1
    with pytest.raises(RuntimeError, match="the block is closed"):
        block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))
    assert block.cold_chain(0)["k"].tolist() == [1]
    block.close()  # a second close is harmless


def _published(tmp_path, executor, monkeypatch):
    """Run two turns with ``executor``; return the published states and chains."""
    monkeypatch.setattr(block_module, "_make_executor", lambda workers: executor)
    block = UCBBlock(
        grid=_grid(),
        start_sources=_rows(),
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    rng = np.random.default_rng(9)
    states = []
    with block:
        state = block.initial_block_state(_grid())
        for _ in range(2):
            state = block.sample(_grid(), _cov(), state, rng=rng)
            states.append(state)
        chains = [block.cold_chain(s.index) for s in block.table.segments]
    return states, chains


def _assert_same_run(a, b):
    for sa, sb in zip(a[0], b[0], strict=True):
        for ch in CHANNELS:
            np.testing.assert_array_equal(sa.template[ch], sb.template[ch])
        for xa, xb in zip(
            sa.parameters["segments"], sb.parameters["segments"], strict=True
        ):
            np.testing.assert_array_equal(xa["native"], xb["native"])
    for ca, cb in zip(a[1], b[1], strict=True):
        for key in ca:
            np.testing.assert_array_equal(ca[key], cb[key])


def test_worker_processes_publish_what_the_main_process_does(tmp_path, monkeypatch):
    serial = _published(
        tmp_path / "serial", SerialExecutor(fakes.shift_runner), monkeypatch
    )
    parallel = _published(
        tmp_path / "parallel",
        ProcessExecutor(2, runner_factory=fakes.shift_runner),
        monkeypatch,
    )
    _assert_same_run(serial, parallel)


class _RecordingExecutor(SerialExecutor):
    """Shift runners in the main process; records each parity's jobs and results."""

    def __init__(self):
        super().__init__(fakes.shift_runner)
        self.rounds = []

    def advance(self, jobs):
        results = super().advance(jobs)
        self.rounds.append((dict(jobs), results))
        return results


def test_snapshot_sweep_matches_the_sequential_loop(tmp_path, monkeypatch):
    # Replay the second turn one segment at a time, as the old loop ran it:
    # each segment's job must equal, bit for bit, the one built after the
    # segments before it were applied.
    executor = _RecordingExecutor()
    monkeypatch.setattr(block_module, "_make_executor", lambda workers: executor)
    block = UCBBlock(
        grid=_grid(),
        start_sources=_rows(),
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    rng = np.random.default_rng(5)
    data = {
        ch: 1e-21 * (rng.standard_normal(N_FREQ) + 1j * rng.standard_normal(N_FREQ))
        for ch in CHANNELS
    }
    r = np.stack([data["A"], data["E"]])
    state = block.sample(_grid(data=data), _cov(), None, rng=rng)
    model = block._model.copy()
    contrib = dict(block._contrib)
    block.sample(_grid(data=data), _cov(), state, rng=rng)
    assert [sorted(jobs) for jobs, _ in executor.rounds[-2:]] == [[0, 2], [1]]
    changed = False
    for jobs, results in executor.rounds[-2:]:
        start = model.copy()
        for index in sorted(jobs):
            seg = block.table.segments[index]
            lo, hi = seg.bin_lo, seg.bin_hi
            own = contrib[index].window(lo, hi)[:2]
            np.testing.assert_array_equal(
                jobs[index][0], r[:, lo:hi] - (model[:2, lo:hi] - own)
            )
            contrib[index].add_to(model, -1.0)
            results[index].contribution.add_to(model, 1.0)
            contrib[index] = results[index].contribution
            changed = changed or not np.array_equal(model, start)
    assert changed  # the updates moved the model, so the check is not vacuous
    block.close()


def test_collecting_the_block_stops_its_workers(tmp_path, monkeypatch):
    executor = ProcessExecutor(2, runner_factory=fakes.shift_runner)
    monkeypatch.setattr(block_module, "_make_executor", lambda workers: executor)
    block = UCBBlock(
        grid=_grid(),
        start_sources=_rows(),
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))
    procs = list(executor._procs)
    assert procs and all(p.is_alive() for p in procs)
    del block
    gc.collect()
    assert not any(p.is_alive() for p in procs)


def test_a_worker_failure_fails_the_block(tmp_path, monkeypatch):
    executor = ProcessExecutor(2, runner_factory=fakes.raising_runner)
    monkeypatch.setattr(block_module, "_make_executor", lambda workers: executor)
    block = UCBBlock(
        grid=_grid(),
        start_sources=_rows(),
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    with pytest.raises(RuntimeError, match="failed in segment 2"):
        block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))
    assert not any(p.is_alive() for p in executor._procs)
    with pytest.raises(RuntimeError, match="failed mid-sweep"):
        block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))


@pytest.mark.slow
def test_parallel_block_reproduces_the_serial_block(tmp_path, monkeypatch):
    rj_impulse()
    serial = _published(tmp_path / "serial", SerialExecutor(), monkeypatch)
    parallel = _published(tmp_path / "parallel", ProcessExecutor(3), monkeypatch)
    _assert_same_run(serial, parallel)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_block.py -q 2>&1 | grep -v "OMP: Warning" | tail -3`
Expected: errors and failures. The fixture's `SegmentRunner` patch still works, but `workers=` is an unexpected keyword, `_make_executor` does not exist, and `cold_chain` still reads `runner.archive.load()`.

- [ ] **Step 3: Implement in `src/binary_galactica/segments/block.py`.** Make each replacement exactly once.

(a) Module docstring, step 3 becomes:

```
3. sweeps even segments, then odd. Each segment's sampler sees the conditional
   residual minus the other segments' current owned sources, on its padded
   window. Every job of a parity is built from the model as it stood when the
   parity started, so a parity can run in worker processes (``workers``);
```

(b) Imports: add `import weakref` after `import json`. Replace `from .runner import SegmentRunner` with `from .runner import ChainArchive`, and add

```python
from .executor import ProcessExecutor, RunnerSpec, SegmentExecutor, SerialExecutor
```

after `from .config import UCBBlockConfig`.

(c) Above `def _names`, add:

```python
def _make_executor(workers: int) -> SegmentExecutor:
    """Return the sweep's executor: in-process for one worker, else processes."""
    return SerialExecutor() if workers == 1 else ProcessExecutor(workers)
```

(d) In the class docstring, after the `name` entry, add:

```
    workers
        Processes for the sweep. ``1`` (the default) runs every segment in the
        main process. More starts that many ``spawn`` worker processes at the
        first turn, each owning a fixed shard of segments for the whole run
        (:class:`~binary_galactica.segments.executor.ProcessExecutor`). The
        chains do not depend on ``workers``. Stop the workers with
        :meth:`close`, or use the block as a context manager; a script that
        uses workers needs an ``if __name__ == "__main__":`` guard.
```

(e) Signature and first lines of `__init__`:

```python
        name: str = "ucb",
        workers: int = 1,
    ) -> None:
        """Validate the inputs, build the segment table and write the run files."""
        if isinstance(workers, bool) or int(workers) != workers or workers < 1:
            raise ValueError(f"workers must be a positive integer, got {workers!r}")
        self.workers = int(workers)
        self.outdir = Path(outdir)
```

Replace `self._runners: dict[int, SegmentRunner | None] | None = None` with:

```python
        self._active: set[int] | None = None
        self._executor: SegmentExecutor | None = None
        self._finalizer: weakref.finalize | None = None
        self._closed = False
```

(f) Replace the body of `_start` up to (not including) `self._contrib = {` with:

```python
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
```

Seeds are drawn in the same order as before (only for segments with a runner), so a serial run on the same table draws the same seeds.

(g) Replace `cold_chain` and add the lifecycle methods after it:

```python
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
```

(h) In `sample`'s docstring, replace the paragraph that starts "Within a parity the segments advance in index order" with:

```
        Every job of a parity is built from the model as it stood when the
        parity started; the parity's updates are applied in index order once
        it finishes. The segment table guarantees that no same-parity update
        reaches another same-parity window
        (:func:`~binary_galactica.segments.table.same_parity_disjoint`), so
        this equals advancing the segments one by one in index order, whether
        they run in the main process or in ``workers`` processes.
```

After the `if self._failed:` block in `sample`, add:

```python
        if self._closed:
            raise RuntimeError("the block is closed; build a fresh UCBBlock")
```

and make the failure handler stop the workers:

```python
        try:
            model = self._turn(r, c, rng)
        except BaseException:
            self._failed = True
            if self._finalizer is not None:
                self._finalizer()  # stop the workers; the block cannot continue
            raise
```

(i) In `_turn`, replace everything from `if self._runners is None:` through the end of the parity loop with:

```python
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
```

(`_render` stays: `initial_block_state` and `_start` still use it.)

- [ ] **Step 4: Run the block tests**

Run: `python -m pytest tests/segments/test_block.py -q 2>&1 | grep -v "OMP: Warning" | tail -2`
Expected: all passed. A prototype gave 29 passed, with `check_block` included.

- [ ] **Step 5: Full suite, lint, types**

Run: `ruff check . && ruff format --check . && python -m pytest -q 2>&1 | grep -v "OMP: Warning" | tail -2 && python -m mypy src 2>&1 | tail -1`
Expected: ruff clean. pytest all passed (a prototype of all six tasks' code gave 505 passed in about 160 s). mypy reports no more than the 41-error baseline, with no new errors in `segments/`.

- [ ] **Step 6: Commit**

```bash
git add src/binary_galactica/segments/block.py tests/segments/test_block.py
git commit -m "Run each sweep parity through an executor, optionally in worker processes

UCBBlock(workers=N) builds a parity's jobs from the model at the start of
the parity and applies the results in index order, so the chains do not
depend on N. The block closes its workers on failure, on close() and on
exit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

```json:metadata
{"files": ["src/binary_galactica/segments/block.py", "tests/segments/test_block.py"], "verifyCommand": "python -m pytest tests/segments/test_block.py -q", "acceptanceCriteria": ["workers validated before any file is written; 1 -> SerialExecutor, more -> ProcessExecutor", "existing block tests pass with the fixture patching executor.SegmentRunner; order [0, 2, 1]", "serial and ProcessExecutor runs publish bitwise equal templates, parameters and archives (fake and real runners)", "worker failure raises RuntimeError, marks the block failed, leaves no worker alive", "close() and with-block end sampling; cold_chain still works; double close harmless", "peak_rss_bytes returns one entry per process"], "modelTier": "standard"}
```

---

### Task 4: Worker-count measurement

**Goal:** Add `tools/segments/parallel_timing.py`, run it on the 2–4 mHz Sangria band, and commit `tools/performance/segments-parallel.json`.

**Files:**
- Create: `tools/segments/parallel_timing.py`
- Create: `tools/performance/segments-parallel.json` (generated)

**Acceptance Criteria:**
- [ ] The tool times workers 1, 2, 4, 8 and 16, each with inherited and with single-threaded XLA flags. Each configuration runs in its own subprocess, and a failed configuration is recorded with its stderr rather than aborting the run.
- [ ] Each record holds `segments`, `active_segments`, `turn_s`, `steady_turn_s`, `steps_per_s`, `worker_peak_rss_bytes`, `main_peak_rss_bytes` and `speedup_vs_one_worker`.
- [ ] `tools/performance/segments-parallel.json` is committed from a full default run.

**Verify:** `python -c "import json; d=json.load(open('tools/performance/segments-parallel.json')); print([(r['workers'], r['single_thread'], round(r.get('speedup_vs_one_worker', 0), 2)) for r in d['results']])"` → ten entries.

**Steps:**

- [ ] **Step 1: Create `tools/segments/parallel_timing.py`**:

```python
"""Time the parallel even/odd sweep on a Sangria sub-band for several worker counts.

Each configuration runs in a fresh Python process (``--one``), so its peak
memory and ``XLA_FLAGS`` are its own. A configuration builds the block on the
band with the residual noise reference (as ``segment767.py`` does), runs
``--turns`` turns of ``--steps`` steps, and records:

- ``turn_s``: every turn's wall time; the last is the steady value, because
  the first includes the compiles;
- ``steps_per_s``: active segments x steps / the last turn's time;
- ``worker_peak_rss_bytes``: each worker's peak resident memory (the main
  process's with one worker) and ``main_peak_rss_bytes``.

``single_thread`` configurations set ``XLA_FLAGS`` to ``SINGLE_THREAD`` for
the configuration's processes; the others inherit the caller's environment.

    PYTHONPATH=src:... python tools/segments/parallel_timing.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import runs
import sangria

from binary_galactica.segments import UCBBlock, UCBBlockConfig
from binary_galactica.segments.executor import peak_rss_bytes

SINGLE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
MARGIN = runs.MARGIN
HALF_WIDTH = runs.HALF_WIDTH


def measure(
    *,
    workers: int,
    band_mhz: tuple[float, float],
    steps: int,
    turns: int,
    seed: int,
    path: Path,
    outdir: Path,
) -> dict:
    """Run one configuration in this process and return its record."""
    n, dt = sangria.grid(path)
    T = n * dt
    band = (int(band_mhz[0] * 1e-3 * T), int(band_mhz[1] * 1e-3 * T))
    rows = runs.catalogue_rows(path, T=T, band=band, snr_min=runs.START_SNR)
    spectra = sangria.aet_spectra(*sangria.combine(["obs"], ["mbhb", "vgb"], path))
    data = runs.frequency_data(spectra, n, dt)
    lo, hi = band[0] - MARGIN, band[1] + MARGIN
    resid, _ = runs.residual_spectra(
        spectra, path, T=T, lo=lo - HALF_WIDTH, hi=hi + HALF_WIDTH
    )
    cov = runs.covariance(
        data, runs.running_median_variance(resid, lo, hi, HALF_WIDTH), lo, hi
    )
    config = UCBBlockConfig(band_bins=band, steps_per_turn=steps)
    with UCBBlock(
        grid=data,
        start_sources=rows,
        outdir=outdir,
        config=config,
        reference_covariance=cov,
        workers=workers,
    ) as block:
        segs = block.table.segments
        if segs[0].bin_lo < lo or segs[-1].bin_hi > hi:
            raise RuntimeError(
                f"padded band {segs[0].bin_lo}..{segs[-1].bin_hi} exceeds "
                f"the running-median range {lo}..{hi}"
            )
        _, seconds = runs.run_block(block, data, cov, turns, seed)
        worker_rss = block.peak_rss_bytes()
        active = sum(
            1 for s in segs if config.birth_death or block.owned_start[s.index]
        )
    return {
        "workers": workers,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "band_mhz": list(band_mhz),
        "band_bins": list(band),
        "start_sources": int(len(rows)),
        "segments": len(segs),
        "active_segments": active,
        "steps_per_turn": steps,
        "turn_s": seconds,
        "steady_turn_s": seconds[-1],
        "steps_per_s": active * steps / seconds[-1],
        "worker_peak_rss_bytes": worker_rss,
        "main_peak_rss_bytes": peak_rss_bytes(),
    }


def _one(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="parallel-timing-") as tmp:
        record = measure(
            workers=args.workers[0],
            band_mhz=tuple(args.band_mhz),
            steps=args.steps,
            turns=args.turns,
            seed=args.seed,
            path=args.path,
            outdir=Path(tmp) / "run",
        )
    args.record.write_text(json.dumps(record, indent=1) + "\n")


def _configuration(args: argparse.Namespace, workers: int, single: bool) -> dict:
    env = dict(os.environ)
    if single:
        env["XLA_FLAGS"] = SINGLE_THREAD
    with tempfile.TemporaryDirectory(prefix="parallel-timing-record-") as tmp:
        record = Path(tmp) / "record.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--one",
            "--record",
            str(record),
            "--workers",
            str(workers),
            "--band-mhz",
            *(str(x) for x in args.band_mhz),
            "--steps",
            str(args.steps),
            "--turns",
            str(args.turns),
            "--seed",
            str(args.seed),
            "--path",
            str(args.path),
        ]
        done = subprocess.run(command, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            return {
                "workers": workers,
                "single_thread": single,
                "error": done.stderr[-4000:],
            }
        result = json.loads(record.read_text())
    result["single_thread"] = single
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--band-mhz", type=float, nargs=2, default=[2.0, 4.0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--path", type=Path, default=sangria.DEFAULT_PATH)
    parser.add_argument(
        "--threading",
        choices=("both", "inherit", "single"),
        default="both",
        help="XLA threading variants to time",
    )
    parser.add_argument("--out", default="segments-parallel.json")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        _one(args)
        return
    variants = {"both": (False, True), "inherit": (False,), "single": (True,)}
    results = []
    for single in variants[args.threading]:
        for workers in args.workers:
            result = _configuration(args, workers, single)
            results.append(result)
            if "error" in result:
                print(f"workers={workers} single_thread={single}: FAILED")
                print(result["error"])
                continue
            print(
                f"workers={workers:2d} single_thread={single!s:5s} "
                f"segments={result['active_segments']} "
                f"steady turn {result['steady_turn_s']:7.1f} s "
                f"{result['steps_per_s']:8.0f} steps/s "
                f"worker rss max {max(result['worker_peak_rss_bytes']) / 2**30:.2f} GiB"
            )
    serial = {
        r["single_thread"]: r["steps_per_s"]
        for r in results
        if r.get("workers") == 1 and "error" not in r
    }
    for r in results:
        if "error" not in r and r["single_thread"] in serial:
            r["speedup_vs_one_worker"] = r["steps_per_s"] / serial[r["single_thread"]]
    report = {
        "machine_cpu_count": os.cpu_count(),
        "single_thread_flags": SINGLE_THREAD,
        "steps_per_turn": args.steps,
        "turns": args.turns,
        "note": (
            "steady_turn_s is the last turn's wall time; the first turn includes "
            "the compiles. speedup_vs_one_worker compares steps_per_s with the "
            "one-worker run of the same threading."
        ),
        "results": results,
    }
    print(f"wrote {runs.write_report(args.out, report)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run** (a few minutes; it writes a throwaway report):

```sh
python tools/segments/parallel_timing.py --workers 1 2 --band-mhz 3.0 3.05 \
  --steps 5 --turns 2 --threading inherit --out segments-parallel-smoke.json \
  2>&1 | grep -v "OMP: Warning"
rm tools/performance/segments-parallel-smoke.json
```

Expected: two `workers=` lines with no `FAILED`, then `wrote .../segments-parallel-smoke.json`. A prototype run gave steady turns of about 1.6 s on 9 segments, about 1.4 GiB per worker, and about 5.5 GiB in the main process, which holds the Sangria data.

- [ ] **Step 3: Full run** (about 30–60 minutes; run it in the background and do not run other heavy jobs meanwhile, since it is a timing):

```sh
python tools/segments/parallel_timing.py 2>&1 | grep -v "OMP: Warning" | tee $TMPDIR/parallel_timing.log
```

Expected: ten configuration lines and `wrote .../tools/performance/segments-parallel.json`. A 16-worker run may come close to the machine's 64 GiB: if one fails, its record keeps the error, and you report it.

- [ ] **Step 4: Report the recommendation** — print the table the README will use:

```sh
python - <<'EOF'
import json
d = json.load(open("tools/performance/segments-parallel.json"))
ok = [r for r in d["results"] if "error" not in r]


def total(r):
    # One worker runs in the main process; with more, the main process holds the data.
    extra = sum(r["worker_peak_rss_bytes"]) if r["workers"] > 1 else 0
    return r["main_peak_rss_bytes"] + extra


for r in ok:
    print(r["workers"], r["single_thread"], f"{r['steady_turn_s']:.1f} s",
          f"{r['speedup_vs_one_worker']:.2f}x",
          f"{max(r['worker_peak_rss_bytes']) / 2**30:.2f} GiB/worker",
          f"{total(r) / 2**30:.1f} GiB total")
fit = [r for r in ok if total(r) <= 48 * 2**30]
best = max(fit, key=lambda r: r["steps_per_s"])
print("recommended:", best["workers"], "workers, single_thread =", best["single_thread"])
EOF
```

The recommended count is the fastest configuration whose processes together (the main process plus its workers) stay under 48 GiB, 75% of this machine's memory. Record that line in your report.

- [ ] **Step 5: Commit**

```bash
git add tools/segments/parallel_timing.py tools/performance/segments-parallel.json
git commit -m "Time the parallel sweep for 1-16 workers on the 2-4 mHz band

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

```json:metadata
{"files": ["tools/segments/parallel_timing.py", "tools/performance/segments-parallel.json"], "verifyCommand": "python -c \"import json; d=json.load(open('tools/performance/segments-parallel.json')); print(len(d['results']))\"", "acceptanceCriteria": ["tool times workers 1, 2, 4, 8, 16 with inherited and single-threaded XLA flags, each in its own subprocess; failures recorded, not fatal", "records hold segments, active_segments, turn_s, steady_turn_s, steps_per_s, worker_peak_rss_bytes, main_peak_rss_bytes, speedup_vs_one_worker", "segments-parallel.json committed from a full default run", "recommended worker count reported (fastest with main plus workers under 48 GiB)"], "modelTier": "standard"}
```

---

### Task 5: Re-run the validation on the new table

**Goal:** Give the validation tools a `--workers` option and re-run the dry runs, the segment-767 band and the GLASS comparison on the new table.

**Files:**
- Modify: `tools/segments/segment767.py` (`run(..., workers=1)`, `--workers`)
- Modify: `tools/segments/compare_glass.py` (`compare_segment(..., workers=1)`, `--workers`)
- Regenerate: `tools/performance/segments-dry-run.json`, `tools/performance/segments-dry-run-skip-gaps.json`, `tools/performance/segments-767.json`, `tools/performance/segments-glass-compare.json`

**Acceptance Criteria:**
- [ ] `segment767.py` and `compare_glass.py` accept `--workers N` (default 1) and pass it to `UCBBlock`. Their existing tests pass.
- [ ] Both dry-run reports are regenerated: `n_segments` is 770 for the default and 862 for `skip_gaps`.
- [ ] `segments-767.json` is regenerated for `fixed` and `rj`, and `segments-glass-compare.json` for GLASS segments 113, 125 and 276. Each `pass` value is reported as found. A `false` is reported, not tuned away.

**Verify:** `python -c "import json; p='tools/performance/'; print(json.load(open(p+'segments-dry-run.json'))['n_segments'], json.load(open(p+'segments-dry-run-skip-gaps.json'))['n_segments'], {m: json.load(open(p+'segments-767.json'))[m]['pass'] for m in ('fixed','rj')}, json.load(open(p+'segments-glass-compare.json'))['pass'])"` → `770 862 {...} ...` with the pass values.

**Steps:**

- [ ] **Step 1: Add `--workers` to `tools/segments/segment767.py`.** In `run`, add a keyword parameter `workers: int = 1` after `path`, and pass `workers=workers` to the `UCBBlock(...)` call. Wrap the sampling so the workers stop. Replace

```python
    inner = runs.inner_segments(block, core)
    _, seconds = runs.run_block(block, data, cov, turns, seed)
```

with

```python
    inner = runs.inner_segments(block, core)
    with block:
        _, seconds = runs.run_block(block, data, cov, turns, seed)
```

(`cold_chain` keeps working after close.) In `main`, add

```python
    parser.add_argument("--workers", type=int, default=1)
```

and pass `workers=args.workers` to `run(...)`.

- [ ] **Step 2: Add `--workers` to `tools/segments/compare_glass.py`.** In `compare_segment`, add a keyword parameter `workers=1` after `path`, and pass `workers=workers` to `UCBBlock(...)`. Replace

```python
    _, seconds = runs.run_block(block, data, cov, turns, seed)
```

with

```python
    with block:
        _, seconds = runs.run_block(block, data, cov, turns, seed)
```

In `main`, add `parser.add_argument("--workers", type=int, default=1)` and pass `workers=args.workers` to `compare_segment(...)`.

- [ ] **Step 3: Tool tests**

Run: `python -m pytest tests/segments/test_segment767.py tests/segments/test_compare_glass.py tests/segments/test_runs.py -q 2>&1 | tail -2 && ruff check tools && ruff format --check tools`
Expected: passed (or skipped where the Sangria file is missing); ruff clean.

- [ ] **Step 4: Re-run the dry runs** (serial by design: they time single segments):

```sh
python tools/segments/dry_run.py 2>&1 | tail -5
python tools/segments/dry_run.py --skip-gaps --out segments-dry-run-skip-gaps.json 2>&1 | tail -5
```

Expected: the reports are rewritten with `n_segments` 770 and 862.

- [ ] **Step 5: Re-run the segment-767 band and the GLASS comparison** with the worker count Task 4 recommended (call it `W`; set `XLA_FLAGS` to the single-thread flags if Task 4 recommended them). These take tens of minutes each, so run them in the background one after another:

```sh
python tools/segments/segment767.py --mode fixed --workers W 2>&1 | grep -v "OMP: Warning" | tail -4
python tools/segments/segment767.py --mode rj --workers W 2>&1 | grep -v "OMP: Warning" | tail -4
python tools/segments/compare_glass.py --workers W 2>&1 | grep -v "OMP: Warning" | tail -6
```

Expected: each ends with `PASS` or `FAIL`. Under the previous table, all three passed: fixed with 66 sources at SNR ≥ 7, RJ with 49 at SNR ≥ 10, GLASS overlap 0.85.

- [ ] **Step 6: Commit**

```bash
git add tools/segments/segment767.py tools/segments/compare_glass.py \
  tools/performance/segments-dry-run.json tools/performance/segments-dry-run-skip-gaps.json \
  tools/performance/segments-767.json tools/performance/segments-glass-compare.json
git commit -m "Re-run the dry runs and validation on the parity-safe table

The validation tools take --workers and close their blocks.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

```json:metadata
{"files": ["tools/segments/segment767.py", "tools/segments/compare_glass.py", "tools/performance/segments-dry-run.json", "tools/performance/segments-dry-run-skip-gaps.json", "tools/performance/segments-767.json", "tools/performance/segments-glass-compare.json"], "verifyCommand": "python -m pytest tests/segments/test_segment767.py tests/segments/test_compare_glass.py tests/segments/test_runs.py -q", "acceptanceCriteria": ["segment767.py and compare_glass.py accept --workers and pass it to UCBBlock; their tests pass", "dry-run reports regenerated with n_segments 770 and 862", "segments-767.json (fixed, rj) and segments-glass-compare.json regenerated; pass values reported as found"], "modelTier": "standard"}
```

---

### Task 6: README

**Goal:** Document the parallel sweep, the merge rule and the new numbers in `README.md`.

**Files:**
- Modify: `README.md` (the "UCB segment block" section)

**Acceptance Criteria:**
- [ ] The section describes `workers`, the parity-start snapshot, the bitwise-equality guarantee, `close()`/`with`, and the `__main__` guard.
- [ ] The merge rule and the Sangria counts (770 segments, 862 with `skip_gaps`, at most 42 owned start sources) are stated.
- [ ] Every number in the dry-run, 767, GLASS and parallel bullets matches the committed JSON reports, as printed by the Step 1 script.

**Verify:** `python -m pytest tests -q -k "readme or doc" 2>&1 | tail -1 && ruff check .` (doctests run with the suite), then a careful read of the diff against the Step 1 printout.

**Steps:**

- [ ] **Step 1: Print the numbers** the README needs:

```sh
python - <<'EOF'
import json
p = "tools/performance/"
d = json.load(open(p + "segments-dry-run.json"))
g = json.load(open(p + "segments-dry-run-skip-gaps.json"))
print("dry run:", d["n_segments"], "segments;", [round(x / 3600, 2) for x in d["predicted_turn_s_range"]], "h; tail share", round(d["tail"]["share_of_total"], 4))
print("skip_gaps:", g["n_segments"], "segments in", g["n_clusters"], "clusters, cover", round(g["interior_fraction_of_band"], 3), ";", [round(x / 3600, 2) for x in g["predicted_turn_s_range"]], "h")
s = json.load(open(p + "segments-767.json"))
for m in ("fixed", "rj"):
    r = s[m]
    print("767", m, "pass", r["pass"], "n_at_floor", r["n_at_floor"], "criterion", r["criterion"])
c = json.load(open(p + "segments-glass-compare.json"))
print("glass: pass", c["pass"], "n_bright", c["n_bright"], "overlap", round(c["bright_overlap_fraction"], 2), "->", round(c["bright_overlap_fraction"] * c["n_bright"]), "of", c["n_bright"])
q = json.load(open(p + "segments-parallel.json"))
ok = [r for r in q["results"] if "error" not in r]


def total(r):
    # One worker runs in the main process; with more, the main process holds the data.
    extra = sum(r["worker_peak_rss_bytes"]) if r["workers"] > 1 else 0
    return r["main_peak_rss_bytes"] + extra


fit = [r for r in ok if total(r) <= 48 * 2**30]
best = max(fit, key=lambda r: r["steps_per_s"])
one = next(r for r in ok if r["workers"] == 1 and r["single_thread"] == best["single_thread"])
print("parallel:", best["segments"], "segments;", best["workers"], "workers, single_thread", best["single_thread"], f"{best['speedup_vs_one_worker']:.1f}x", f"{one['steady_turn_s']:.0f} s -> {best['steady_turn_s']:.0f} s", f"{max(best['worker_peak_rss_bytes']) / 2**30:.1f} GiB/worker")
print("full-band turn with that speedup:", [round(x / 3600 / best["speedup_vs_one_worker"], 2) for x in d["predicted_turn_s_range"]], "h")
EOF
```

- [ ] **Step 2: Edit the "UCB segment block" section of `README.md`.**

(a) In the `skip_gaps` paragraph, replace "On the Sangria start set (10369 sources) the 971 segments then cover" with "On the Sangria start set (10369 sources) the 862 segments then cover". The rest of that sentence (345,338 of 663,332 bins; widest 182,294 → 3,093) is unchanged by the new rule.

(b) After that paragraph, insert:

```markdown
Segments of the same parity never share a bin. With `r = bw//2 + 1`, how far
an owned template reaches past its interior, and `e = max(pad, r)`, the tiling
merges neighbouring interiors until every interior satisfies
`w_s >= max(pad_s, e_{s-1} + pad_{s+1}, pad_{s-1} + e_{s+1})`; under
`skip_gaps` it also merges clusters that would break this across a gap. Then
the windows of segments `s - 1` and `s + 1` are disjoint and neither one's
owned templates reach the other's window (`same_parity_disjoint` checks it).
On Sangria that gives 770 segments (862 with `skip_gaps`), with at most 42
owned start sources per segment.
```

(c) After the numbered `sample()` list, insert this paragraph, filling the bracketed values from the Step 1 `parallel:` line:

```markdown
Each parity's jobs are built from the model as it stood when the parity
started, so a parity can run in parallel: `UCBBlock(..., workers=N)` starts N
`spawn` worker processes at the first turn, each owning a fixed shard of
segments for the whole run. The chains do not depend on N; serial and worker
runs agree bit for bit. Stop the workers with `block.close()` or a `with`
block, and guard a script that uses them with `if __name__ == "__main__":`. On
this machine (16 cores, 64 GiB) the 2-4 mHz band ([segments] segments,
`segments-parallel.json`) ran [speedup]x faster with [workers] workers than
with one ([one] s → [best] s per 200-step turn), at up to [rss] GiB per
worker.
```

If Task 4 recommended single-threaded flags, add the sentence: "Set `XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'` before starting Python; each worker then uses one core."

(d) Update the validation bullets from the Step 1 printout:
- **Segment 767:** the fixed and RJ sentences use the new `n_at_floor` values and PASS/FAIL. The "+-600 bins" wording stays.
- **GLASS comparison:** the counts become "[k] of [n_bright] … ([overlap])". If the report no longer passes, say so plainly.
- **Dry run:** "become 770 segments", the new hour range and tail share, and "With `skip_gaps` … there are 862 segments in [n_clusters] clusters, covering [cover]% of the band, and [range] h per turn". Append: "With [workers] workers that is about [full-band range] h per turn."

- [ ] **Step 3: Check and commit**

Run: `ruff check . && python -m pytest -q 2>&1 | grep -v "OMP: Warning" | tail -1`
Expected: clean; all passed.

```bash
git add README.md
git commit -m "Document the parallel sweep, the parity-safe merge rule and the new runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

```json:metadata
{"files": ["README.md"], "verifyCommand": "python -m pytest -q", "acceptanceCriteria": ["README describes workers, the parity-start snapshot, bitwise equality, close()/with and the __main__ guard", "README states the merge rule and 770 / 862 segments, max 42 owned", "every README number in the dry-run, 767, GLASS and parallel bullets matches the committed JSON reports"], "modelTier": "mechanical"}
```
