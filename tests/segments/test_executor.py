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
