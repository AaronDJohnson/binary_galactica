"""The UCB block sweeps segments, publishes owned sources and survives failures."""

import gc
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from binary_galactica.glass import ae_noise
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
CHANNELS = ("A", "E", "T")
QS = (2940.3, 2952.3, 2964.3)


class _Data(SimpleNamespace):
    def block_state(self, template=None, *, covariance_matrix=None, parameters=None):
        return SimpleNamespace(template=template, parameters=parameters)


def _grid(channels=CHANNELS, domain="frequency", data=None, t0=0.0):
    data = data or {ch: np.zeros(N_FREQ, dtype=complex) for ch in channels}
    return _Data(
        channel_names=channels,
        num_time_samples=N_TIME,
        sample_rate_hz=1 / DT,
        start_time_gps=t0,
        data_domain=domain,
        channel_data=data,
    )


def _cov(channels=CHANNELS):
    f = np.arange(N_FREQ) / T_SHORT
    f[0] = f[1]
    m = np.zeros((N_FREQ, len(channels), len(channels)))
    for i in range(len(channels)):
        m[:, i, i] = 0.5 * T_SHORT * ae_noise(f)
    mask = np.ones(N_FREQ, dtype=bool)
    mask[0] = False
    return SimpleNamespace(
        data_domain="frequency",
        covariance_matrix=m,
        channel_names=channels,
        active_mask=mask,
    )


def _rows(qs=QS, amplitudes=None):
    amplitudes = amplitudes or [np.exp(-48.0)] * len(qs)
    rows = np.concatenate(
        [
            ldc_rows(Frequency=q / T_SHORT, Amplitude=a)
            for q, a in zip(qs, amplitudes, strict=True)
        ]
    )
    return rows


def _config(**kw):
    base = {
        "band_bins": (2930, 2980),
        "n_target": 1,
        "min_interior_bins": 8,
        "steps_per_turn": 10,
        "ntemps": 2,
        "hottest_temperature": 10.0,
    }
    base.update(kw)
    return UCBBlockConfig(**base)


def _render(sources, seg, n_channels=3):
    return render_sources(
        sources, T=T_SHORT, t0=0.0, BW=seg.bw, n_channels=n_channels, n_freq=N_FREQ
    )


class _Noise:
    def psd(self, f):
        return ae_noise(f)


class _InjectedFailureError(Exception):
    pass


class _FakeRunner:
    """Records each advance; ``move`` replaces and ``extra`` appends cold sources.

    With ``fail = n``, the advance made after ``n`` recorded calls raises.
    """

    calls: list = []
    extra = None
    move = None
    fail = None

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

    def cold_sources(self):
        base = self.start.copy()
        if _FakeRunner.move is not None:
            moved = _FakeRunner.move(self.segment)
            if moved is not None:
                base = moved
        if _FakeRunner.extra is None:
            return base
        return np.concatenate([base, _FakeRunner.extra(self.segment)])


@pytest.fixture
def fake_runner(monkeypatch):
    _FakeRunner.calls = []
    _FakeRunner.extra = None
    _FakeRunner.move = None
    _FakeRunner.fail = None
    monkeypatch.setattr(executor_module, "SegmentRunner", _FakeRunner)
    return _FakeRunner


def test_cold_chain_accessor(tmp_path, fake_runner):
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    with pytest.raises(RuntimeError):
        block.cold_chain(0)
    block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))
    chain = block.cold_chain(1)
    assert chain["k"].tolist() == [1] and chain["lnlike"].tolist() == [1.0]
    with pytest.raises(KeyError):
        block.cold_chain(99)
    block._active.discard(1)  # a segment with no start sources and a fixed count
    assert block.cold_chain(1) is None


def test_rejects_unsupported_inputs(tmp_path, fake_runner):
    with pytest.raises(ValueError, match="channels"):
        UCBBlock(
            grid=_grid(("X", "Y", "Z")),
            start_sources=_rows(),
            outdir=tmp_path,
            config=_config(),
        )
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="frequency"):
        block.sample(_grid(domain="time"), _cov(), None, rng=rng)
    with pytest.raises(TypeError, match="DataCovariance"):
        block.sample(_grid(), SimpleNamespace(data_domain="time"), None, rng=rng)
    with pytest.raises(ValueError, match="start time"):
        block.sample(_grid(t0=100.0), _cov(), None, rng=rng)


def test_rejects_bad_covariances(tmp_path, fake_runner):
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    rng = np.random.default_rng(0)
    correlated = _cov()
    m = correlated.covariance_matrix
    m[:, 0, 1] = m[:, 1, 0] = 0.1 * np.sqrt(m[:, 0, 0] * m[:, 1, 1])
    with pytest.raises(ValueError, match="uncorrelated"):
        block.sample(_grid(), correlated, None, rng=rng)
    with pytest.raises(ValueError, match="covariance channels"):
        block.sample(_grid(), _cov(("A", "E", "X")), None, rng=rng)


def test_render_rejects_t_without_sangria():
    source = np.array([[2952.3, 0.2, 1.1, -48.0, 0.3, 0.7, 2.0, 0.0]])
    for fmt in ("phase", "frequency"):
        with pytest.raises(ValueError, match="sangria"):
            render_sources(
                source, T=T_SHORT, t0=0.0, BW=16, n_channels=3, n_freq=N_FREQ, fmt=fmt
            )


def test_writes_run_files(tmp_path, fake_runner):
    UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    for name in ("config.json", "segment_table.json", "dropped_start_sources.json"):
        assert (tmp_path / name).exists()
    assert UCBBlockConfig.read(tmp_path / "config.json") == _config()


def test_drops_weak_start_sources(tmp_path, fake_runner):
    rows = _rows(amplitudes=[np.exp(-48.0), 1e-23, np.exp(-48.0)])
    block = UCBBlock(
        grid=_grid(),
        start_sources=rows,
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    dropped = json.loads((tmp_path / "dropped_start_sources.json").read_text())
    assert [d["index"] for d in dropped] == [1]
    assert dropped[0]["reason"] == "snr below cut" and dropped[0]["snr"] < 5
    assert block.owned_start[1] == []


def test_sweep_order_and_conditional_residual(tmp_path, fake_runner):
    # Zero data: segment 1 must then see exactly minus its neighbours' sources.
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))
    assert [s for s, _ in fake_runner.calls] == [0, 2, 1]
    seg = block.table.segments[1]
    others = sum(
        render_sources(
            block.start_native[[i]],
            T=T_SHORT,
            t0=0.0,
            BW=block.table.segments[i].bw,
            n_channels=3,
            n_freq=N_FREQ,
        ).window(seg.bin_lo, seg.bin_hi)[:2]
        for i in (0, 2)
    )
    assert np.max(np.abs(others)) > 0
    scale = np.max(np.abs(others))
    np.testing.assert_allclose(
        fake_runner.calls[2][1], -others, rtol=1e-12, atol=1e-12 * scale
    )


def test_sweep_subtracts_updated_neighbours(tmp_path, fake_runner):
    # The even segments move their owned source towards segment 1 (staying in
    # their interior) and brighten it. Segment 1 advances after them, so it
    # must see the data minus the UPDATED sources, not the start set.
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    segs = block.table.segments
    moved = {}
    for i, edge in ((0, segs[0].hi - 0.5), (2, segs[2].lo + 0.5)):
        src = block.start_native[segs[i].owned[0]].copy()
        src[0] = edge
        src[3] += 0.5
        moved[i] = src[None]
    fake_runner.move = lambda segment: moved.get(segment.index)

    seg = segs[1]
    lo, hi = seg.bin_lo, seg.bin_hi
    updated = sum(_render(moved[i], segs[i]).window(lo, hi)[:2] for i in (0, 2))
    stale = sum(
        _render(block.start_native[[i]], segs[i]).window(lo, hi)[:2] for i in (0, 2)
    )
    scale = np.max(np.abs(updated))
    assert np.max(np.abs(updated - stale)) > 0.1 * scale
    rng = np.random.default_rng(5)
    data = {
        ch: scale * (rng.standard_normal(N_FREQ) + 1j * rng.standard_normal(N_FREQ))
        for ch in CHANNELS
    }
    r = np.stack([data["A"], data["E"]])

    state = block.sample(_grid(data=data), _cov(), None, rng=np.random.default_rng(1))
    assert [s for s, _ in fake_runner.calls] == [0, 2, 1]
    np.testing.assert_allclose(
        fake_runner.calls[2][1], r[:, lo:hi] - updated, rtol=1e-12, atol=1e-12 * scale
    )

    published = [moved[0], block.start_native[[1]], moved[2]]
    expected = sum(
        _render(x, s).window(0, N_FREQ) for x, s in zip(published, segs, strict=True)
    )
    for j, ch in enumerate(CHANNELS):
        np.testing.assert_allclose(
            state.template[ch],
            expected[j],
            rtol=1e-12,
            atol=1e-12 * np.max(np.abs(expected)),
        )
    np.testing.assert_array_equal(state.parameters["segments"][2]["native"], moved[2])


def test_rejects_a_ledger_out_of_step(tmp_path, fake_runner):
    block = UCBBlock(
        grid=_grid(),
        start_sources=_rows(),
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    rng = np.random.default_rng(4)
    state0 = block.initial_block_state(_grid())
    assert state0.parameters["turn"] == 0
    state1 = block.sample(_grid(), _cov(), state0, rng=rng)
    assert state1.parameters["turn"] == 1 == block.turn
    with pytest.raises(ValueError, match="turn 0 but the block is at turn 1"):
        block.sample(_grid(), _cov(), state0, rng=rng)
    assert block.sample(_grid(), _cov(), state1, rng=rng).parameters["turn"] == 2


def test_refuses_to_continue_after_a_failed_sweep(tmp_path, fake_runner):
    block = UCBBlock(
        grid=_grid(),
        start_sources=_rows(),
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    rng = np.random.default_rng(7)
    # A rejected input changes nothing, so the block stays usable.
    with pytest.raises(ValueError, match="start time"):
        block.sample(_grid(t0=100.0), _cov(), None, rng=rng)
    state = block.sample(_grid(), _cov(), None, rng=rng)
    assert block.turn == 1

    fake_runner.calls = []
    fake_runner.fail = 1  # segment 0 advances, then segment 2's advance raises
    with pytest.raises(_InjectedFailureError, match="segment 2"):
        block.sample(_grid(), _cov(), state, rng=rng)
    assert [s for s, _ in fake_runner.calls] == [0]
    assert block.turn == 1

    fake_runner.fail = None
    with pytest.raises(RuntimeError, match="failed mid-sweep; build a fresh UCBBlock"):
        block.sample(_grid(), _cov(), state, rng=rng)
    assert [s for s, _ in fake_runner.calls] == [0]


def test_initial_block_state_needs_the_reference(tmp_path, fake_runner):
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    with pytest.raises(ValueError, match="reference_covariance"):
        block.initial_block_state(_grid())


def test_ae_layout(tmp_path, fake_runner):
    ae = ("A", "E")
    block = UCBBlock(
        grid=_grid(ae),
        start_sources=_rows(),
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(ae),
    )
    segs = block.table.segments
    initial = block.initial_block_state(_grid(ae))
    state = block.sample(_grid(ae), _cov(ae), initial, rng=np.random.default_rng(6))
    assert [s for s, _ in fake_runner.calls] == [0, 2, 1]
    expected = sum(
        _render(block.start_native[[i]], segs[i]).window(0, N_FREQ) for i in range(3)
    )
    for published in (initial, state):
        assert set(published.template) == set(ae)
        for j, ch in enumerate(ae):
            np.testing.assert_allclose(
                published.template[ch],
                expected[j],
                rtol=1e-12,
                atol=1e-12 * np.max(np.abs(expected)),
            )


def test_publishes_owned_sources_only(tmp_path, fake_runner):
    def padding_source(segment):
        src = block.start_native[segment.owned[0]].copy()
        src[0] = segment.bin_lo + 0.5
        return src[None]

    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    fake_runner.extra = padding_source
    state = block.sample(_grid(), _cov(), None, rng=np.random.default_rng(2))
    counts = [s["count"] for s in state.parameters["segments"]]
    assert counts == [1, 1, 1]
    assert set(state.template) == set(CHANNELS)
    first = state.parameters["segments"][0]
    np.testing.assert_allclose(first["ldc"]["Frequency"], QS[0] / T_SHORT, rtol=1e-12)


def test_initial_block_state_renders_the_start_set(tmp_path, fake_runner):
    block = UCBBlock(
        grid=_grid(),
        start_sources=_rows(),
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    state = block.initial_block_state(_grid())
    expected = sum(
        render_sources(
            block.start_native[[i]],
            T=T_SHORT,
            t0=0.0,
            BW=block.table.segments[i].bw,
            n_channels=3,
            n_freq=N_FREQ,
        ).window(0, N_FREQ)
        for i in range(3)
    )
    for j, ch in enumerate(CHANNELS):
        np.testing.assert_allclose(
            state.template[ch], expected[j], rtol=1e-12, atol=1e-30
        )


def test_skip_gaps_block_leaves_the_gap_empty(tmp_path, fake_runner):
    # Margins are 2 bins on this grid, so the 26-bin gap above 2944.3 splits
    # the start set into two clusters.
    qs = (2940.3, 2944.3, 2970.3)
    block = UCBBlock(
        grid=_grid(),
        start_sources=_rows(qs=qs),
        outdir=tmp_path,
        config=_config(skip_gaps=True),
    )
    below, above = block.table.segments
    assert below.owned == (0, 1) and above.owned == (2,)
    assert below.hi < above.lo and below.bin_hi <= above.bin_lo
    state = block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))
    assert [s for s, _ in fake_runner.calls] == [0, 1]
    assert [s["count"] for s in state.parameters["segments"]] == [2, 1]
    for ch in CHANNELS:
        model = state.template[ch]
        assert np.all(model[below.bin_hi : above.bin_lo] == 0)
        assert np.any(model[below.lo : below.hi] != 0)
        assert np.any(model[above.lo : above.hi] != 0)


def _check_block_run(tmp_path, rows, config):
    """Run enchilada's check_block for two cycles with the real segment runners."""
    enchilada = pytest.importorskip("enchilada")
    from enchilada.testing import check_block

    rj_impulse()
    kwargs = {
        "sample_rate_hz": 1 / DT,
        "tdi_generation": "1.5",
        "physical_observable": "fractional_frequency",
        "channel_names": CHANNELS,
        "num_time_samples": N_TIME,
        "data_domain": "frequency",
    }
    grid = enchilada.L1Data(
        {ch: np.zeros(N_FREQ, dtype=complex) for ch in CHANNELS}, **kwargs
    )
    cov = enchilada.DataCovariance.from_psd(grid, _Noise())
    block = UCBBlock(
        grid=grid,
        start_sources=rows,
        outdir=tmp_path,
        config=config,
        reference_covariance=cov,
    )
    signal = block.initial_block_state(grid).template
    rng = np.random.default_rng(3)
    data = {}
    for i, ch in enumerate(CHANNELS):
        var = cov.covariance_matrix[:, i, i].real
        noise = np.sqrt(var / 2) * (
            rng.standard_normal(N_FREQ) + 1j * rng.standard_normal(N_FREQ)
        )
        noise[0] = 0.0
        noise[-1] = np.sqrt(var[-1]) * rng.standard_normal()
        data[ch] = signal[ch] + noise
    observed = enchilada.L1Data(data, **kwargs)
    check_block(
        block,
        observed,
        num_cycles=2,
        initial_block_state=block.initial_block_state(observed),
        initial_noise_covariance=cov,
    )
    assert block.turn == 2
    for seg in block.table.segments:
        cold = tmp_path / f"segment_{seg.index:05d}" / "cold"
        assert len(list(cold.glob("turn_*.npz"))) == 2
    return block


@pytest.mark.slow
def test_check_block(tmp_path):
    config = UCBBlockConfig(
        band_bins=(2920, 2984), steps_per_turn=10, ntemps=2, hottest_temperature=10.0
    )
    _check_block_run(tmp_path, _rows(qs=(2952.3,)), config)


@pytest.mark.slow
def test_check_block_three_segments(tmp_path):
    # Cross-segment subtraction on the real path: three sources, one per segment.
    block = _check_block_run(tmp_path, _rows(), _config())
    assert len(block.table.segments) >= 3
    assert all(len(block.owned_start[s.index]) == 1 for s in block.table.segments)


def test_refuses_an_outdir_that_holds_a_run(tmp_path, fake_runner):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="already holds a run"):
        UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    assert (tmp_path / "config.json").read_text() == "{}"
    assert not (tmp_path / "segment_table.json").exists()


def test_refuses_an_outdir_that_holds_archived_turns(tmp_path, fake_runner):
    cold = tmp_path / "segment_00000" / "cold"
    cold.mkdir(parents=True)
    (cold / "turn_00000.npz").write_bytes(b"")
    with pytest.raises(ValueError, match="already holds a run"):
        UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    assert not (tmp_path / "config.json").exists()


def test_accepts_an_empty_existing_outdir(tmp_path, fake_runner):
    UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    assert (tmp_path / "config.json").exists()


def test_rejected_reference_leaves_no_files(tmp_path, fake_runner):
    with pytest.raises(ValueError, match="covariance channels"):
        UCBBlock(
            grid=_grid(),
            start_sources=_rows(),
            outdir=tmp_path / "run",
            config=_config(),
            reference_covariance=_cov(("A", "E", "X")),
        )
    assert not (tmp_path / "run").exists()


def test_initial_block_state_checks_the_domain(tmp_path, fake_runner):
    block = UCBBlock(
        grid=_grid(),
        start_sources=_rows(),
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    with pytest.raises(ValueError, match="frequency-domain"):
        block.initial_block_state(_grid(domain="time"))


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
