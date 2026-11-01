import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from binary_galactica.glass import ae_noise
from binary_galactica.segments import UCBBlock, UCBBlockConfig, render_sources

from .support import DT, N_FREQ, N_TIME, T_SHORT, ldc_rows, rj_impulse

block_module = importlib.import_module("binary_galactica.segments.block")
CHANNELS = ("A", "E", "T")
QS = (2940.3, 2952.3, 2964.3)


class _Data(SimpleNamespace):
    def block_state(self, template=None, *, covariance_matrix=None, parameters=None):
        return SimpleNamespace(template=template, parameters=parameters)


def _grid(channels=CHANNELS, domain="frequency", data=None, t0=0.0):
    data = data or {ch: np.zeros(N_FREQ, dtype=complex) for ch in channels}
    return _Data(channel_names=channels, num_time_samples=N_TIME, sample_rate_hz=1 / DT,
                 start_time_gps=t0, data_domain=domain, channel_data=data)


def _cov(channels=CHANNELS):
    f = np.arange(N_FREQ) / T_SHORT
    f[0] = f[1]
    m = np.zeros((N_FREQ, len(channels), len(channels)))
    for i in range(len(channels)):
        m[:, i, i] = 0.5 * T_SHORT * ae_noise(f)
    mask = np.ones(N_FREQ, dtype=bool)
    mask[0] = False
    return SimpleNamespace(data_domain="frequency", covariance_matrix=m,
                           channel_names=channels, active_mask=mask)


def _rows(qs=QS, amplitudes=None):
    amplitudes = amplitudes or [np.exp(-48.0)] * len(qs)
    rows = np.concatenate([ldc_rows(Frequency=q / T_SHORT, Amplitude=a)
                           for q, a in zip(qs, amplitudes)])
    return rows


def _config(**kw):
    base = {"band_bins": (2930, 2980), "n_target": 1, "min_interior_bins": 8,
            "steps_per_turn": 10, "ntemps": 2, "hottest_temperature": 10.0}
    base.update(kw)
    return UCBBlockConfig(**base)


def _render(sources, seg, n_channels=3):
    return render_sources(sources, T=T_SHORT, t0=0.0, BW=seg.bw, n_channels=n_channels,
                          n_freq=N_FREQ)


class _Noise:
    def psd(self, f):
        return ae_noise(f)


class _FakeRunner:
    """Records each advance; ``move`` replaces and ``extra`` appends cold sources."""

    calls: list = []
    extra = None
    move = None

    def __init__(self, segment, start, **kwargs):
        self.segment = segment
        self.start = np.array(start).reshape(-1, 8)

    def advance(self, residual, c):
        _FakeRunner.calls.append((self.segment.index, np.array(residual)))

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
    monkeypatch.setattr(block_module, "SegmentRunner", _FakeRunner)
    return _FakeRunner


def test_rejects_unsupported_inputs(tmp_path, fake_runner):
    with pytest.raises(ValueError, match="channels"):
        UCBBlock(grid=_grid(("X", "Y", "Z")), start_sources=_rows(), outdir=tmp_path,
                 config=_config())
    block = UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="frequency"):
        block.sample(_grid(domain="time"), _cov(), None, rng=rng)
    with pytest.raises(TypeError, match="DataCovariance"):
        block.sample(_grid(), SimpleNamespace(data_domain="time"), None, rng=rng)
    with pytest.raises(ValueError, match="start time"):
        block.sample(_grid(t0=100.0), _cov(), None, rng=rng)


def test_rejects_bad_covariances(tmp_path, fake_runner):
    block = UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
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
            render_sources(source, T=T_SHORT, t0=0.0, BW=16, n_channels=3, n_freq=N_FREQ,
                           fmt=fmt)


def test_writes_run_files(tmp_path, fake_runner):
    UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    for name in ("config.json", "segment_table.json", "dropped_start_sources.json"):
        assert (tmp_path / name).exists()
    assert UCBBlockConfig.read(tmp_path / "config.json") == _config()


def test_drops_weak_start_sources(tmp_path, fake_runner):
    rows = _rows(amplitudes=[np.exp(-48.0), 1e-23, np.exp(-48.0)])
    block = UCBBlock(grid=_grid(), start_sources=rows, outdir=tmp_path, config=_config(),
                     reference_covariance=_cov())
    dropped = json.loads((tmp_path / "dropped_start_sources.json").read_text())
    assert [d["index"] for d in dropped] == [1]
    assert dropped[0]["reason"] == "snr below cut" and dropped[0]["snr"] < 5
    assert block.owned_start[1] == []


def test_sweep_order_and_conditional_residual(tmp_path, fake_runner):
    # Zero data: segment 1 must then see exactly minus its neighbours' sources.
    block = UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))
    assert [s for s, _ in fake_runner.calls] == [0, 2, 1]
    seg = block.table.segments[1]
    others = sum(
        render_sources(block.start_native[[i]], T=T_SHORT, t0=0.0,
                       BW=block.table.segments[i].bw, n_channels=3, n_freq=N_FREQ
                       ).window(seg.bin_lo, seg.bin_hi)[:2]
        for i in (0, 2)
    )
    assert np.max(np.abs(others)) > 0
    scale = np.max(np.abs(others))
    np.testing.assert_allclose(fake_runner.calls[2][1], -others, rtol=1e-12,
                               atol=1e-12 * scale)


def test_sweep_subtracts_updated_neighbours(tmp_path, fake_runner):
    # The even segments move their owned source towards segment 1 (staying in
    # their interior) and brighten it. Segment 1 advances after them, so it
    # must see the data minus the UPDATED sources, not the start set.
    block = UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
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
    stale = sum(_render(block.start_native[[i]], segs[i]).window(lo, hi)[:2] for i in (0, 2))
    scale = np.max(np.abs(updated))
    assert np.max(np.abs(updated - stale)) > 0.1 * scale
    rng = np.random.default_rng(5)
    data = {ch: scale * (rng.standard_normal(N_FREQ) + 1j * rng.standard_normal(N_FREQ))
            for ch in CHANNELS}
    r = np.stack([data["A"], data["E"]])

    state = block.sample(_grid(data=data), _cov(), None, rng=np.random.default_rng(1))
    assert [s for s, _ in fake_runner.calls] == [0, 2, 1]
    np.testing.assert_allclose(fake_runner.calls[2][1], r[:, lo:hi] - updated, rtol=1e-12,
                               atol=1e-12 * scale)

    published = [moved[0], block.start_native[[1]], moved[2]]
    expected = sum(_render(x, s).window(0, N_FREQ) for x, s in zip(published, segs))
    for j, ch in enumerate(CHANNELS):
        np.testing.assert_allclose(state.template[ch], expected[j], rtol=1e-12,
                                   atol=1e-12 * np.max(np.abs(expected)))
    np.testing.assert_array_equal(state.parameters["segments"][2]["native"], moved[2])


def test_rejects_a_ledger_out_of_step(tmp_path, fake_runner):
    block = UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config(),
                     reference_covariance=_cov())
    rng = np.random.default_rng(4)
    state0 = block.initial_block_state(_grid())
    assert state0.parameters["turn"] == 0
    state1 = block.sample(_grid(), _cov(), state0, rng=rng)
    assert state1.parameters["turn"] == 1 == block.turn
    with pytest.raises(ValueError, match="turn 0 but the block is at turn 1"):
        block.sample(_grid(), _cov(), state0, rng=rng)
    assert block.sample(_grid(), _cov(), state1, rng=rng).parameters["turn"] == 2


def test_initial_block_state_needs_the_reference(tmp_path, fake_runner):
    block = UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    with pytest.raises(ValueError, match="reference_covariance"):
        block.initial_block_state(_grid())


def test_ae_layout(tmp_path, fake_runner):
    ae = ("A", "E")
    block = UCBBlock(grid=_grid(ae), start_sources=_rows(), outdir=tmp_path, config=_config(),
                     reference_covariance=_cov(ae))
    segs = block.table.segments
    initial = block.initial_block_state(_grid(ae))
    state = block.sample(_grid(ae), _cov(ae), initial, rng=np.random.default_rng(6))
    assert [s for s, _ in fake_runner.calls] == [0, 2, 1]
    expected = sum(_render(block.start_native[[i]], segs[i]).window(0, N_FREQ)
                   for i in range(3))
    for published in (initial, state):
        assert set(published.template) == set(ae)
        for j, ch in enumerate(ae):
            np.testing.assert_allclose(published.template[ch], expected[j], rtol=1e-12,
                                       atol=1e-12 * np.max(np.abs(expected)))


def test_publishes_owned_sources_only(tmp_path, fake_runner):
    def padding_source(segment):
        src = block.start_native[segment.owned[0]].copy()
        src[0] = segment.bin_lo + 0.5
        return src[None]

    block = UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    fake_runner.extra = padding_source
    state = block.sample(_grid(), _cov(), None, rng=np.random.default_rng(2))
    counts = [s["count"] for s in state.parameters["segments"]]
    assert counts == [1, 1, 1]
    assert set(state.template) == set(CHANNELS)
    first = state.parameters["segments"][0]
    np.testing.assert_allclose(first["ldc"]["Frequency"], QS[0] / T_SHORT, rtol=1e-12)


def test_initial_block_state_renders_the_start_set(tmp_path, fake_runner):
    block = UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config(),
                     reference_covariance=_cov())
    state = block.initial_block_state(_grid())
    expected = sum(
        render_sources(block.start_native[[i]], T=T_SHORT, t0=0.0,
                       BW=block.table.segments[i].bw, n_channels=3, n_freq=N_FREQ
                       ).window(0, N_FREQ)
        for i in range(3)
    )
    for j, ch in enumerate(CHANNELS):
        np.testing.assert_allclose(state.template[ch], expected[j], rtol=1e-12, atol=1e-30)


def _check_block_run(tmp_path, rows, config):
    """Run enchilada's check_block for two cycles with the real segment runners."""
    enchilada = pytest.importorskip("enchilada")
    from enchilada.testing import check_block

    rj_impulse()
    kwargs = {"sample_rate_hz": 1 / DT, "tdi_generation": "1.5",
              "physical_observable": "fractional_frequency", "channel_names": CHANNELS,
              "num_time_samples": N_TIME, "data_domain": "frequency"}
    grid = enchilada.L1Data({ch: np.zeros(N_FREQ, dtype=complex) for ch in CHANNELS},
                            **kwargs)
    cov = enchilada.DataCovariance.from_psd(grid, _Noise())
    block = UCBBlock(grid=grid, start_sources=rows, outdir=tmp_path, config=config,
                     reference_covariance=cov)
    signal = block.initial_block_state(grid).template
    rng = np.random.default_rng(3)
    data = {}
    for i, ch in enumerate(CHANNELS):
        var = cov.covariance_matrix[:, i, i].real
        noise = np.sqrt(var / 2) * (rng.standard_normal(N_FREQ) + 1j * rng.standard_normal(N_FREQ))
        noise[0] = 0.0
        noise[-1] = np.sqrt(var[-1]) * rng.standard_normal()
        data[ch] = signal[ch] + noise
    observed = enchilada.L1Data(data, **kwargs)
    check_block(block, observed, num_cycles=2,
                initial_block_state=block.initial_block_state(observed),
                initial_noise_covariance=cov)
    assert block.turn == 2
    for seg in block.table.segments:
        cold = tmp_path / f"segment_{seg.index:05d}" / "cold"
        assert len(list(cold.glob("turn_*.npz"))) == 2
    return block


def test_check_block(tmp_path):
    config = UCBBlockConfig(band_bins=(2920, 2984), steps_per_turn=10, ntemps=2,
                            hottest_temperature=10.0)
    _check_block_run(tmp_path, _rows(qs=(2952.3,)), config)


def test_check_block_three_segments(tmp_path):
    # Cross-segment subtraction on the real path: three sources, one per segment.
    block = _check_block_run(tmp_path, _rows(), _config())
    assert len(block.table.segments) >= 3
    assert all(len(block.owned_start[s.index]) == 1 for s in block.table.segments)
