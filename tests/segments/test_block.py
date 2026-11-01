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


def _grid(channels=CHANNELS, domain="frequency", data=None):
    data = data or {ch: np.zeros(N_FREQ, dtype=complex) for ch in channels}
    return _Data(channel_names=channels, num_time_samples=N_TIME, sample_rate_hz=1 / DT,
                 start_time_gps=0.0, data_domain=domain, channel_data=data)


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


class _FakeRunner:
    calls: list = []
    extra = None

    def __init__(self, segment, start, **kwargs):
        self.segment = segment
        self.start = np.array(start).reshape(-1, 8)

    def advance(self, residual, c):
        _FakeRunner.calls.append((self.segment.index, np.array(residual)))

    def cold_sources(self):
        if _FakeRunner.extra is None:
            return self.start.copy()
        return np.concatenate([self.start, _FakeRunner.extra(self.segment)])


@pytest.fixture
def fake_runner(monkeypatch):
    _FakeRunner.calls = []
    _FakeRunner.extra = None
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
    block = UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    state = block.initial_block_state(_grid())
    expected = sum(
        render_sources(block.start_native[[i]], T=T_SHORT, t0=0.0,
                       BW=block.table.segments[i].bw, n_channels=3, n_freq=N_FREQ
                       ).window(0, N_FREQ)
        for i in range(3)
    )
    for j, ch in enumerate(CHANNELS):
        np.testing.assert_allclose(state.template[ch], expected[j], rtol=1e-12, atol=1e-30)


def test_check_block(tmp_path):
    enchilada = pytest.importorskip("enchilada")
    from enchilada.testing import check_block

    rj_impulse()
    kwargs = {"sample_rate_hz": 1 / DT, "tdi_generation": "1.5",
              "physical_observable": "fractional_frequency", "channel_names": CHANNELS,
              "num_time_samples": N_TIME, "data_domain": "frequency"}
    grid = enchilada.L1Data({ch: np.zeros(N_FREQ, dtype=complex) for ch in CHANNELS},
                            **kwargs)

    class _Noise:
        def psd(self, f):
            return ae_noise(f)

    cov = enchilada.DataCovariance.from_psd(grid, _Noise())
    config = UCBBlockConfig(band_bins=(2920, 2984), steps_per_turn=10, ntemps=2,
                            hottest_temperature=10.0)
    rows = _rows(qs=(2952.3,))
    block = UCBBlock(grid=grid, start_sources=rows, outdir=tmp_path, config=config)
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
    assert len(list((tmp_path / "segment_00000" / "cold").glob("turn_*.npz"))) == 2
