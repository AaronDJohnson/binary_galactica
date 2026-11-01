import math

import numpy as np
import pytest

from binary_galactica.segments import (
    SegmentTable,
    UCBBlockConfig,
    build_segment_table,
    doppler_bins,
    padding_bins,
)

from .support import native_source

T = 31_536_000.0
N_FREQ = 3_153_601


def _sources(qs, lnA=-50.0, fdot_T2=0.0):
    return np.array([native_source(q, lnA=lnA, fdot_T2=fdot_T2) for q in qs])


def _check_rules(table):
    segs = table.segments
    for a, b in zip(segs[:-1], segs[1:]):
        assert a.hi == b.lo
    for s, seg in enumerate(segs):
        left = segs[s - 1].pad if s > 0 else 0
        right = segs[s + 1].pad if s < len(segs) - 1 else 0
        assert seg.width >= max(seg.pad, left + right)
        assert seg.bw & (seg.bw - 1) == 0 and seg.bw <= seg.n_bins
    for a, b in zip(segs[:-2], segs[2:]):
        assert a.bin_hi <= b.bin_lo


def test_doppler_and_padding_rule():
    f, fdot = 4e-3, 2e-16
    expected = 2 * T * ((4 + 2 * np.pi * f * 1.49597870700e11 / 299792458.0)
                        / 3.15581497632e7 + fdot * T)
    assert doppler_bins(f, fdot, T) == pytest.approx(expected, rel=1e-12)
    assert padding_bins(f, -fdot, T) == math.ceil(2 * expected)


def test_sparse_sources_fill_n_target():
    qs = 120_000 + 80.0 * np.arange(205) + 0.37
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    _check_rules(table)
    counts = [len(seg.owned) for seg in table.segments]
    assert all(c == 10 for c in counts[:-1]) and 0 < counts[-1] <= 10
    assert sum(counts) == len(qs)


def test_ownership_is_half_open_and_unique():
    qs = 120_000 + 80.0 * np.arange(60) + 0.37
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    owners = table.owners(qs)
    assert np.all(owners >= 0)
    for i, q in enumerate(qs):
        seg = table.segments[owners[i]]
        assert seg.lo <= q < seg.hi and i in seg.owned
    assert table.owners(np.array([table.segments[1].lo]))[0] == 1


def test_dense_cluster_forces_merges():
    rng = np.random.default_rng(3)
    qs = np.sort(120_000 + rng.uniform(0, 40, 30))
    cfg = UCBBlockConfig(band_bins=(119_800, 120_300))
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    _check_rules(table)
    assert sum(len(s.owned) for s in table.segments) == 30
    assert max(len(s.owned) for s in table.segments) > 10


def test_padding_uses_owned_fdot():
    qs = 120_000 + 80.0 * np.arange(20) + 0.37
    fast = _sources(qs, fdot_T2=0.5)
    table = build_segment_table(fast, T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    seg = table.segments[0]
    assert seg.pad == padding_bins(seg.hi / T, 0.5 / T**2, T)


def test_k_max_and_prior_band():
    qs = 120_000 + 80.0 * np.arange(12) + 0.37
    cfg = UCBBlockConfig(n_target=12, k_max_floor=10)
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    seg = table.segments[0]
    assert seg.k_max == 24
    prior = seg.prior(T)
    assert prior.lo[0] == pytest.approx(seg.bin_lo, abs=1e-6)
    assert prior.hi[0] == pytest.approx(seg.bin_hi, abs=1e-6)


def test_explicit_band_lists_outside_sources():
    qs = np.array([119_000.5, 120_010.5, 120_030.5, 125_000.5])
    cfg = UCBBlockConfig(band_bins=(120_000, 120_400))
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    assert table.outside == (0, 3)
    assert table.segments[0].lo == 120_000 and table.segments[-1].hi == 120_400


def test_window_touching_dc_raises():
    with pytest.raises(ValueError, match="grid"):
        build_segment_table(_sources([3.5]), T=T, n_freq=N_FREQ, config=UCBBlockConfig())


def test_source_outside_prior_raises():
    with pytest.raises(ValueError, match="prior"):
        build_segment_table(
            _sources([120_000.5], lnA=-60.0),
            T=T,
            n_freq=N_FREQ,
            config=UCBBlockConfig(band_bins=(120_000, 120_400)),
        )


def test_empty_start_set_needs_band():
    with pytest.raises(ValueError, match="band_bins"):
        build_segment_table(np.empty((0, 8)), T=T, n_freq=N_FREQ, config=UCBBlockConfig())


def test_json_round_trip(tmp_path):
    qs = 120_000 + 80.0 * np.arange(30) + 0.37
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    path = tmp_path / "segment_table.json"
    table.write(path)
    assert SegmentTable.read(path) == table
