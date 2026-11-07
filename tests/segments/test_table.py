"""The segment table fixes padding, ownership and merging."""

import math

import numpy as np
import pytest

from binary_galactica.segments import (
    Segment,
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
    for a, b in zip(segs[:-1], segs[1:], strict=True):
        assert a.hi == b.lo
    for s, seg in enumerate(segs):
        left = segs[s - 1].pad if s > 0 else 0
        right = segs[s + 1].pad if s < len(segs) - 1 else 0
        assert seg.width >= max(seg.pad, left + right)
        assert seg.bw & (seg.bw - 1) == 0 and seg.bw <= seg.n_bins
    for a, b in zip(segs[:-2], segs[2:], strict=True):
        assert a.bin_hi <= b.bin_lo


def test_doppler_and_padding_rule():
    f, fdot = 4e-3, 2e-16
    expected = (
        2
        * T
        * (
            (4 + 2 * np.pi * f * 1.49597870700e11 / 299792458.0) / 3.15581497632e7
            + fdot * T
        )
    )
    assert doppler_bins(f, fdot, T) == pytest.approx(expected, rel=1e-12)
    assert padding_bins(f, -fdot, T) == math.ceil(2 * expected)


def test_sparse_sources_fill_n_target():
    qs = 120_000 + 80.0 * np.arange(205) + 0.37
    table = build_segment_table(
        _sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
    )
    _check_rules(table)
    counts = [len(seg.owned) for seg in table.segments]
    assert all(c == 10 for c in counts[:-1]) and 0 < counts[-1] <= 10
    assert sum(counts) == len(qs)


def test_ownership_is_half_open_and_unique():
    qs = 120_000 + 80.0 * np.arange(60) + 0.37
    table = build_segment_table(
        _sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
    )
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
    src = _sources(qs, fdot_T2=0.0)
    src[:10, 7] = 0.5  # only the first segment's sources drift fast
    table = build_segment_table(src, T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    first, second = table.segments[0], table.segments[1]
    assert first.pad == padding_bins(first.hi / T, 0.5 / T**2, T)
    assert second.pad == padding_bins(second.hi / T, 0.0, T)
    assert second.pad < first.pad or second.hi != first.hi


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
        build_segment_table(
            _sources([3.5]), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
        )


def test_source_outside_prior_raises():
    with pytest.raises(ValueError, match="prior"):
        build_segment_table(
            _sources([120_000.5], lnA=-60.0),
            T=T,
            n_freq=N_FREQ,
            config=UCBBlockConfig(),
        )


def test_lone_source_builds_one_segment():
    table = build_segment_table(
        _sources([120_000.5]), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
    )
    _check_rules(table)
    assert len(table.segments) == 1 and table.segments[0].owned == (0,)
    assert table.segments[0].width >= table.segments[0].pad


def test_too_narrow_explicit_band_raises():
    with pytest.raises(ValueError, match="band_bins"):
        build_segment_table(
            _sources([120_000.5]),
            T=T,
            n_freq=N_FREQ,
            config=UCBBlockConfig(band_bins=(120_000, 120_016)),
        )


def test_empty_start_set_needs_band():
    with pytest.raises(ValueError, match="band_bins"):
        build_segment_table(
            np.empty((0, 8)), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
        )


def test_json_round_trip(tmp_path):
    qs = 120_000 + 80.0 * np.arange(30) + 0.37
    table = build_segment_table(
        _sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
    )
    path = tmp_path / "segment_table.json"
    table.write(path)
    assert SegmentTable.read(path) == table


def _pinned_start_set():
    qs = [120_000.37 + 80.0 * k for k in range(25)]
    qs += [124_000.11 + 1.3 * k for k in range(30)]
    qs += [150_000.5, 185_000.25, 300_000.75]
    qs += [200_000.6 + 2_000.0 * k for k in range(12)]
    fdots = [0.5 if k % 7 == 0 else 0.0 for k in range(len(qs))]
    return np.array(
        [native_source(q, lnA=-50.0, fdot_T2=f) for q, f in zip(qs, fdots, strict=True)]
    )


def _rows(table):
    return [
        (s.index, s.lo, s.hi, s.pad, s.bw, s.k_max, s.owned) for s in table.segments
    ]


def test_default_tiling_is_pinned():
    # Recorded from the tiling before skip_gaps and max_interior_bins existed.
    x = _pinned_start_set()
    table = build_segment_table(x, T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    assert table.outside == ()
    assert _rows(table) == [
        (0, 120_000, 120_761, 66, 256, 20, tuple(range(0, 10))),
        (1, 120_761, 121_561, 67, 256, 20, tuple(range(10, 20))),
        (2, 121_561, 124_038, 68, 256, 70, tuple(range(20, 55))),
        (3, 124_038, 215_001, 104, 512, 20, (55, 56, *range(58, 66))),
        (4, 215_001, 300_001, 136, 512, 10, (57, *range(66, 70))),
    ]
    cfg = UCBBlockConfig(band_bins=(120_500, 190_000))
    table = build_segment_table(x, T=T, n_freq=N_FREQ, config=cfg)
    assert table.outside == (*range(0, 7), *range(57, 70))
    assert _rows(table) == [
        (0, 120_500, 121_321, 67, 256, 20, tuple(range(7, 17))),
        (1, 121_321, 124_035, 68, 256, 70, tuple(range(17, 52))),
        (2, 124_035, 190_000, 94, 512, 10, tuple(range(52, 57))),
    ]


def _clusters(table):
    """Return the runs of contiguous segments."""
    runs = [[table.segments[0]]]
    for a, b in zip(table.segments[:-1], table.segments[1:], strict=True):
        if a.hi == b.lo:
            runs[-1].append(b)
        else:
            runs.append([b])
    return runs


def _check_cluster_rules(table, q):
    segs = table.segments
    assert [s.index for s in segs] == list(range(len(segs)))
    for a, b in zip(segs[:-1], segs[1:], strict=True):
        assert a.hi <= b.lo
        if a.hi < b.lo:  # a gap: the padded windows must not overlap
            assert a.bin_hi <= b.bin_lo
    for run in _clusters(table):
        for s, seg in enumerate(run):
            left = run[s - 1].pad if s > 0 else 0
            right = run[s + 1].pad if s < len(run) - 1 else 0
            assert seg.width >= max(seg.pad, left + right)
    for a, b in zip(segs[:-2], segs[2:], strict=True):
        assert a.bin_hi <= b.bin_lo
    owned = sorted(i for seg in segs for i in seg.owned)
    inside = sorted(set(range(len(q))) - set(table.outside))
    assert owned == inside  # every in-band source exactly once
    for seg in segs:
        assert all(seg.lo <= q[i] < seg.hi for i in seg.owned)


def test_skip_gaps_without_wide_gaps_matches_the_default():
    qs = 120_000 + 80.0 * np.arange(60) + 0.37
    src = _sources(qs)
    src[::9, 7] = 0.5
    default = build_segment_table(src, T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    skipped = build_segment_table(
        src, T=T, n_freq=N_FREQ, config=UCBBlockConfig(skip_gaps=True)
    )
    assert skipped == default


def _gapped_sources():
    """Two groups and one lone source far above them, with gaps far wider than pads."""
    qs = np.concatenate(
        [
            120_000.37 + 80.0 * np.arange(30),
            140_000.37 + 50.0 * np.arange(20),
            [200_000.5],
        ]
    )
    return qs, _sources(qs)


def _span(first, last):
    m_first, m_last = padding_bins(first / T, 0.0, T), padding_bins(last / T, 0.0, T)
    return int(np.floor(first)) - m_first, int(np.floor(last)) + 1 + m_last


def test_skip_gaps_leaves_wide_gaps_unsampled():
    qs, src = _gapped_sources()
    band = (110_000, 260_000)
    cfg = UCBBlockConfig(band_bins=band, skip_gaps=True)
    table = build_segment_table(src, T=T, n_freq=N_FREQ, config=cfg)
    _check_cluster_rules(table, qs)
    runs = _clusters(table)
    spans = [(run[0].lo, run[-1].hi) for run in runs]
    assert spans == [_span(qs[0], qs[29]), _span(qs[30], qs[49]), _span(qs[50], qs[50])]
    gaps = [(a[1], b[0]) for a, b in zip(spans[:-1], spans[1:], strict=True)]
    for lo, hi in [(band[0], spans[0][0]), *gaps, (spans[-1][1], band[1])]:
        assert not np.any((qs >= lo) & (qs < hi))
    covered = sum(hi - lo for lo, hi in spans)
    assert covered < 0.1 * (band[1] - band[0])
    lone = runs[-1]
    assert len(lone) == 1 and lone[0].owned == (50,)
    assert lone[0].lo <= qs[50] < lone[0].hi
    # Without skip_gaps the same start set tiles the whole band.
    default = build_segment_table(
        src, T=T, n_freq=N_FREQ, config=UCBBlockConfig(band_bins=band)
    )
    assert len(_clusters(default)) == 1


def test_owners_are_minus_one_in_a_gap():
    qs, src = _gapped_sources()
    cfg = UCBBlockConfig(band_bins=(110_000, 260_000), skip_gaps=True)
    table = build_segment_table(src, T=T, n_freq=N_FREQ, config=cfg)
    first, second = _clusters(table)[:2]
    below, above = first[-1], second[0]
    probes = [
        below.hi - 0.5,
        below.hi,
        0.5 * (below.hi + above.lo),
        above.lo - 0.5,
        above.lo,
        table.segments[0].lo - 1.0,
        table.segments[-1].hi,
    ]
    assert table.owners(probes).tolist() == [
        below.index,
        -1,
        -1,
        -1,
        above.index,
        -1,
        -1,
    ]
    owners = table.owners(qs)
    assert all(i in table.segments[o].owned for i, o in enumerate(owners))


def test_close_clusters_merge_when_their_windows_overlap():
    q_a = 120_000.5
    m_a = q_b = padding_bins(q_a / T, 0.0, T)
    for _ in range(3):
        m_b = padding_bins(q_b / T, 0.0, T)
        q_b = q_a + 2 * (m_a + m_b) + 0.01
    assert padding_bins(q_b / T, 0.0, T) == m_b
    cfg = UCBBlockConfig(band_bins=(110_000, 130_000), skip_gaps=True)
    # Just past the split threshold 2 (m_a + m_b), the two lone-source
    # clusters' padded windows still overlap, so they merge into one.
    near = np.array([q_a, q_b])
    table = build_segment_table(_sources(near), T=T, n_freq=N_FREQ, config=cfg)
    _check_cluster_rules(table, near)
    assert len(_clusters(table)) == 1
    lo, hi = _span(near[0], near[1])
    assert (table.segments[0].lo, table.segments[-1].hi) == (lo, hi)
    # Well past it they stay apart.
    far = np.array([q_a, q_b + 50.0])
    table = build_segment_table(_sources(far), T=T, n_freq=N_FREQ, config=cfg)
    _check_cluster_rules(table, far)
    assert len(_clusters(table)) == 2


def test_cluster_narrower_than_its_padding_is_widened():
    # The middle source drifts fast (20 mHz admits a large fdot), so the
    # cluster's padding exceeds the margins of its edge sources.
    qs = np.array([630_000.5, 630_001.5, 630_002.5])
    src = _sources(qs)
    src[1, 7] = 100.0
    cfg = UCBBlockConfig(band_bins=(620_000, 640_000), skip_gaps=True)
    table = build_segment_table(src, T=T, n_freq=N_FREQ, config=cfg)
    _check_cluster_rules(table, qs)
    (seg,) = table.segments
    lo, hi = _span(qs[0], qs[2])
    assert seg.width == seg.pad > hi - lo
    assert abs((seg.lo + seg.hi) - (lo + hi)) <= 1  # widened about its centre
    with pytest.raises(ValueError, match="band_bins"):
        build_segment_table(
            src,
            T=T,
            n_freq=N_FREQ,
            config=UCBBlockConfig(band_bins=(629_800, 630_300), skip_gaps=True),
        )


def test_max_interior_bins_caps_wide_interiors():
    qs = 120_000.37 + 2_000.0 * np.arange(15)
    default = build_segment_table(
        _sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
    )
    assert max(s.width for s in default.segments) > 3_000
    cfg = UCBBlockConfig(max_interior_bins=3_000)
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    _check_rules(table)
    assert all(cfg.min_interior_bins <= s.width <= 3_000 for s in table.segments)
    assert (table.segments[0].lo, table.segments[-1].hi) == (
        default.segments[0].lo,
        default.segments[-1].hi,
    )
    assert sorted(i for s in table.segments for i in s.owned) == list(range(15))


def test_padding_rule_overrides_a_tight_cap():
    # Pads here are about 64 bins, above the cap, so the merge pass re-widens
    # every piece past it.
    qs = 120_000.37 + 2_000.0 * np.arange(15)
    cfg = UCBBlockConfig(max_interior_bins=50)
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    _check_rules(table)
    segs = table.segments
    assert len(segs) > 100
    assert all(seg.width > 50 and seg.pad > 50 for seg in segs)


def test_capped_pieces_keep_the_minimum_width():
    from .support import N_FREQ as N_SHORT
    from .support import T_SHORT

    src = np.array([native_source(3_010.5)])
    # 33 bins under a cap of 16: three pieces would be 11 bins wide, under
    # min_interior_bins, so the interior splits into two pieces instead.
    cfg = UCBBlockConfig(band_bins=(3_000, 3_033), max_interior_bins=16)
    table = build_segment_table(src, T=T_SHORT, n_freq=N_SHORT, config=cfg)
    assert [s.width for s in table.segments] == [16, 17]
    cfg = UCBBlockConfig(band_bins=(3_000, 3_033), max_interior_bins=20)
    table = build_segment_table(src, T=T_SHORT, n_freq=N_SHORT, config=cfg)
    assert [s.width for s in table.segments] == [16, 17]


def test_max_interior_bins_with_skip_gaps():
    qs = np.concatenate([120_000.37 + 200.0 * np.arange(30), [160_000.5]])
    cfg = UCBBlockConfig(band_bins=(110_000, 170_000), skip_gaps=True)
    plain = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    assert max(s.width for s in plain.segments) > 700
    cfg = UCBBlockConfig(
        band_bins=(110_000, 170_000), skip_gaps=True, max_interior_bins=700
    )
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    _check_cluster_rules(table, qs)
    assert all(s.width <= 700 for s in table.segments)
    assert len(table.segments) > len(plain.segments)
    spans = [(r[0].lo, r[-1].hi) for r in _clusters(table)]
    assert spans == [(r[0].lo, r[-1].hi) for r in _clusters(plain)]
    assert len(spans) == 2


def test_skip_gaps_needs_a_start_source_in_the_band():
    cfg = UCBBlockConfig(band_bins=(120_000, 120_400), skip_gaps=True)
    for src in (np.empty((0, 8)), _sources([119_000.5, 125_000.5])):
        with pytest.raises(ValueError, match="skip_gaps"):
            build_segment_table(src, T=T, n_freq=N_FREQ, config=cfg)


def test_segment_owns_scalars_and_arrays():
    import numpy as np

    seg = Segment(index=0, lo=10, hi=20, pad=3, bw=16, k_max=10, owned=())
    assert (
        seg.owns(10.0) and seg.owns(19.9) and not seg.owns(20.0) and not seg.owns(9.9)
    )
    assert list(seg.owns(np.array([9.0, 10.0, 15.0, 20.0]))) == [
        False,
        True,
        True,
        False,
    ]
