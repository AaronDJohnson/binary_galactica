import re

import numpy as np
import pytest

from binary_galactica.glass import ae_noise, template_on_window
from binary_galactica.segments import (
    ChainArchive,
    Segment,
    SegmentRunner,
    UCBBlockConfig,
    glass_ladder,
)

from .support import T_SHORT, native_source, rj_impulse

SEG = Segment(index=0, lo=2920, hi=2984, pad=4, bw=64, k_max=3, owned=(0,))
SOURCE = native_source(2949.12)


def _window(seed=0, offset=0.0):
    n = SEG.n_bins
    f = (SEG.bin_lo + np.arange(n)) / T_SHORT
    c = np.stack([0.5 * T_SHORT * ae_noise(f)] * 2)
    rng = np.random.default_rng(seed)
    noise = np.sqrt(c / 2) * (rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape))
    h = np.asarray(template_on_window(SOURCE, T=T_SHORT, bin_lo=SEG.bin_lo, n_bins=n,
                                      BW=SEG.bw, backend="onfly"))
    return noise + h + offset * np.sqrt(c), c


def _config(**kw):
    base = {"steps_per_turn": 50, "ntemps": 2, "hottest_temperature": 10.0}
    base.update(kw)
    return UCBBlockConfig(**base)


def test_glass_ladder():
    ladder = glass_ladder(12)
    np.testing.assert_allclose(ladder[:11], 1.2 ** np.arange(11), rtol=1e-15)
    assert ladder[-1] == 1e12
    assert glass_ladder(1).tolist() == [1.0]


def test_archive_round_trip(tmp_path):
    archive = ChainArchive(tmp_path)
    archive.append(0, samples=np.ones((3, 16)), k=np.full(3, 2), lnlike=np.zeros(3))
    archive.append(1, samples=np.zeros((2, 16)), k=np.full(2, 1), lnlike=np.ones(2))
    chain = archive.load()
    assert chain["samples"].shape == (5, 16)
    assert chain["turn"].tolist() == [0, 0, 0, 1, 1]
    assert chain["k"].tolist() == [2, 2, 2, 1, 1]


def test_archive_refuses_a_used_directory(tmp_path):
    ChainArchive(tmp_path).append(0, samples=np.ones((1, 8)), k=[1], lnlike=[0.0])
    with pytest.raises(ValueError, match=re.escape(str(tmp_path))):
        ChainArchive(tmp_path)


def test_fixed_count_smoke(tmp_path):
    rj_impulse()
    r, c = _window()
    runner = SegmentRunner(SEG, SOURCE[None], T=T_SHORT, t0=0.0,
                           config=_config(birth_death=False), outdir=tmp_path, seed=1,
                           reference=c)
    runner.advance(r, c)
    assert runner.ks.tolist() == [1, 1]
    assert np.all(np.isfinite(runner.cold_sources())) and runner.cold_sources().shape == (1, 8)
    assert runner.bank.count(0) == 1 and runner.bank.count(1) == 1
    chain = runner.archive.load()
    assert chain["samples"].shape == (50, 8)
    assert np.all(chain["k"] == 1) and np.all(np.isfinite(chain["lnlike"]))
    assert not list((tmp_path / "impulse").glob("chain_*"))
    assert runner.sampler.short_chain is None


def test_positions_and_counts_carry_over_between_turns(tmp_path):
    rj_impulse()
    r, c = _window()
    runner = SegmentRunner(SEG, SOURCE[None], T=T_SHORT, t0=0.0,
                           config=_config(steps_per_turn=20), outdir=tmp_path, seed=5,
                           reference=c)
    calls = []
    original = runner.sampler.sample

    def spy(initial_position, num_iterations, thin=1, initial_k=None):
        calls.append((np.array(initial_position), np.array(initial_k)))
        return original(initial_position, num_iterations, thin=thin, initial_k=initial_k)

    runner.sampler.sample = spy
    runner.advance(r, c)
    width = 8 * runner.k_max
    final = np.asarray(runner.sampler.state.positions)
    np.testing.assert_array_equal(runner.positions, final[:, :width])
    np.testing.assert_array_equal(runner.ks, final[:, width])
    assert not np.array_equal(runner.positions, calls[0][0], equal_nan=True)
    after_x, after_k = runner.positions, runner.ks
    runner.advance(r, c)
    np.testing.assert_array_equal(calls[1][0], after_x)
    np.testing.assert_array_equal(calls[1][1], after_k)


def test_residual_change_recomputes_likelihoods(tmp_path):
    rj_impulse()
    r, c = _window()
    runner = SegmentRunner(SEG, SOURCE[None], T=T_SHORT, t0=0.0,
                           config=_config(steps_per_turn=5), outdir=tmp_path, seed=2,
                           reference=c)
    runner.advance(r, c)
    r2, _ = _window(offset=50.0)
    runner.advance(r2, c)
    expected = runner.likelihood(runner.positions, runner.ks)
    np.testing.assert_allclose(runner.sampler.state.lnlikes, expected, rtol=1e-12)


def test_prior_only_count_is_uniform(tmp_path):
    rj_impulse()
    stats = pytest.importorskip("scipy.stats")
    cfg = _config(ntemps=1, steps_per_turn=20_000, thin=10)
    runner = SegmentRunner(SEG, np.empty((0, 8)), T=T_SHORT, t0=0.0, config=cfg,
                           outdir=tmp_path, seed=3, prior_only=True)
    runner.advance()
    chain = runner.archive.load()
    freq = np.bincount(chain["k"], minlength=4) / chain["k"].size
    np.testing.assert_allclose(freq, 0.25, atol=0.05)
    f0 = chain["samples"][::10, 0][chain["k"][::10] >= 1]
    p = stats.kstest((f0 - SEG.bin_lo) / SEG.n_bins, "uniform").pvalue
    assert p > 1e-3
    # f0*T in bins, like SEG.lo/hi: the padding holds 8/72 ~ 0.11 of the band.
    assert 0.04 < np.mean((f0 < SEG.lo) | (f0 >= SEG.hi)) < 0.2


def test_fixed_count_needs_sources(tmp_path):
    with pytest.raises(ValueError, match="start source"):
        SegmentRunner(SEG, np.empty((0, 8)), T=T_SHORT, t0=0.0,
                      config=_config(birth_death=False), outdir=tmp_path, seed=4,
                      prior_only=True)
