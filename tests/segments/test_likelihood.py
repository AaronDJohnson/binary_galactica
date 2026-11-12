"""The segment likelihood agrees with independent templates and inner products."""

import numpy as np
import pytest

from binary_galactica.glass import ae_noise, template_on_window
from binary_galactica.glass.constants import YEAR
from binary_galactica.segments import SegmentLikelihood, quantize_bins

from .support import (
    DT,
    N_FREQ,
    N_TIME,
    SANGRIA,
    T_SHORT,
    native_source,
    requires_sangria,
)

BIN_LO, N_BINS, BW = 2900, 120, 64
SOURCES = np.array([native_source(2949.12), native_source(2960.4, lnA=-48.5)])


def _c():
    f = (BIN_LO + np.arange(N_BINS)) / T_SHORT
    return np.stack([0.5 * T_SHORT * ae_noise(f)] * 2)


def _lik(k_max=3, snr_min=5.0):
    lik = SegmentLikelihood(
        T=T_SHORT,
        t0=0.0,
        bin_lo=BIN_LO,
        n_bins=N_BINS,
        BW=BW,
        k_max=k_max,
        backend="onfly",
        snr_min=snr_min,
    )
    rng = np.random.default_rng(0)
    c = _c()
    r = np.sqrt(c / 2) * (
        rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)
    )
    lik.set_data(r, c)
    lik.set_reference(c)
    return lik, r, c


def _row(sources, k_max=3, fill=np.nan):
    x = np.full(8 * k_max, fill)
    x[: sources.size] = np.ravel(sources)
    return x


def _independent(src):
    return np.asarray(
        template_on_window(
            src, T=T_SHORT, bin_lo=BIN_LO, n_bins=N_BINS, BW=BW, backend="onfly"
        )
    )


def test_quantize_bins_ladder():
    for n in range(1, 400):
        m = quantize_bins(n)
        assert m >= n
        assert m & (m - 1) == 0 or (m % 3 == 0 and (m // 3) & (m // 3 - 1) == 0)
        assert quantize_bins(m) == m


def test_masked_sum_equals_explicit_sum():
    lik, r, c = _lik()
    h = _independent(SOURCES[0]) + _independent(SOURCES[1])
    expected = -np.sum(np.abs(r - h) ** 2 / c)
    got = lik(_row(SOURCES)[None], np.array([2]))[0]
    assert got == pytest.approx(expected, rel=1e-10)
    empty = lik(_row(SOURCES[:0])[None], np.array([0]))[0]
    assert empty == pytest.approx(-np.sum(np.abs(r) ** 2 / c), rel=1e-12)


@pytest.mark.parametrize("fill", [np.nan, np.inf, 1e300, -7.0])
def test_inactive_slots_cannot_leak(fill):
    lik, _, _ = _lik()
    base = lik(_row(SOURCES[:1])[None], np.array([1]))[0]
    other = lik(_row(SOURCES[:1], fill=fill)[None], np.array([1]))[0]
    assert other == base


def test_snr_cut_and_invalid_parameters():
    lik, _, _ = _lik()
    weak = native_source(2949.12, lnA=-54.0)
    assert lik(_row(weak)[None], np.array([1]))[0] == -np.inf
    off, _, _ = _lik(snr_min=0.0)
    assert np.isfinite(off(_row(weak)[None], np.array([1]))[0])
    bad = native_source(2949.12)
    bad[4] = 1.5
    assert lik(_row(bad)[None], np.array([1]))[0] == -np.inf


def test_snr_matches_independent_inner_product_and_fisher():
    lik, _, _ = _lik()
    f = (BIN_LO + np.arange(N_BINS)) / T_SHORT
    for src in SOURCES:
        h = _independent(src)
        rho = np.sqrt(4.0 / T_SHORT * np.sum(np.abs(h) ** 2 / ae_noise(f)))
        assert lik.snr(src[None])[0] == pytest.approx(rho, rel=1e-10)
    F = lik.fisher_matrices(SOURCES)
    assert F.shape == (2, 8, 8)
    np.testing.assert_allclose(
        F, np.swapaxes(F, 1, 2), rtol=1e-10, atol=1e-10 * np.max(np.abs(F))
    )
    np.testing.assert_allclose(F[:, 3, 3], lik.snr(SOURCES) ** 2, rtol=1e-8)
    assert np.all(np.linalg.eigvalsh(F) > -1e-8 * F[:, 3, 3][:, None])


def test_temperature_batch_matches_rows():
    lik, _, _ = _lik()
    rows = np.stack([_row(SOURCES), _row(SOURCES[:1]), _row(SOURCES[:0])])
    k = np.array([2, 1, 0])
    batch = lik(rows, k)
    single = [lik(rows[i][None], k[i : i + 1])[0] for i in range(3)]
    np.testing.assert_allclose(batch, single, rtol=1e-12)


def test_compile_key_sharing():
    # The kernel cache is process-wide, so compare compile counts before and
    # after the second segment rather than asserting an absolute count.
    liks, compiled = [], []
    for bin_lo, n_bins in ((1000, 100), (5000, 120)):
        lik = SegmentLikelihood(
            T=T_SHORT, t0=0.0, bin_lo=bin_lo, n_bins=n_bins, BW=64, k_max=2, snr_min=0.0
        )
        f = (bin_lo + np.arange(n_bins)) / T_SHORT
        cov = np.stack([0.5 * T_SHORT * ae_noise(f)] * 2)
        lik.set_data(np.zeros(cov.shape, dtype=complex), cov)
        lik.set_reference(cov)
        x = np.full((1, 16), np.nan)
        x[0, :8] = native_source(bin_lo + 50.3)
        assert np.isfinite(lik(x, np.array([1]))[0])
        liks.append(lik)
        compiled.append(
            (lik.kernels.render._cache_size(), lik.kernels.assemble._cache_size())
        )
    a, b = liks
    c = SegmentLikelihood(T=T_SHORT, t0=0.0, bin_lo=5000, n_bins=200, BW=64, k_max=2)
    assert a.key == b.key and a.kernels is b.kernels
    assert compiled[1] == compiled[0] and min(compiled[0]) >= 1
    assert c.key != a.key
    # The template renderer depends only on BW and the waveform settings, so
    # one compile serves windows of every size and slot count.
    assert c.kernels.render is a.kernels.render


def test_automatic_response_grid_is_resolved_at_construction():
    long_T = 1.5 * YEAR
    for T, bin_lo, q, count in (
        (T_SHORT, BIN_LO, 2949.12, 32),
        (long_T, 94000, 94040.4, 256),
    ):
        lik = SegmentLikelihood(
            T=T, t0=0.0, bin_lo=bin_lo, n_bins=N_BINS, BW=BW, k_max=3
        )
        assert lik.key[-1] == count
        src = native_source(q)
        ref = np.asarray(
            template_on_window(
                src,
                T=T,
                bin_lo=bin_lo,
                n_bins=N_BINS,
                BW=BW,
                backend="onfly",
                response_samples=None,
            )
        )
        # Two float64 paths differ at about 1e-12 relative on older JAX/NumPy.
        np.testing.assert_allclose(
            lik.templates(src[None])[0], ref, rtol=0, atol=1e-11 * np.max(np.abs(ref))
        )


@pytest.mark.parametrize("q", [2880.3, 3030.6])
def test_templates_across_window_edges(q):
    # 2880.3 straddles bin_lo; 3030.6 runs into and past the quantization
    # padding (bins 3020 .. 3027 of n_pad = 128) and beyond it.
    lik, r, c = _lik(snr_min=0.0)
    src = native_source(q)
    ref = _independent(src)
    assert 0 < np.count_nonzero(ref[0]) < N_BINS
    # Two float64 paths differ at about 1e-12 relative on older JAX/NumPy.
    np.testing.assert_allclose(
        lik.templates(src[None])[0], ref, rtol=0, atol=1e-11 * np.max(np.abs(ref))
    )
    got = lik(_row(src[None])[None], np.array([1]))[0]
    assert got == pytest.approx(-np.sum(np.abs(r - ref) ** 2 / c), rel=1e-10)


def test_snr_cut_at_threshold():
    lik, _, _ = _lik()
    src = native_source(2949.12)
    rho = lik.snr(src[None])[0]
    for target, finite in ((4.9, False), (5.1, True)):
        scaled = src.copy()
        scaled[3] += np.log(target / rho)
        assert lik.snr(scaled[None])[0] == pytest.approx(target, rel=1e-10)
        value = lik(_row(scaled[None])[None], np.array([1]))[0]
        assert bool(np.isfinite(value)) is finite


def test_covariance_zero_or_inf_excludes_and_nan_or_negative_raises():
    lik, r, c = _lik()
    for bad in (np.nan, -1.0):
        c_bad = c.copy()
        c_bad[0, 5] = bad
        with pytest.raises(ValueError, match="NaN or negative"):
            lik.set_data(r, c_bad)
        with pytest.raises(ValueError, match="NaN or negative"):
            lik.set_reference(c_bad)
    c_excl = c.copy()
    c_excl[0, 5] = 0.0
    c_excl[1, 7] = np.inf
    keep = np.ones(c.shape, dtype=bool)
    keep[0, 5] = keep[1, 7] = False
    lik.set_data(r, c_excl)
    empty = lik(_row(SOURCES[:0])[None], np.array([0]))[0]
    assert empty == pytest.approx(-np.sum(np.abs(r[keep]) ** 2 / c[keep]), rel=1e-12)
    lik.set_reference(c_excl)
    h = _independent(SOURCES[0])
    rho = np.sqrt(2.0 * np.sum(np.abs(h[keep]) ** 2 / c[keep]))
    assert lik.snr(SOURCES[:1])[0] == pytest.approx(rho, rel=1e-10)


def test_construction_and_call_validation():
    base = dict(T=T_SHORT, t0=0.0, bin_lo=BIN_LO, n_bins=N_BINS, BW=BW, k_max=3)
    for bad in (
        {"BW": 63},
        {"BW": 0},
        {"backend": "taylor"},
        {"fmt": "xyz"},
        {"response_samples": 3},
        {"response_samples": True},
        {"t0": np.nan},
        {"t0": np.inf},
    ):
        with pytest.raises(ValueError):
            SegmentLikelihood(**{**base, **bad})
    lik, _, _ = _lik()
    rows = np.stack([_row(SOURCES)] * 2)
    with pytest.raises(ValueError, match="entries"):
        lik(rows, np.array([2]))
    for k in (-1, 4):
        with pytest.raises(ValueError, match="k must lie"):
            lik(rows[:1], np.array([k]))


def test_matches_enchilada_covariance_convention():
    """The log likelihood is -0.5 times the quadratic form 2 sum |x|^2 / C.

    ``C`` is read through enchilada's ``Covariance.pair``, and enchilada's own
    ``noise_psd`` confirms ``C = (T/2) S`` on the band.
    """
    enchilada = pytest.importorskip("enchilada")
    lik, r, c = _lik()
    h = _independent(SOURCES[0]) + _independent(SOURCES[1])
    band = slice(BIN_LO, BIN_LO + N_BINS)
    variances = {}
    for i, ch in enumerate("AE"):
        full = np.ones(N_FREQ)
        full[band] = c[i]
        variances[ch] = full
    data = enchilada.L1Data(
        {ch: np.zeros(N_FREQ, dtype=complex) for ch in "AE"},
        sample_rate_hz=1 / DT,
        tdi_generation="1.5",
        physical_observable="fractional_frequency",
        channel_names=("A", "E"),
        num_time_samples=N_TIME,
        data_domain="frequency",
    )
    cov = enchilada.Covariance.like(data, variances)
    T = N_TIME * DT
    expected = 0.0
    for i, ch in enumerate("AE"):
        C = cov.pair(ch, ch)[band]
        np.testing.assert_allclose(cov.noise_psd(ch, data)[band], 2 * C / T, rtol=1e-12)
        expected += -0.5 * np.sum(2 * np.abs(r[i] - h[i]) ** 2 / C)
    got = lik(_row(SOURCES)[None], np.array([2]))[0]
    assert got == pytest.approx(expected, rel=1e-10)


def _sources(rng, n):
    """Draw ``n`` sources near the test band: two bright groups, one edge, one weak."""
    base = np.array(
        [
            native_source(2949.12),
            native_source(2960.4, lnA=-48.5),
            native_source(2880.3),  # straddles bin_lo
            native_source(2949.12, lnA=-54.0),  # below the SNR cut
        ]
    )
    out = base[rng.integers(0, len(base), n)].copy()
    out[:, 0] += rng.uniform(-0.5, 0.5, n)
    out[:, 3] += rng.uniform(-0.1, 0.1, n)
    out[:, 6] = rng.uniform(0.0, 2 * np.pi, n)
    return out


def _random_batch(rng, n_rows=6, k_max=4):
    ks = rng.integers(0, k_max + 1, n_rows)
    rows = np.stack([_row(_sources(rng, k), k_max=k_max) for k in ks])
    return rows, ks


def _agree(lik, rows, ks):
    got = lik(rows, ks)
    ref = lik.lnlike_uncached(rows, ks)
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref))
    ok = np.isfinite(ref)
    np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-12)
    return got


def test_cached_likelihood_matches_uncached():
    lik, _, _ = _lik(k_max=4)
    rng = np.random.default_rng(3)
    rows, ks = _random_batch(rng, n_rows=8)
    bad = native_source(2949.12)
    bad[4] = 1.5  # invalid cos(inclination): -inf in both paths
    rows[0, :8], ks[0] = bad, max(ks[0], 1)
    first = _agree(lik, rows, ks)
    assert first[0] == -np.inf
    assert np.isfinite(first).any() and (~np.isfinite(first[1:])).any()
    renders = lik.cache_info()["renders"]
    again = lik(rows, ks)  # every source is now cached
    np.testing.assert_array_equal(again, first)
    assert lik.cache_info()["renders"] == renders


def test_cache_renders_only_the_sources_a_move_creates():
    lik, _, _ = _lik(k_max=4)
    rng = np.random.default_rng(4)
    srcs = _sources(rng, 9)
    rows = np.stack([_row(srcs[0:3], 4), _row(srcs[3:6], 4), _row(srcs[6:9], 4)])
    ks = np.array([3, 3, 3])
    _agree(lik, rows, ks)
    assert lik.cache_info()["renders"] == 9

    def renders_for(new_rows, new_ks):
        before = lik.cache_info()["renders"]
        _agree(lik, new_rows, new_ks)
        return lik.cache_info()["renders"] - before

    moved = rows.copy()
    moved[0, 8:16] = _sources(rng, 1)[0]  # within-model move of one slot
    assert renders_for(moved, ks) == 1
    assert renders_for(rows, ks) == 0  # rejected: the old value is still cached
    death = rows.copy()
    death[1] = _row(np.delete(srcs[3:6], 0, axis=0), 4)  # later slots shift down
    assert renders_for(death, np.array([3, 2, 3])) == 0
    birth = rows.copy()
    birth[2] = _row(np.insert(srcs[6:9], 1, _sources(rng, 1), axis=0), 4)
    assert renders_for(birth, np.array([3, 3, 4])) == 1
    assert renders_for(rows[::-1], ks[::-1]) == 0  # tempering swap
    assert renders_for(rows[1:], ks[1:]) == 0  # only the finite-prior rows


def test_cache_survives_a_new_residual():
    lik, r, c = _lik(k_max=4)
    rows, ks = _random_batch(np.random.default_rng(5))
    _agree(lik, rows, ks)
    renders = lik.cache_info()["renders"]
    lik.set_data(r + 3.0 * np.sqrt(c), c)
    _agree(lik, rows, ks)
    assert lik.cache_info()["renders"] == renders


def test_cache_eviction_and_clear_keep_values_exact():
    lik = SegmentLikelihood(
        T=T_SHORT,
        t0=0.0,
        bin_lo=BIN_LO,
        n_bins=N_BINS,
        BW=BW,
        k_max=4,
        backend="onfly",
        cache_slots=8,
    )
    _, r, c = _lik()
    lik.set_data(r, c)
    lik.set_reference(c)
    rng = np.random.default_rng(6)
    rows, ks = _random_batch(rng, n_rows=4)
    for _ in range(12):
        i, j = rng.integers(0, len(rows)), rng.integers(0, 4)
        rows[i, 8 * j : 8 * j + 8] = _sources(rng, 1)[0]
        ks[i] = max(ks[i], j + 1)
        _agree(lik, rows, ks)
        # The store grows past cache_slots only to hold one call's sources.
        assert lik.cache_info()["entries"] <= 4 * 4
    first = lik(rows, ks)
    lik.clear_cache()
    assert lik.cache_info()["entries"] == 0
    # Re-rendered in a different chunk size, so equal to roundoff, not bitwise.
    np.testing.assert_allclose(lik(rows, ks), first, rtol=1e-12)


@pytest.mark.slow
@requires_sangria
def test_noise_only_whitened_power_is_unity():
    signal = pytest.importorskip("scipy.signal")
    import sangria

    xyz, dt = sangria.combine(["obs"], ["dgb", "igb", "vgb", "mbhb"], SANGRIA)
    series = sangria.aet_series(xyz)
    n = series["A"].size
    T = n * dt
    bin_lo, n_bins = int(3e-3 * T), 20_000
    freqs = (bin_lo + np.arange(n_bins)) / T
    # Taper before the rfft: an untapered dt*rfft leaks A's low-frequency trend
    # into the band (about 6% excess power at 3 mHz). Normalize by the window's rms.
    w = signal.windows.tukey(n, 0.05)
    w /= np.sqrt(np.mean(w**2))
    c = []
    r = []
    for ch in "AE":
        f_w, s_w = signal.welch(series[ch], fs=1.0 / dt, nperseg=2**16)
        c.append(0.5 * T * np.interp(freqs, f_w, s_w))
        r.append(dt * np.fft.rfft(w * series[ch])[bin_lo : bin_lo + n_bins])
    lik = SegmentLikelihood(T=T, t0=0.0, bin_lo=bin_lo, n_bins=n_bins, BW=16, k_max=1)
    lik.set_data(np.stack(r), np.stack(c))
    lik.set_reference(np.stack(c))
    power = -lik(np.full((1, 8), np.nan), np.array([0]))[0] / (2 * n_bins)
    assert abs(power - 1.0) < 0.03, power
