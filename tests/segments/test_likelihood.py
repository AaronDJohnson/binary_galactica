import numpy as np
import pytest

from binary_galactica.glass import ae_noise, template_on_window
from binary_galactica.segments import SegmentLikelihood, quantize_bins

from .support import DT, N_FREQ, N_TIME, SANGRIA, T_SHORT, native_source, requires_sangria

BIN_LO, N_BINS, BW = 2900, 120, 64
SOURCES = np.array([native_source(2949.12), native_source(2960.4, lnA=-48.5)])


def _c():
    f = (BIN_LO + np.arange(N_BINS)) / T_SHORT
    return np.stack([0.5 * T_SHORT * ae_noise(f)] * 2)


def _lik(k_max=3, snr_min=5.0):
    lik = SegmentLikelihood(T=T_SHORT, t0=0.0, bin_lo=BIN_LO, n_bins=N_BINS, BW=BW,
                            k_max=k_max, backend="onfly", snr_min=snr_min)
    rng = np.random.default_rng(0)
    c = _c()
    r = np.sqrt(c / 2) * (rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape))
    lik.set_data(r, c)
    lik.set_reference(c)
    return lik, r, c


def _row(sources, k_max=3, fill=np.nan):
    x = np.full(8 * k_max, fill)
    x[: sources.size] = np.ravel(sources)
    return x


def _independent(src):
    return np.asarray(template_on_window(src, T=T_SHORT, bin_lo=BIN_LO, n_bins=N_BINS,
                                         BW=BW, backend="onfly"))


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
    np.testing.assert_allclose(F, np.swapaxes(F, 1, 2), rtol=1e-10,
                               atol=1e-10 * np.max(np.abs(F)))
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
    a = SegmentLikelihood(T=T_SHORT, t0=0.0, bin_lo=1000, n_bins=100, BW=64, k_max=3)
    b = SegmentLikelihood(T=T_SHORT, t0=0.0, bin_lo=5000, n_bins=120, BW=64, k_max=3)
    c = SegmentLikelihood(T=T_SHORT, t0=0.0, bin_lo=5000, n_bins=200, BW=64, k_max=3)
    assert a.key == b.key and a.kernels is b.kernels
    assert c.key != a.key


def test_matches_enchilada_quadratic_form():
    enchilada = pytest.importorskip("enchilada")
    lik, r, c = _lik()
    h = _independent(SOURCES[0]) + _independent(SOURCES[1])
    cov = np.ones((N_FREQ, 2, 2)) * np.eye(2)
    cov[BIN_LO : BIN_LO + N_BINS, 0, 0] = c[0]
    cov[BIN_LO : BIN_LO + N_BINS, 1, 1] = c[1]
    mask = np.zeros(N_FREQ, dtype=bool)
    mask[BIN_LO : BIN_LO + N_BINS] = True
    dc = enchilada.DataCovariance(cov, channel_names=("A", "E"), sample_rate_hz=1 / DT,
                                  num_time_samples=N_TIME, data_domain="frequency",
                                  active_mask=mask)
    full = {ch: np.zeros(N_FREQ, dtype=complex) for ch in "AE"}
    for i, ch in enumerate("AE"):
        full[ch][BIN_LO : BIN_LO + N_BINS] = r[i] - h[i]
    got = lik(_row(SOURCES)[None], np.array([2]))[0]
    assert got == pytest.approx(-0.5 * dc.quadratic_form(full), rel=1e-10)


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
