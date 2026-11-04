from types import SimpleNamespace

import numpy as np
import pytest

from .support import SANGRIA, requires_sangria


@requires_sangria
def test_target_band():
    import runs
    import sangria

    n, dt = sangria.grid(SANGRIA)
    T = n * dt
    core = (113993, 114068)
    base = runs.catalogue_rows(SANGRIA, T=T, band=(core[0], core[1]))
    band, rows = runs.target_band(SANGRIA, T=T, core=core, span=100, extra=base)
    assert band == (core[0] - 100, core[1] + 100)
    q = rows["Frequency"] * T
    assert np.all((q >= band[0]) & (q < band[1]))
    assert np.all(np.diff(rows["Frequency"]) >= 0)
    assert len(set(rows["Name"])) == len(rows)
    assert set(base["Name"]) <= set(rows["Name"])
    _, plain = runs.target_band(SANGRIA, T=T, core=core, span=100)
    assert len(plain) <= len(rows)
    # every row not from ``extra`` meets the SNR cut
    snr = runs.analytic_snrs(rows["Frequency"], rows["Amplitude"], T)
    assert np.all(snr[~np.isin(rows["Name"], base["Name"])] >= 7.0)


def test_inner_segments():
    import runs

    segs = [SimpleNamespace(index=i, lo=10 * i, hi=10 * i + 10) for i in range(4)]
    block = SimpleNamespace(table=SimpleNamespace(segments=segs))
    assert [s.index for s in runs.inner_segments(block, (12, 28))] == [1, 2]
    with pytest.raises(ValueError):
        runs.inner_segments(block, (5, 28))
    with pytest.raises(ValueError):
        runs.inner_segments(block, (12, 35))
    with pytest.raises(ValueError):
        runs.inner_segments(
            SimpleNamespace(table=SimpleNamespace(segments=segs[:2])), (0, 1)
        )


@requires_sangria
def test_residual_spectra_removes_noise_free_sources():
    import runs
    import sangria

    n, dt = sangria.grid(SANGRIA)
    T = n * dt
    spectra = sangria.aet_spectra(*sangria.combine(["dgb"], [], SANGRIA))
    before = {ch: s.copy() for ch, s in spectra.items()}
    lo, hi = 113500, 113560
    # the dgb stream has no noise, so subtracting every catalogue row in a wider band
    # (the template wings reach past [lo, hi)) must leave almost nothing in [lo, hi)
    resid, rows = runs.residual_spectra(
        spectra, SANGRIA, T=T, lo=lo - 40, hi=hi + 40, snr_min=None
    )
    assert len(rows) > 5
    for ch in ("A", "E"):
        power_in = np.sum(np.abs(spectra[ch][lo:hi]) ** 2)
        power_out = np.sum(np.abs(resid[ch][lo:hi]) ** 2)
        assert power_in > 0
        assert power_out < 0.02 * power_in  # measured 0.0067 (A), 0.0055 (E)
    for ch in spectra:
        np.testing.assert_array_equal(spectra[ch], before[ch])


@requires_sangria
def test_noise_only_stream_matches_instrument_variance():
    """Noise-only running median over the instrument variance, at four frequencies.

    The measured ratios run 1.03-1.20 (Sangria's noise parameters against GLASS's SPS/SACC
    defaults), which is why the bound [0.9, 1.35] is asymmetric about 1.
    """
    import runs
    import sangria

    n, dt = sangria.grid(SANGRIA)
    T = n * dt
    spectra = sangria.aet_spectra(
        *sangria.combine(["obs"], ["dgb", "igb", "vgb", "mbhb"], SANGRIA)
    )
    for f0 in (2e-3, 5e-3, 8e-3, 12.7e-3):
        q0 = int(f0 * T)
        lo, hi = q0 - 200, q0 + 200
        var = runs.running_median_variance(spectra, lo, hi, half_width=200)
        f = np.arange(lo, hi) / T
        for ch in ("A", "E"):
            ratio = float(np.median(var[ch] / runs.instrument_variance(f, T)))
            assert 0.9 <= ratio <= 1.35


def test_residual_spectra_rejects_unknown_channels():
    import runs

    for bad in (
        {"A": np.zeros(8)},
        {"E": np.zeros(8), "A": np.zeros(8)},
        {"A": np.zeros(8), "E": np.zeros(8), "X": np.zeros(8)},
    ):
        with pytest.raises(ValueError):
            runs.residual_spectra(bad, "unused", T=1.0, lo=1, hi=4)
