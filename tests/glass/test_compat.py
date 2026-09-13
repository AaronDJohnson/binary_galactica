"""Fixed-window compatibility behavior without external generators."""

from __future__ import annotations

import numpy as np

import binary_galactica.glass as glass


def test_template_on_window_matches_direct_overlap(glass_params, injection):
    T, BW = injection["T"], 64
    waveform = glass.to_continuous_ft(
        glass.ucb_waveform(glass_params, T=T, BW=BW),
        T,
    )
    bins = np.asarray(waveform["freq_bin_indices"], dtype=int)
    bin_lo, n_bins = int(bins[0]) + 11, 21

    got = np.asarray(
        glass.template_on_window(
            glass_params,
            T=T,
            bin_lo=bin_lo,
            n_bins=n_bins,
            BW=BW,
        )
    )
    expected = np.stack(
        [
            np.asarray(waveform["A"])[11 : 11 + n_bins],
            np.asarray(waveform["E"])[11 : 11 + n_bins],
        ]
    )
    assert np.allclose(got, expected, rtol=2e-15, atol=0.0)


def test_far_out_of_window_source_contributes_exactly_zero(glass_params, injection):
    T, BW = injection["T"], 64
    got = np.asarray(
        glass.template_on_window(
            glass_params,
            T=T,
            bin_lo=100_000,
            n_bins=100,
            BW=BW,
        )
    )
    assert np.array_equal(got, np.zeros((2, 100), dtype=np.complex128))


def test_bins_below_the_window_do_not_wrap_to_the_end(glass_params, injection):
    T, BW = injection["T"], 64
    qmin = int(glass.ucb_waveform(glass_params, T=T, BW=BW)["qmin"])
    bin_lo = qmin + BW // 2
    got = np.asarray(
        glass.template_on_window(
            glass_params,
            T=T,
            bin_lo=bin_lo,
            n_bins=BW,
            BW=BW,
        )
    )
    assert np.array_equal(got[:, BW // 2 :], np.zeros((2, BW // 2), complex))
