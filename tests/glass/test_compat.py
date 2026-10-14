"""Fixed-window compatibility behavior without external generators."""

from __future__ import annotations

import numpy as np
import pytest

import binary_galactica.glass as glass


# The shared integer check's conditions are covered on ucb_bandwidth's N;
# these cases cover each argument's wiring and n_bins's zero minimum.
def test_window_origin_rejects_noninteger_indices(glass_params):
    with pytest.raises(ValueError, match="bin_lo must be an integer"):
        glass.template_on_window(
            glass_params, T=7_884_000.0, BW=32, bin_lo=7868.1, n_bins=32
        )


def test_window_length_rejects_invalid_counts(glass_params):
    with pytest.raises(ValueError, match="n_bins must be a non-negative integer"):
        glass.template_on_window(
            glass_params, T=7_884_000.0, BW=32, bin_lo=7868, n_bins=-1
        )


def test_window_accepts_numpy_integer_indices_and_empty_windows(glass_params):
    result = glass.template_on_window(
        glass_params,
        T=7_884_000.0,
        BW=32,
        bin_lo=np.int64(-10),
        n_bins=np.int64(0),
    )
    assert result.shape == (2, 0)


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
