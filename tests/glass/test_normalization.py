"""Unit conversion around the raw GLASS spectrum."""

from __future__ import annotations

import numpy as np

import binary_galactica.glass as glass


def test_continuous_fourier_conversion_is_exactly_root_t(glass_params, injection):
    T = injection["T"]
    raw = glass.ucb_waveform(glass_params, T=T, BW=32)
    converted = glass.to_continuous_ft(raw, T)
    assert float(glass.GLASS_TO_CONTINUOUS(T)) == np.sqrt(T)
    for channel in ("X", "Y", "Z", "A", "E"):
        assert np.array_equal(
            np.asarray(converted[channel]),
            np.asarray(raw[channel]) * np.sqrt(T),
        )
    assert converted["qmin"] is raw["qmin"]
    assert converted["freq_bin_indices"] is raw["freq_bin_indices"]
