"""``ucb_alignment`` must reproduce GLASS's band bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

import binary_galactica.glass as gbonfly
from binary_galactica.glass import ucb_alignment
from binary_galactica.glass.bandwidth import ucb_bandwidth


def test_alignment_matches_the_c_expressions(injection, glass_params):
    T = injection["T"]
    N = int(T / 5.0)
    a = ucb_alignment(glass_params, T, N)

    f0 = glass_params[0] / T
    half = ucb_bandwidth(
        f0, glass_params[7] / T**2, glass_params[1], np.exp(glass_params[3]), T, N
    )
    assert 2 * half == a.BW  # the doubling lives here
    assert a.q == int(np.trunc(f0 * T))
    assert a.qmin == a.q - a.BW // 2
    assert a.qmax == a.qmin + a.BW
    assert a.imin == a.qmin and a.imax == a.qmax  # segment_qmin defaults to 0


def test_segment_offset_only_moves_the_i_indices(injection, glass_params):
    T, N = injection["T"], int(injection["T"] / 5.0)
    a = ucb_alignment(glass_params, T, N)
    b = ucb_alignment(glass_params, T, N, segment_qmin=7756)
    assert (b.BW, b.qmin, b.qmax, b.q) == (a.BW, a.qmin, a.qmax, a.q)
    assert b.imin == a.qmin - 7756
    assert b.imax == b.imin + a.BW


def test_alignment_agrees_with_the_generator(injection, glass_params):
    """The whole point: what alignment predicts is what the generator emits."""
    T, N = injection["T"], int(injection["T"] / 5.0)
    a = ucb_alignment(glass_params, T, N)
    w = gbonfly.ucb_waveform(
        glass_params, T=T, BW=a.BW, positions=gbonfly.orbit_cache(T, 0.0, a.BW)
    )
    assert int(w["qmin"]) == a.qmin
    assert np.array_equal(np.asarray(w["freq_bin_indices"]).astype(int), a.bins)
    assert a.bins[-1] == a.qmax - 1


def test_alignment_uses_the_round_tripped_carrier(injection):
    """``(int)(f0*T)``, not ``trunc(params[0])`` -- same rule as the generator."""
    T, N = injection["T"], int(injection["T"] / 5.0)
    victims = [
        n
        for n in range(1000, 3000)
        if np.trunc((float(n) / T) * T) != np.trunc(float(n))
    ]
    assert victims
    n = victims[0]
    p = np.array([float(n), 0.3, 0.0, np.log(1e-22), 0.2, 0.7, 0.5, 0.0])
    a = ucb_alignment(p, T, N)
    assert a.q == n - 1
    w = gbonfly.ucb_waveform(
        p, T=T, BW=a.BW, positions=gbonfly.orbit_cache(T, 0.0, a.BW)
    )
    assert int(w["qmin"]) == a.qmin


def test_safety_widens_the_band_only_through_the_snr_term(injection, glass_params):
    T, N = injection["T"], int(injection["T"] / 5.0)
    base = ucb_alignment(glass_params, T, N)
    wide = ucb_alignment(glass_params, T, N, safety=16.0)
    assert wide.BW > base.BW
    assert wide.q == base.q  # carrier does not move


def test_rejects_too_short_a_parameter_vector(injection):
    with pytest.raises(ValueError, match="at least 8"):
        ucb_alignment(np.zeros(7), injection["T"], 1024)
