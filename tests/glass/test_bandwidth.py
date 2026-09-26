"""GLASS's bandwidth heuristic, including the parts that look like bugs.

``_next_pow2`` reproduces ``(int)pow(2,(int)log2(x-1)+1)``, which is not
"round up to a power of two" at the boundaries, and ``ucb_bw`` reproduces
the factor of two that ``ucb_alignment`` applies far from the heuristic.
"""

from __future__ import annotations

import numpy as np
import pytest

from binary_galactica.glass.bandwidth import _next_pow2, ucb_bandwidth, ucb_bw


@pytest.mark.parametrize(
    "x,want",
    [
        (0.0, 1),
        (0.5, 1),
        (1.0, 1),  # log2 of a non-positive: clamped
        (1.5, 1),  # (int)log2(0.5) = -1 -> 2^0
        (2.0, 2),
        (3.0, 4),
        (5.0, 8),
        (9.0, 16),
        (16.0, 16),  # exact power of two stays put
        (17.0, 32),
    ],
)
def test_next_pow2_matches_c_idiom(x, want):
    """The C idiom is the smallest power of two strictly greater than ``x-1``.

    For integer ``x`` that is "round up to a power of two", but the ``-1``
    is what keeps an exact power of two from doubling, and it is why
    ``x <= 1`` falls off the bottom into the ``Nmin`` clamp.
    """
    assert _next_pow2(x) == want


def test_glass_returns_32_for_the_shipped_injection():
    """The published heuristic on this segment, and the reason it is not enough.

    ``ucb_bandwidth`` gives 16 (both terms hit the ``Nmin`` floor) and
    ``ucb_alignment`` doubles it to 32 slow samples --- against the ~512 the
    indistinguishability bound needs at this SNR.
    """
    T = 7884000.0
    half = ucb_bandwidth(
        f=1e-3,
        fdot=1e-18,
        costheta=np.cos(1.7806669190576343),
        amp=7.1562127917748725e-22,
        T=T,
        N=int(T / 5.0),
    )
    assert half == 16
    assert (
        ucb_bw(
            f=1e-3,
            fdot=1e-18,
            costheta=np.cos(1.7806669190576343),
            amp=7.1562127917748725e-22,
            T=T,
            N=int(T / 5.0),
        )
        == 32
    )


def test_doppler_term_scales_with_observation_time():
    """The Doppler width carries ``/YEAR``, so it is a bin count, not an index.

    Ten times the baseline is ten times the Doppler spread in bins. A
    heuristic that added the modulation index directly would be flat here.
    """
    kw = dict(f=1e-2, fdot=0.0, costheta=0.0, amp=1e-30, N=1 << 22)
    short = ucb_bandwidth(T=7884000.0, **kw)
    long = ucb_bandwidth(T=10 * 7884000.0, **kw)
    assert long > short


def test_snr_term_drives_bandwidth_for_loud_sources():
    """The SNR term is what makes BW scale with rho, which is the right law."""
    kw = dict(f=1e-3, fdot=0.0, costheta=0.0, T=7884000.0, N=1 << 22)
    quiet = ucb_bandwidth(amp=1e-23, **kw)
    loud = ucb_bandwidth(amp=1e-19, **kw)
    assert loud > quiet
    # Roughly proportional: 1e4 in amplitude is ~2^13 in the power-of-two ladder.
    assert loud / quiet >= 64


def test_safety_multiplier_only_touches_the_snr_term():
    kw = dict(f=1e-3, fdot=0.0, costheta=0.0, amp=1e-20, T=7884000.0, N=1 << 22)
    base = ucb_bandwidth(**kw)
    assert ucb_bandwidth(safety=1.0, **kw) == base
    assert ucb_bandwidth(safety=4.0, **kw) >= 4 * base


def test_nmax_clamp():
    """Neither term may exceed 2^floor(log2(N/2))."""
    got = ucb_bandwidth(
        f=1e-2, fdot=1e-10, costheta=0.0, amp=1e-15, T=7884000.0, N=1024
    )
    assert got <= 512


def test_clamp_order_puts_the_floor_last():
    """GLASS clamps to Nmax first and Nmin last, so Nmin wins outright.

    Only visible when Nmax < Nmin, i.e. a grid shorter than 64 bins:
    Nmax = 2^floor(log2(N/2)) = 16 at N=32, and 8 at N=16. GLASS still
    returns 16 there; the natural min(max(x,16),Nmax) would return 8.
    """
    kw = dict(f=1e-3, fdot=0.0, costheta=0.0, amp=1e-30, T=7884000.0)
    assert ucb_bandwidth(N=16, **kw) == 16
    assert ucb_bandwidth(N=32, **kw) == 16


@pytest.mark.parametrize("bandwidth", [ucb_bandwidth, ucb_bw])
@pytest.mark.parametrize("N", [0, -1, True, np.bool_(True), 1.5, 2.0, np.nan, np.inf])
def test_bandwidth_requires_a_positive_integer_grid_length(bandwidth, N):
    with pytest.raises(ValueError, match="N must be a positive integer"):
        bandwidth(f=1e-3, fdot=0.0, costheta=0.0, amp=1e-30, T=7884000.0, N=N)
