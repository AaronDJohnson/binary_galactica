"""GLASS's UCB prior: bounds, flatness, wrapping."""

from __future__ import annotations

import numpy as np
import pytest

from binary_galactica.glass.prior import (
    LNA_MAX,
    LNA_MIN,
    UCBPrior,
    ucb_fdot,
    ucb_prior,
)

T = 7884000.0
FMIN, FMAX = 0.0009837645865043126, 0.0010162354134956874


@pytest.fixture(scope="module")
def pr():
    return ucb_prior(T=T, fmin=FMIN, fmax=FMAX)


def test_bounds_match_the_c(pr):
    assert pr.lo[0] == FMIN * T and pr.hi[0] == FMAX * T
    assert (pr.lo[1], pr.hi[1]) == (-1.0, 1.0)  # cos theta
    assert (pr.lo[2], pr.hi[2]) == (0.0, 2 * np.pi)  # longitude
    assert (pr.lo[3], pr.hi[3]) == (LNA_MIN, LNA_MAX)  # ln A
    assert (pr.lo[4], pr.hi[4]) == (-1.0, 1.0)  # cos iota
    assert (pr.lo[5], pr.hi[5]) == (0.0, np.pi)  # psi
    assert (pr.lo[6], pr.hi[6]) == (0.0, 2 * np.pi)  # phi0


def test_fdot_envelope_is_the_mldc_one(pr):
    assert pr.lo[7] == pytest.approx(-0.000005 * FMIN ** (13 / 3) * T * T)
    assert pr.hi[7] == pytest.approx(0.0000008 * FMAX ** (11 / 3) * T * T)
    # The default box admits negative fdot -- outspiralling systems.
    assert pr.lo[7] < 0.0


def test_detached_uses_the_chirp_mass_relation():
    d = ucb_prior(T=T, fmin=FMIN, fmax=FMAX, detached=True)
    assert d.lo[7] == pytest.approx(ucb_fdot(0.15, FMIN) * T * T)
    assert d.hi[7] == pytest.approx(ucb_fdot(1.00, FMAX) * T * T)
    assert d.lo[7] > 0.0  # GR-driven inspiral only


def test_prior_is_flat_inside_and_minus_inf_outside(pr):
    rng = np.random.default_rng(0)
    draws = pr.sample(rng, 200)
    lp = np.array([pr.log_prior(p) for p in draws])
    assert np.all(np.isfinite(lp))
    assert np.allclose(lp, lp[0])  # flat
    assert lp[0] == pytest.approx(-pr.log_volume)

    bad = draws[0].copy()
    bad[3] = LNA_MAX + 1.0
    assert pr.log_prior(bad) == -np.inf
    bad = draws[0].copy()
    bad[1] = 1.5  # |cos theta| > 1
    assert pr.log_prior(bad) == -np.inf


def test_angles_wrap_rather_than_reject(pr):
    """GLASS folds longitude with atan2 instead of rejecting it."""
    rng = np.random.default_rng(1)
    p = pr.sample(rng)
    for idx, period in ((2, 2 * np.pi), (5, np.pi), (6, 2 * np.pi)):
        q = p.copy()
        q[idx] = p[idx] + 3 * period  # far out of range
        assert np.isfinite(pr.log_prior(q)), f"index {idx} was rejected"
        assert pr.wrap(q)[idx] == pytest.approx(p[idx])


def test_wrap_does_not_mutate_its_input(pr):
    """The C wraps in place; doing that in Python would corrupt a chain."""
    rng = np.random.default_rng(2)
    p = pr.sample(rng)
    p[2] = 9.0
    before = p.copy()
    pr.wrap(p)
    assert np.array_equal(p, before)


def test_nan_is_rejected(pr):
    rng = np.random.default_rng(3)
    p = pr.sample(rng)
    p[0] = np.nan
    assert pr.log_prior(p) == -np.inf


def test_amplitude_prior_is_log_uniform_not_uniform(pr):
    """Flat in ln A means p(A) ~ 1/A -- a different prior from the old path."""
    rng = np.random.default_rng(4)
    A = np.exp(pr.sample(rng, 20000)[:, 3])
    # A log-uniform sample has a flat histogram in log A, so the median sits
    # at the geometric, not arithmetic, centre of the box.
    assert np.median(np.log(A)) == pytest.approx((LNA_MIN + LNA_MAX) / 2, abs=0.1)
    assert np.median(A) < 0.5 * (np.exp(LNA_MIN) + np.exp(LNA_MAX))


def test_snr_cut_matches_glass_threshold():
    """GLASS rejects rho < 5 in the sampler, not the prior file."""
    from binary_galactica.glass.prior import SNR_MIN, snr_cut

    assert SNR_MIN == 5.0
    assert not snr_cut(4.999)
    assert snr_cut(5.0)
    assert snr_cut(7.0)
    # Disabling it is explicit, not a magic sentinel.
    assert snr_cut(1.0, snr_min=0.0)


def test_snr_cut_is_not_folded_into_the_box_prior(pr):
    """It cannot be: the box prior never sees the waveform or the noise."""
    rng = np.random.default_rng(5)
    p = pr.sample(rng)
    p[3] = LNA_MIN + 1e-6  # very faint but inside the box
    assert np.isfinite(pr.log_prior(p))


# One case per guard. Each sits on its boundary (zero, equal bounds, a
# zero-width interval) so that a weakened comparison also fails.
def test_observation_time_must_be_finite_and_positive():
    with pytest.raises(ValueError, match="T must be finite and positive"):
        ucb_prior(T=0.0, fmin=0.001, fmax=0.002)


def test_frequency_bounds_must_be_finite_positive_and_ordered():
    with pytest.raises(ValueError, match="0 < fmin < fmax"):
        ucb_prior(T=T, fmin=0.001, fmax=0.001)


def test_amplitude_bounds_must_be_a_finite_ordered_pair():
    with pytest.raises(ValueError, match="lnA_bounds.*finite.*ordered"):
        ucb_prior(T=T, fmin=FMIN, fmax=FMAX, lnA_bounds=(-54.0, -54.0))


def test_direct_prior_requires_eight_bounds():
    # Eight entries in the wrong shape: a size check alone would pass.
    with pytest.raises(ValueError, match="bounds.*shape.*8"):
        UCBPrior(lo=np.zeros((2, 4)), hi=np.ones((2, 4)))


def test_direct_prior_rejects_invalid_bounds():
    lo, hi = np.zeros(8), np.ones(8)
    hi[3] = 0.0
    with pytest.raises(ValueError, match="finite.*strictly ordered"):
        UCBPrior(lo=lo, hi=hi, periodic={})


def test_prior_snapshots_caller_bounds(pr):
    lo, hi = pr.lo.copy(), pr.hi.copy()
    copied = UCBPrior(lo=lo, hi=hi)
    lo[0] = hi[0]
    hi[3] = np.nan
    assert np.array_equal(copied.lo, pr.lo)
    assert np.array_equal(copied.hi, pr.hi)
    assert not copied.lo.flags.writeable
    assert not copied.hi.flags.writeable


def test_prior_rejects_wrong_parameter_dimensions(pr):
    # wrap's guard; log_prior (and contains, which calls it) has its own,
    # covered by the batch case below.
    with pytest.raises(ValueError, match=r"params must have shape \(\.\.\., 8\)"):
        pr.wrap(np.zeros(7))


def test_log_prior_requires_one_parameter_vector(pr):
    with pytest.raises(ValueError, match="params.*shape.*8"):
        pr.log_prior(pr.sample(np.random.default_rng(9), 2))


def test_wrap_supports_batches_of_eight_parameters(pr):
    samples = pr.sample(np.random.default_rng(8), 3)
    shifted = samples.copy()
    shifted[:, 5] += 3 * np.pi
    assert np.allclose(pr.wrap(shifted), samples)


# The index guard (8 would otherwise raise IndexError) and the period/width
# match; a zero or infinite period can never match a finite positive width.
@pytest.mark.parametrize("periodic", [{8: 1.0}, {2: 1.0}])
def test_prior_rejects_invalid_periodic_coordinates(pr, periodic):
    with pytest.raises(ValueError, match="periodic"):
        UCBPrior(lo=pr.lo, hi=pr.hi, periodic=periodic)


def test_periodic_wrapping_uses_the_lower_bound(pr):
    lo, hi = pr.lo.copy(), pr.hi.copy()
    lo[2], hi[2] = -np.pi, np.pi
    shifted = UCBPrior(lo=lo, hi=hi)
    p = shifted.sample(np.random.default_rng(0))
    p[2] = 1.5 * np.pi
    assert shifted.wrap(p)[2] == pytest.approx(-np.pi / 2)
    assert shifted.contains(p)
