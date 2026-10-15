"""GLASS's proposal cocktail: the pieces that replace the coordinate change."""

from __future__ import annotations

import numpy as np
import pytest

from binary_galactica.glass.constants import YEAR
from binary_galactica.glass.prior import ucb_prior
from binary_galactica.glass.proposals import (
    ExtrinsicPriorDraw,
    FisherProposal,
    PriorDraw,
    ProposalMix,
    draw_from_fisher,
    fm_shift,
    psi_phi_jump,
)

T = 7884000.0
P0 = np.array([7884.0, -0.208, 0.0, -48.68, 0.3, 0.7, 0.5, 6.2e-5])


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


class _DeterministicRNG:
    """Supply one fixed normal amplitude and choose eigendirection zero."""

    def __init__(self, amplitude):
        self.amplitude = amplitude

    def standard_normal(self, n):
        return np.full(n, self.amplitude)

    def uniform(self):
        return 0.0


# --- fisher ---------------------------------------------------------------


def test_fisher_moves_along_exactly_one_eigenvector(rng):
    """One eigendirection per jump -- that is what makes it the rotation."""
    # Eigenvalues large enough that no step can leave the cosine range, so
    # the boundary rejection cannot confound "exactly one coordinate moved".
    evec = np.eye(8)
    evals = np.full(8, 1e4)
    moved = set()
    for _ in range(200):
        out, lq = draw_from_fisher(P0, rng, evals, evec)
        assert lq == 0.0
        nz = np.nonzero(~np.isclose(out - P0, 0.0, atol=0))[0]
        assert nz.size == 1, f"moved {nz.size} coordinates, expected 1"
        moved.add(int(nz[0]))
    assert moved == set(range(8)), "not all eigendirections were visited"


def test_fisher_step_scales_as_one_over_sqrt_eigenvalue(rng):
    """N(0,1)/sqrt(lambda), with no 1/sqrt(NP) -- the sqNP line is commented out."""
    # Large eigenvalues => small steps => the boundary rejection never fires and
    # the measured spread is the raw proposal scale.
    evec = np.eye(8)
    for lam in (1e4, 1e6):
        steps = []
        r = np.random.default_rng(7)
        for _ in range(4000):
            out, _ = draw_from_fisher(P0, r, np.full(8, lam), evec)
            d = out - P0
            steps.append(d[np.argmax(np.abs(d))])
        # Only 1/8 of draws hit any given direction, but the magnitude
        # distribution is the same for all, so the pooled sd is 1/sqrt(lam).
        assert np.std(steps) == pytest.approx(1 / np.sqrt(lam), rel=0.08)


def test_fisher_rejects_the_whole_nonfinite_jump():
    """Invalid eigensystems produce a symmetric self-proposal.

    Negative, zero and NaN eigenvalues all reach the same non-finite-jump
    guard; a singular Fisher matrix (zero) is the common one.
    """
    out, logq = draw_from_fisher(
        P0,
        _DeterministicRNG(1.0),
        np.full(8, 0.0),
        np.eye(8),
    )
    assert np.array_equal(out, P0)
    assert logq == 0.0


def test_fisher_rejects_a_nonfinite_proposed_output():
    """Finite inputs that overflow during addition produce a self-proposal."""
    params = P0.copy()
    params[0] = np.finfo(np.float64).max
    out, logq = draw_from_fisher(
        params,
        _DeterministicRNG(np.finfo(np.float64).max),
        np.ones(8),
        np.eye(8),
    )
    assert np.array_equal(out, params)
    assert logq == 0.0


def test_fisher_rejects_the_whole_correlated_jump_at_a_cosine_boundary():
    """A boundary hit must not alter the remaining eigenvector coordinates."""
    params = np.zeros(8)
    params[1] = 0.9
    evectors = np.eye(8)
    evectors[:3, :3] = np.array(
        [
            [1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(6)],
            [1 / np.sqrt(3), -1 / np.sqrt(2), 1 / np.sqrt(6)],
            [1 / np.sqrt(3), 0.0, -2 / np.sqrt(6)],
        ]
    )
    assert np.allclose(evectors.T @ evectors, np.eye(8))

    for amplitude in (0.1 * np.sqrt(3), 0.2 * np.sqrt(3)):
        rejected, logq = draw_from_fisher(
            params,
            _DeterministicRNG(amplitude),
            np.ones(8),
            evectors,
        )
        assert np.array_equal(rejected, params)
        assert logq == 0.0

    accepted, logq = draw_from_fisher(
        params,
        _DeterministicRNG(0.05 * np.sqrt(3)),
        np.ones(8),
        evectors,
    )
    expected = params.copy()
    expected[:3] += 0.05
    assert np.allclose(accepted, expected)
    assert logq == 0.0


# --- fm_shift -------------------------------------------------------------


def test_fm_shift_quantum_is_T_over_year_in_bins(rng):
    fm = T / YEAR
    scales = []
    for _ in range(600):
        out, lq = fm_shift(P0, rng, T)
        assert lq == 0.0
        s = (out[0] - P0[0]) / fm
        assert s == pytest.approx(round(s), abs=1e-9)
        scales.append(int(round(s)))
    assert set(scales) == {-3, -2, -1, 1, 2, 3}, set(scales)


def test_fm_shift_also_jitters_longitude_and_nothing_else(rng):
    out, _ = fm_shift(P0, rng, T)
    changed = np.nonzero(~np.isclose(out, P0, atol=0))[0]
    assert set(changed.tolist()) == {0, 2}
    assert abs(out[2] - P0[2]) <= 0.2


# --- psi_phi --------------------------------------------------------------


def test_psi_phi_jump_keeps_the_two_to_one_ratio(rng):
    for _ in range(300):
        out, lq = psi_phi_jump(P0, rng)
        assert lq == 0.0
        changed = np.nonzero(~np.isclose(out, P0, atol=0))[0]
        assert set(changed.tolist()) <= {5, 6}
        d_psi, d_phi = out[5] - P0[5], out[6] - P0[6]
        assert abs(abs(d_psi) - abs(d_phi) / 2) < 1e-12


def test_psi_phi_signs_are_independent(rng):
    """Both relative signs of the continuous angular shift are sampled."""
    same = 0
    n = 600
    for _ in range(n):
        out, _ = psi_phi_jump(P0, rng)
        if np.sign(out[5] - P0[5]) == np.sign(out[6] - P0[6]):
            same += 1
    assert 0.4 * n < same < 0.6 * n, f"{same}/{n} -- signs are not independent"


@pytest.mark.parametrize("scale, invariant", [(np.pi, True), (np.pi / 3, False)])
def test_psi_phi_discrete_symmetry_does_not_extend_to_continuous_shifts(
    scale, invariant
):
    """Only the discrete shift preserves a waveform at generic inclination."""
    from binary_galactica.glass.waveform import ucb_waveform

    class AngularRNG:
        """Choose the angular scale and positive signs without sampling error."""

        def __init__(self):
            self.values = iter([scale / (2 * np.pi), 0.75, 0.75])

        def uniform(self):
            return next(self.values)

    proposed, logq = psi_phi_jump(P0, AngularRNG())
    original_wave = ucb_waveform(P0, T=T, BW=32)
    proposed_wave = ucb_waveform(proposed, T=T, BW=32)
    assert logq == 0.0
    for channel in ("X", "Y", "Z", "A", "E"):
        original = np.asarray(original_wave[channel])
        delta = np.asarray(proposed_wave[channel]) - original
        relative_error = np.linalg.norm(delta) / np.linalg.norm(original)
        if invariant:
            assert relative_error < 1e-11
        else:
            assert relative_error > 0.1


# --- prior draws and the mix ---------------------------------------------


def test_prior_draw_lands_in_the_box(rng):
    pr = ucb_prior(T=T, fmin=0.00098376, fmax=0.00101624)
    d = PriorDraw(pr)
    for _ in range(100):
        out, lq = d(P0, rng)
        assert lq == 0.0 and pr.contains(out)


def test_extrinsic_draw_holds_f0_amp_fdot(rng):
    pr = ucb_prior(T=T, fmin=0.00098376, fmax=0.00101624)
    d = ExtrinsicPriorDraw(pr)
    out, _ = d(P0, rng)
    for held in (0, 3, 7):
        assert out[held] == P0[held]
    assert not np.allclose(out[[1, 2, 4, 5, 6]], P0[[1, 2, 4, 5, 6]])


def test_mix_selects_in_proportion_to_weight(rng):
    calls = {k: 0 for k in ("fisher", "fm_shift", "psi_phi", "prior")}

    def make(name):
        def f(p, r):
            calls[name] += 1
            return np.asarray(p), 0.0

        return f

    mix = ProposalMix({k: make(k) for k in calls})
    for _ in range(6000):
        mix(P0, rng)
    total = sum(calls.values())
    # DEFAULT_WEIGHTS resolves fisher as the remainder: 1 - 3*0.1 = 0.7.
    assert calls["fisher"] / total == pytest.approx(0.7, abs=0.03)
    assert calls["prior"] / total == pytest.approx(0.1, abs=0.03)


def test_mix_rejects_a_proposal_with_no_weight():
    with pytest.raises(ValueError, match="no weight"):
        ProposalMix({"mystery": lambda p, r: (p, 0.0)})


# One guard. A negative weight with a positive total is the only case that
# no other condition (finite weights, finite positive total) would catch.
def test_mix_requires_finite_nonnegative_weights_and_finite_positive_total():
    weights = {"first": -1.0, "second": 2.0}
    with pytest.raises(ValueError, match="weights.*non-negative.*positive"):
        ProposalMix({name: lambda p, r: (p, 0.0) for name in weights}, weights)


def test_mix_snapshots_the_proposal_mapping(rng):
    proposals = {"only": lambda p, r: (p.copy(), 0.0)}
    mix = ProposalMix(proposals, {"only": 1.0})
    proposals.clear()
    out, logq, name = mix(P0, rng)
    assert np.array_equal(out, P0)
    assert logq == 0.0
    assert name == "only"


def test_fisher_proposal_wraps_an_oracle(rng):
    seen = []

    def oracle(p):
        seen.append(np.asarray(p).copy())
        return np.full(8, 9.0), np.eye(8)

    out, lq = FisherProposal(oracle)(P0, rng)
    assert lq == 0.0 and len(seen) == 1
    assert np.allclose(seen[0], P0)


# --- single-source configuration -----------------------------------------


def _fisher_oracle(calls=None):
    def f(p):
        if calls is not None:
            calls.append(np.asarray(p).copy())
        return np.full(8, 1e4), np.eye(8)

    return f


def test_single_source_mix_gives_fisher_the_remainder():
    """Check that GLASS gives the Fisher proposal the remaining weight.

    F-statistic raises it from 0.3 to 0.7 instead of rescaling the others.
    """
    from binary_galactica.glass.prior import ucb_prior
    from binary_galactica.glass.proposals import single_source_mix

    pr = ucb_prior(T=T, fmin=0.00098376, fmax=0.00101624)
    mix = single_source_mix(prior=pr, fisher=_fisher_oracle(), T=T)
    w = dict(zip(mix.names, mix.p, strict=True))
    assert w["fisher"] == pytest.approx(0.7)
    for k in ("fm_shift", "psi_phi", "prior"):
        assert w[k] == pytest.approx(0.1)
    assert sum(w.values()) == pytest.approx(1.0)


def test_single_source_mix_runs_and_stays_finite(rng):
    from binary_galactica.glass.prior import ucb_prior
    from binary_galactica.glass.proposals import single_source_mix

    pr = ucb_prior(T=T, fmin=0.00098376, fmax=0.00101624)
    mix = single_source_mix(prior=pr, fisher=_fisher_oracle(), T=T)
    p = P0.copy()
    used = set()
    for _ in range(2000):
        out, lq, name = mix(p, rng)
        used.add(name)
        assert lq == 0.0, "the standard mix reports zero proposal correction"
        assert np.all(np.isfinite(out))
        if pr.contains(out):
            p = pr.wrap(out)
    assert used == {"fisher", "fm_shift", "psi_phi", "prior"}


def test_extra_proposal_requires_a_weight():
    from binary_galactica.glass.prior import ucb_prior
    from binary_galactica.glass.proposals import single_source_mix

    pr = ucb_prior(T=T, fmin=0.00098376, fmax=0.00101624)
    with pytest.raises(ValueError, match="no weight"):
        single_source_mix(
            prior=pr,
            fisher=_fisher_oracle(),
            T=T,
            extra={"fstat": lambda p, r: (p, 0.0)},
        )


def test_weights_that_leave_nothing_for_fisher_are_rejected():
    from binary_galactica.glass.prior import ucb_prior
    from binary_galactica.glass.proposals import single_source_mix

    pr = ucb_prior(T=T, fmin=0.00098376, fmax=0.00101624)
    with pytest.raises(ValueError, match="remainder"):
        single_source_mix(
            prior=pr,
            fisher=_fisher_oracle(),
            T=T,
            weights={"prior": 0.5, "fm_shift": 0.3, "psi_phi": 0.3},
        )


# --- the Fisher refresh cadence ------------------------------------------


def test_fisher_cache_uses_the_configured_refresh_interval():
    """The donor-compatible helper counts calls to the Fisher oracle wrapper."""
    from binary_galactica.glass.proposals import FISHER_REFRESH_STEPS, FisherCache

    calls = []
    cache = FisherCache(_fisher_oracle(calls))
    assert FISHER_REFRESH_STEPS == 80
    for i in range(FISHER_REFRESH_STEPS):
        cache(P0 + i)
    assert len(calls) == 1
    cache(P0)
    assert len(calls) == 2


def test_fisher_cache_refreshes_at_the_position_it_is_called_with():
    from binary_galactica.glass.proposals import FisherCache

    calls = []
    # A NumPy integer interval; plain ints are used throughout this file.
    cache = FisherCache(_fisher_oracle(calls), every=np.uint64(2))
    cache(P0)
    cache(P0 + 1.0)  # cached, oracle not called
    cache(P0 + 2.0)  # refresh, at THIS position
    assert len(calls) == 2
    assert np.allclose(calls[1], P0 + 2.0)


def test_fisher_cache_requires_a_positive_integer_refresh_interval():
    # Wiring of the shared integer check, whose conditions are covered on
    # ucb_bandwidth's N.
    from binary_galactica.glass.proposals import FisherCache

    with pytest.raises(ValueError, match="every must be a positive integer"):
        FisherCache(_fisher_oracle(), every=0)


def test_fisher_cache_invalidate_forces_a_refresh():
    from binary_galactica.glass.proposals import FisherCache

    calls = []
    cache = FisherCache(_fisher_oracle(calls), every=1000)
    cache(P0)
    cache(P0)
    assert len(calls) == 1
    cache.invalidate()
    cache(P0)
    assert len(calls) == 2


def test_mix_wraps_a_bare_oracle_in_the_cache():
    from binary_galactica.glass.prior import ucb_prior
    from binary_galactica.glass.proposals import single_source_mix

    pr = ucb_prior(T=T, fmin=0.00098376, fmax=0.00101624)
    calls = []
    mix = single_source_mix(prior=pr, fisher=_fisher_oracle(calls), T=T)
    r = np.random.default_rng(0)
    fisher_draws = 0
    for _ in range(400):
        _, _, name = mix(P0, r)
        fisher_draws += name == "fisher"
    assert 0 < fisher_draws < 400
    # Other mixture choices never advance the cache's Fisher-call counter.
    assert len(calls) == (fisher_draws + 79) // 80
