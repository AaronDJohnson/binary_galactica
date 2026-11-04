from types import SimpleNamespace

import numpy as np
import pytest

from binary_galactica.glass import draw_from_fisher, ucb_prior
from binary_galactica.segments import (
    FisherBank,
    FisherMove,
    FmShiftMove,
    PriorRedrawMove,
    PsiPhiMove,
    SegmentPrior,
    fisher_decomposition,
    floor_eigenvalues,
)

from .support import rj_impulse

T = 31_536_000.0
PRIOR = ucb_prior(T=T, fmin=1000 / T, fmax=1100 / T)
WIDTHS = PRIOR.hi - PRIOR.lo
K_MAX = 2


class _Layout:
    """The RJLayout slot convention for n_shared = 0 and 8 parameters per slot."""

    ndim = 8 * K_MAX

    @staticmethod
    def k(row):
        return int(row[-1])

    @staticmethod
    def slot(j):
        return slice(8 * j, 8 * j + 8)


def _stats(row, rng, chain=0):
    view = np.asarray(row, dtype=float).view()
    view.flags.writeable = False
    return SimpleNamespace(
        current_sample=view, layout=_Layout, rng=rng, chain_index=chain
    )


def _covering_bank(rng):
    """Fisher matrices built in width-normalised coordinates.

    The f0 scale is one bin, so proposals cross the 2-bin bank cells, and the
    rotation is a small perturbation of the identity so the fdot width does
    not crush the steps.
    """
    bank = FisherBank(1, match_bins=4.0)
    qs = np.arange(1000.0, 1101.0, 2.0)
    scale = np.full(8, 0.3)
    scale[0] = 1.0 / WIDTHS[0]
    Dinv = np.diag(1.0 / WIDTHS)
    evalues, evectors = [], []
    for _ in qs:
        Q, _ = np.linalg.qr(np.eye(8) + 0.05 * rng.standard_normal((8, 8)))
        F = Dinv @ Q @ np.diag(1.0 / scale**2) @ Q.T @ Dinv
        ev, vec = fisher_decomposition(F, WIDTHS)
        evalues.append(ev)
        evectors.append(vec)
    bank.store(0, qs, np.array(evalues), np.array(evectors))
    return bank


class _NoDeclineFisherMove(FisherMove):
    """FisherMove without the keep-the-matrix decline rule (test control)."""

    def propose(self, source, rng, chain):
        i = self.bank.lookup(chain, float(source[0]))
        if i is None:
            return None
        evalues, evectors = self.bank.decomposition(chain, i)
        return draw_from_fisher(source, rng, evalues, evectors)[0]


def _moves(rng):
    return {
        "fisher": FisherMove(_covering_bank(rng)),
        "prior": PriorRedrawMove(PRIOR),
        "fm_shift": FmShiftMove(T),
        "psi_phi": PsiPhiMove(),
    }


def test_segment_prior_matches_box_prior():
    rng = np.random.default_rng(1)
    sp = SegmentPrior(PRIOR, k_max=3)
    x = np.full((2, 24), np.nan)
    x[0, :16] = PRIOR.sample(rng, size=2).ravel()
    x[1, :8] = PRIOR.sample(rng)
    out = sp(x, np.array([2, 1]))
    expected = -np.log(4) + np.array(
        [
            PRIOR.log_prior(x[0, :8]) + PRIOR.log_prior(x[0, 8:16]),
            PRIOR.log_prior(x[1, :8]),
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-14)
    bad = x.copy()
    bad[0, 4] = 1.2
    assert sp(bad, np.array([2, 1]))[0] == -np.inf
    assert sp(x, np.array([4, 1]))[0] == -np.inf
    assert SegmentPrior(PRIOR, k_max=3, k_min=2)(x, np.array([2, 1]))[1] == -np.inf


def test_floor_eigenvalues_bounds_unit_steps():
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    ev = floor_eigenvalues(np.array([1e-30, -1.0, np.nan, 1, 2, 3, 4, 5]), Q, WIDTHS)
    steps = np.abs(Q) / np.sqrt(ev)[None, :]
    assert np.all(steps <= WIDTHS[:, None] * (1 + 1e-12))
    assert fisher_decomposition(np.full((8, 8), np.nan), WIDTHS) is None


def test_bank_lookup_nearest_within_range():
    bank = FisherBank(2, match_bins=4.0)
    bank.store(
        1, np.array([1010.0, 1020.0]), np.ones((2, 8)), np.tile(np.eye(8), (2, 1, 1))
    )
    assert bank.lookup(1, 1013.9) == 0
    assert bank.lookup(1, 1016.0) == 1
    assert bank.lookup(1, 1030.0) is None
    assert bank.lookup(0, 1010.0) is None
    bank.clear()
    assert bank.lookup(1, 1010.0) is None


@pytest.mark.parametrize("name", ["fisher", "prior", "fm_shift", "psi_phi"])
def test_move_contract(name):
    rng = np.random.default_rng(3)
    move = _moves(rng)[name]
    row = np.full(8 * K_MAX + 1, np.nan)
    row[-1] = 0
    out, qxy = move(_stats(row, rng))
    assert qxy == 0.0 and np.array_equal(out, row, equal_nan=True)
    row[:8] = PRIOR.sample(rng)
    row[-1] = 1
    for _ in range(50):
        out, qxy = move(_stats(row, rng))
        assert (
            out[-1] == 1
            and np.all(np.isnan(out[8:-1]))
            and np.all(np.isfinite(out[:8]))
        )
        assert not np.shares_memory(out, row)


def _count_switches(move, bank, rng, n=500):
    declined = switched = 0
    for _ in range(n):
        src = PRIOR.sample(rng)
        before = bank.lookup(0, src[0])
        new = move.propose(src.copy(), rng, 0)
        if new is None:
            declined += 1
        elif bank.lookup(0, new[0]) != before:
            switched += 1
    return declined, switched


def test_fisher_move_keeps_its_matrix():
    rng = np.random.default_rng(4)
    bank = _covering_bank(rng)
    declined, switched = _count_switches(FisherMove(bank), bank, rng)
    assert switched == 0
    assert declined >= 1
    # Control: without the decline rule the same loop does change matrix.
    _, switched_ctl = _count_switches(_NoDeclineFisherMove(bank), bank, rng)
    assert switched_ctl >= 1


@pytest.mark.parametrize("name", ["fisher", "prior", "fm_shift", "psi_phi"])
def test_one_step_leaves_prior_invariant(name):
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(5)
    move = _moves(rng)[name]
    sp = SegmentPrior(PRIOR, k_max=K_MAX, k_min=K_MAX)
    n = 4000
    x = PRIOR.sample(rng, size=n * K_MAX).reshape(n, 8 * K_MAX)
    out = np.empty_like(x)
    for i in range(n):
        new, _ = move(_stats(np.append(x[i], K_MAX), rng))
        y = PRIOR.wrap(new[:-1].reshape(K_MAX, 8)).reshape(-1)
        out[i] = y if np.isfinite(sp(y[None], np.array([K_MAX]))[0]) else x[i]
    for d in range(8 * K_MAX):
        lo, hi = PRIOR.lo[d % 8], PRIOR.hi[d % 8]
        p = stats.kstest((out[:, d] - lo) / (hi - lo), "uniform").pvalue
        assert p > 1e-4, (name, d, p)


def test_moves_register_on_rjptsampler(tmp_path):
    impulse = rj_impulse()
    rng = np.random.default_rng(6)
    sampler = impulse.RJPTSampler(
        n_shared=0,
        component_dim=8,
        max_components=K_MAX,
        min_components=K_MAX,
        lnlike=lambda x, k: np.zeros(len(x)),
        lnprior=SegmentPrior(PRIOR, k_max=K_MAX, k_min=K_MAX),
        component_proposal=impulse.IndependentComponentProposal(
            draw=lambda r: PRIOR.sample(r), logpdf=PRIOR.log_prior
        ),
        component_periodic={2: (0.0, 2 * np.pi), 5: (0.0, np.pi), 6: (0.0, 2 * np.pi)},
        vectorized=True,
        ntemps=1,
        am_weight=0,
        scam_weight=0,
        de_weight=0,
        outdir=str(tmp_path),
        seed=1,
        verbose=False,
    )
    for move in _moves(rng).values():
        sampler.add_custom_jump(move, 1.0)
    sampler.sample(PRIOR.sample(rng, size=K_MAX).ravel(), 200, initial_k=K_MAX)
    rates = sampler.proposal_acceptance_rates()
    names = {"FisherMove", "PriorRedrawMove", "FmShiftMove", "PsiPhiMove"}
    assert names <= set(rates)
    for name in names:
        assert rates[name]["calls"] > 0, name
