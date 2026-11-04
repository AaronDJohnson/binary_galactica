"""Fixed-count check on one isolated synthetic source over 12 months.

Inject one source (SNR ~38) into Gaussian noise with C = (T/2) ae_noise on a
single segment window, run SegmentRunner at fixed count, and compare the
cold-chain posterior with the injection and with the Fisher matrix.

    PYTHONPATH=src:... python tools/segments/isolated_source.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from binary_galactica.segments import COMPONENT_PERIODIC

# runs (hence h5py, via sangria) is imported inside the functions that need it,
# so ``fold_offsets`` imports without h5py; runs enables x64 first.

T = 31_536_000.0
N_FREQ = 3_153_601


def injection(f0: float = 3e-3, lnA: float = -50.0) -> np.ndarray:
    """Return the injected native source."""
    from binary_galactica.glass import ucb_fdot

    return np.array(
        [f0 * T + 0.3, 0.2, 1.1, lnA, 0.3, 0.7, 2.0, ucb_fdot(0.3, f0) * T**2]
    )


def fold_offsets(d: np.ndarray) -> np.ndarray:
    """Wrap offsets from the injection onto their periods, folding the exact symmetry.

    ``(psi, phi0) -> (psi + pi/2, phi0 + pi)`` flips the sign of the waveform
    twice over, so it leaves the likelihood unchanged and the posterior has two
    equal modes. Each sample is compared with the nearer image of the injection.
    """
    d = d.copy()
    for i, (_, period) in COMPONENT_PERIODIC.items():
        d[:, i] = (d[:, i] + period / 2) % period - period / 2
    flip = np.abs(d[:, 5]) > np.pi / 4
    d[flip, 5] -= np.sign(d[flip, 5]) * np.pi / 2
    d[flip, 6] = (d[flip, 6] + np.pi + np.pi) % (2 * np.pi) - np.pi
    return d


def run(
    *, turns: int = 20, steps: int = 1000, ntemps: int = 12, seed: int = 11, outdir=None
):
    """Run the check and return the report."""
    import runs  # enables x64 before the package is used  # noqa: F401

    from binary_galactica.glass import ae_noise, template_on_window
    from binary_galactica.segments import (
        SegmentRunner,
        UCBBlockConfig,
        build_segment_table,
    )

    src = injection()
    q = int(np.floor(src[0]))
    config = UCBBlockConfig(
        band_bins=(q - 40, q + 40),
        birth_death=False,
        steps_per_turn=steps,
        ntemps=ntemps,
    )
    seg = build_segment_table(src[None], T=T, n_freq=N_FREQ, config=config).segments[0]
    f = (seg.bin_lo + np.arange(seg.n_bins)) / T
    c = np.stack([0.5 * T * ae_noise(f)] * 2)
    rng = np.random.default_rng(seed)
    noise = np.sqrt(c / 2) * (
        rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)
    )
    h = np.asarray(
        template_on_window(
            src,
            T=T,
            bin_lo=seg.bin_lo,
            n_bins=seg.n_bins,
            BW=seg.bw,
            backend=config.backend,
        )
    )
    outdir = (
        Path(outdir)
        if outdir is not None
        else Path(tempfile.mkdtemp(prefix="isolated-"))
    )
    runner = SegmentRunner(
        seg,
        src[None],
        T=T,
        t0=0.0,
        config=config,
        outdir=outdir,
        seed=seed,
        reference=c,
    )
    turn_seconds = []
    for _ in range(turns):
        start = time.perf_counter()
        runner.advance(h + noise, c)
        turn_seconds.append(time.perf_counter() - start)
    seconds = float(sum(turn_seconds))
    # Turn 0 includes the JIT compile; the steady-state rate excludes it.
    steady = turn_seconds[1:] or turn_seconds
    chain = runner.archive.load()
    x = chain["samples"][int(0.25 * chain["k"].size) :]
    d = x - src
    d = fold_offsets(d)
    mean, std = d.mean(axis=0), d.std(axis=0)
    cov = np.linalg.inv(runner.likelihood.fisher_matrices(src[None])[0])
    sigma = np.sqrt(np.diag(cov))
    ratio = std / sigma
    # The fdot prior box is narrower than the Fisher sigma at this frequency, and
    # f0, fdot and longitude are 0.9-0.99 correlated, so the prior truncates the
    # posterior. Compare with the Fisher Gaussian truncated to the same box.
    draws = rng.multivariate_normal(np.zeros(8), cov, size=400_000)
    wrapped = runner.prior.wrap(src + draws)
    inside = np.all((wrapped >= runner.prior.lo) & (wrapped <= runner.prior.hi), axis=1)
    trunc_mean = draws[inside].mean(axis=0)
    trunc_sigma = draws[inside].std(axis=0)
    trunc_ratio = std / trunc_sigma
    checks = {
        "mean_within_4_std": bool(np.all(np.abs((mean - trunc_mean) / std) < 4)),
        "widths_match_truncated_fisher": bool(
            np.all((trunc_ratio > 0.6) & (trunc_ratio < 1.6))
        ),
    }
    return {
        "injection": src,
        "snr": float(runner.likelihood.snr(src[None])[0]),
        "segment": {"bin_lo": seg.bin_lo, "n_bins": seg.n_bins, "bw": seg.bw},
        "mean_offset": mean,
        "std": std,
        "fisher_sigma": sigma,
        "width_ratio": ratio,
        "truncated_fisher_sigma": trunc_sigma,
        "truncated_width_ratio": trunc_ratio,
        "truncated_fisher_mean": trunc_mean,
        "prior_inside_fraction": float(inside.mean()),
        "checks": checks,
        "turns": turns,
        "steps": steps,
        "ntemps": ntemps,
        "seconds": seconds,
        "seconds_per_step": float(np.mean(steady)) / steps,
        "seconds_per_step_incl_compile": seconds / (turns * steps),
        "acceptance": runner.sampler.proposal_acceptance_rates(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ntemps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    import runs

    report = run(turns=args.turns, steps=args.steps, ntemps=args.ntemps, seed=args.seed)
    path = runs.write_report("segments-isolated.json", report)
    names = ["f0T", "sinb", "lam", "lnA", "cosi", "psi", "phi0", "fdotT2"]
    for i, name in enumerate(names):
        print(
            f"{name:7s} offset={report['mean_offset'][i]:+.3e} "
            f"std={report['std'][i]:.3e} "
            f"fisher={report['fisher_sigma'][i]:.3e} "
            f"ratio={report['width_ratio'][i]:.2f} "
            f"trunc-ratio={report['truncated_width_ratio'][i]:.2f}"
        )
    print(
        f"snr={report['snr']:.1f}  "
        f"{report['seconds_per_step'] * 1e3:.2f} ms/step  -> {path}"
    )
    ok = all(report["checks"].values())
    print("PASS" if ok else f"FAIL {report['checks']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
