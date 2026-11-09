"""Full-band dry run: segment table, compile keys, memory and time per turn.

The stand-in start set is every dgb/igb catalogue row with analytic SNR > 7
(no confusion noise); the user's noise block supplies the real one. Timing runs
one cold turn (which includes compilation) and three warm turns, keeping the last
as the steady time, on a few segments spread across the band. Data: obs minus
MBHB and VGB, with a running-median covariance. The warm per-step time is
extrapolated to every segment with a non-negative least-squares (NNLS) fit, the
primary model. Its four terms follow the cached likelihood's cost:
- a fixed cost per step, for impulse's loop and the moves;
- about one new template per chain, which scales with BW;
- assembly over every slot, which scales with K * BW;
- the window's inner product, which scales with n_pad.
A model-free check gives each segment the time of the nearest sample. The
sampled segments are evenly spaced picks, the most crowded (largest K * BW),
the widest window, and the "tail": the segment with the widest padded window.
Turns run at ``steps_per_turn`` steps by default, so the fixed per-turn cost
(Fisher refresh, archive) is amortized as in a run.

``--skip-gaps`` and ``--max-interior-bins N`` set the matching
``UCBBlockConfig`` options; ``--out`` names the report under tools/performance.

    PYTHONPATH=src:... python tools/segments/dry_run.py --sample 6
    PYTHONPATH=src:... python tools/segments/dry_run.py --skip-gaps \
        --out segments-dry-run-skip-gaps.json
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import runs
import sangria

from binary_galactica.segments import (
    SegmentLikelihood,
    SegmentRunner,
    UCBBlockConfig,
    build_segment_table,
    ldc_to_native,
    quantize_bins,
)

# The first warm turns still compile the template renderer's chunk sizes (a
# run pays that once per process), so the steady time is the last of these.
WARM_TURNS = 3


def _stats(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "min": float(v.min()),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "mean": float(v.mean()),
    }


def run(
    sample: int,
    steps: int,
    seed: int,
    path: Path,
    *,
    skip_gaps: bool = False,
    max_interior_bins: int | None = None,
) -> dict:
    """Build the table, time sampled segments, and return the report."""
    n, dt = sangria.grid(path)
    T, n_freq = n * dt, n // 2 + 1
    rows = runs.catalogue_rows(path, T=T, snr_min=7.0)
    native = ldc_to_native(rows, T)
    config = UCBBlockConfig(
        steps_per_turn=steps, skip_gaps=skip_gaps, max_interior_bins=max_interior_bins
    )
    start = time.perf_counter()
    table = build_segment_table(native, T=T, n_freq=n_freq, config=config)
    table_s = time.perf_counter() - start
    segs = table.segments
    for s, seg in enumerate(segs):
        # A neighbour across a gap (skip_gaps) counts as absent.
        left = segs[s - 1].pad if s and segs[s - 1].hi == seg.lo else 0
        right = segs[s + 1].pad if s < len(segs) - 1 and segs[s + 1].lo == seg.hi else 0
        if seg.width < max(seg.pad, left + right):
            raise AssertionError(f"segment {s} violates the padding rule")
    clusters = 1 + sum(a.hi != b.lo for a, b in zip(segs[:-1], segs[1:], strict=True))
    interior = sum(seg.width for seg in segs)
    band = segs[-1].hi - segs[0].lo
    widest = max(segs, key=lambda seg: seg.width)
    keys = Counter((seg.bw, quantize_bins(seg.n_bins), seg.k_max) for seg in segs)
    ntemps = config.ntemps
    dims = np.array([8 * seg.k_max for seg in segs])
    # Resident between turns (impulse's short_chain is freed after each turn): chain
    # positions + counts, and the Fisher bank (8 eigenvalues + 8x8 eigenvectors per
    # source, at most k_max per chain, so an upper bound), per segment, in float64.
    ks = np.array([seg.k_max for seg in segs])
    memory = int(np.sum(ntemps * (dims + 1 + 72 * ks) * 8))
    full_steps = UCBBlockConfig().steps_per_turn
    archive = int(np.sum((full_steps // config.thin) * (dims + 2) * 8))

    spectra = sangria.aet_spectra(*sangria.combine(["obs"], ["mbhb", "vgb"], path))
    if sample < 3:
        raise ValueError(
            "--sample must be at least 3 "
            "(evenly spaced picks plus the most crowded and the widest window)"
        )
    # The tail is the widest padded window: the last segment under the default
    # tiling, but anywhere with skip_gaps. It is always sampled. The two heavy
    # picks span the fit's other terms: the most crowded segment (K * BW) and the
    # widest window besides the tail (n_pad).
    tail_index = int(np.argmax([seg.n_bins for seg in segs]))
    k_bw = np.array([seg.k_max * seg.bw for seg in segs], dtype=np.float64)
    n_pad = np.array([quantize_bins(seg.n_bins) for seg in segs], dtype=np.float64)
    crowded = next(int(i) for i in np.argsort(-k_bw, kind="stable") if i != tail_index)
    wide = next(
        int(i)
        for i in np.argsort(-n_pad, kind="stable")
        if i not in (tail_index, crowded)
    )
    heavy = [crowded, wide]
    even = np.linspace(0, len(segs) - 1, sample - 2).round().astype(int)
    pick = np.unique(np.concatenate([even, heavy, [tail_index]]))
    work = Path(tempfile.mkdtemp(prefix="dry-run-"))
    timings = []
    for s in pick:
        seg = segs[s]
        lo, hi = seg.bin_lo, seg.bin_hi
        var = runs.running_median_variance({ch: spectra[ch] for ch in "AE"}, lo, hi)
        c = np.stack([var["A"], var["E"]])
        r = np.stack([spectra["A"][lo:hi], spectra["E"][lo:hi]])
        own = native[list(seg.owned)]
        if len(own):
            lik = SegmentLikelihood.for_segment(seg, T=T, t0=0.0, config=config)
            lik.set_reference(c)
            own = own[lik.snr(own) >= config.snr_min]
        runner = SegmentRunner(
            seg,
            own,
            T=T,
            t0=0.0,
            config=config,
            outdir=work / f"seg{s}",
            seed=seed + int(s),
            reference=c,
        )
        turns = []
        for _ in range(1 + WARM_TURNS):
            t1 = time.perf_counter()
            runner.advance(r, c)
            turns.append(time.perf_counter() - t1)
        first, warm = turns[0], turns[1:]
        steady = warm[-1]
        archived = sum(p.stat().st_size for p in runner.archive.files())
        rates = runner.sampler.proposal_acceptance_rates()
        timings.append(
            {
                "segment": int(s),
                "frequency_hz": 0.5 * (seg.lo + seg.hi) / T,
                f"measured_archive_bytes_per_turn_at_{steps}_steps": archived
                // len(turns),
                "acceptance": {
                    name: {"calls": r["calls"], "rate": r["rate"]}
                    for name, r in rates.items()
                    if r["calls"]
                },
                "bw": seg.bw,
                "n_pad": quantize_bins(seg.n_bins),
                "k_max": seg.k_max,
                "owned": len(own),
                "first_turn_s": first,
                "warm_turns_s": warm,
                "steady_turn_s": steady,
                "per_step_s": steady / steps,
            }
        )
    from scipy.optimize import nnls

    ks_all = ks.astype(np.float64)
    bws = np.array([seg.bw for seg in segs], dtype=np.float64)

    def basis(k, bw, window):
        # The cached likelihood's cost per step (see the module docstring).
        return np.stack([np.ones_like(k), bw, k * bw, window], axis=-1)

    y = np.array([t["per_step_s"] for t in timings])
    tk, tb, tn = (
        np.array([t[name] for t in timings], dtype=np.float64)
        for name in ("k_max", "bw", "n_pad")
    )
    # Model 1 (primary): non-negative least squares in the four cost terms.
    coef, _ = nnls(basis(tk, tb, tn), y)
    per_seg = basis(ks_all, bws, n_pad) @ coef
    tail = n_pad == n_pad.max()
    models = {"cached_nnls": float(per_seg.sum() * full_steps)}
    # Model 2 (model-free check): each segment takes the measured time of the
    # sample nearest in (log BW, log n_pad, log K).
    sampled = np.log(np.stack([tb, tn, tk], axis=1))
    every = np.log(np.stack([bws, n_pad, ks_all], axis=1))
    nearest = np.argmin(((every[:, None] - sampled[None]) ** 2).sum(axis=-1), axis=1)
    models["nearest_sample"] = float(y[nearest].sum() * full_steps)
    tail_t = [t for t in timings if t["n_pad"] == n_pad.max()]
    tail_step = (
        float(np.mean([t["per_step_s"] for t in tail_t])) if tail_t else float("nan")
    )
    tail_s = float(per_seg[tail].sum() * full_steps)
    predicted = models["cached_nnls"]
    return {
        "T": T,
        "skip_gaps": config.skip_gaps,
        "max_interior_bins": config.max_interior_bins,
        "excluded_names": list(runs.EXCLUDED_NAMES),
        "start_sources": int(native.shape[0]),
        "n_segments": len(segs),
        "n_clusters": clusters,
        "interior_bins_total": interior,
        "band_bins": band,
        "interior_fraction_of_band": interior / band,
        "widest_segment": {
            "segment": widest.index,
            "width": widest.width,
            "frequency_mhz": [widest.lo / T * 1e3, widest.hi / T * 1e3],
            "owned": len(widest.owned),
        },
        "outside_band": len(table.outside),
        "table_seconds": table_s,
        "compile_keys": len(keys),
        "compile_key_counts": {str(k): v for k, v in sorted(keys.items())},
        "width": _stats([seg.width for seg in segs]),
        "pad": _stats([seg.pad for seg in segs]),
        "n_bins": _stats([seg.n_bins for seg in segs]),
        "bw": dict(Counter(seg.bw for seg in segs)),
        "k_max": _stats([seg.k_max for seg in segs]),
        "owned": _stats([len(seg.owned) for seg in segs]),
        "memory_bytes": memory,
        "memory_counted": "chain positions+counts and Fisher bank at k_max sources "
        "per chain, float64, summed over segments; impulse short_chain, JAX "
        "executables, the likelihood's template cache (one segment at a time, "
        "cleared after each turn) and the tool's full-band spectra excluded",
        "archive_bytes_per_turn": archive,
        "archive_note": "archive_bytes_per_turn is an uncompressed upper bound at "
        "1000 steps, thin 1; the measured_* per-sample keys are at the sampled "
        "step count",
        "timings": timings,
        "fit_seconds_per_step": {
            "fixed": float(coef[0]),
            "per_bw_bin": float(coef[1]),
            "per_slot_bw_bin": float(coef[2]),
            "per_window_bin": float(coef[3]),
        },
        "timed_steps_per_turn": steps,
        "tail": {
            "segment": tail_index,
            "frequency_mhz": [
                segs[tail_index].lo / T * 1e3,
                segs[tail_index].hi / T * 1e3,
            ],
            "segments": int(tail.sum()),
            "n_pad": float(n_pad.max()),
            "predicted_turn_s": tail_s,
            "share_of_total": tail_s / predicted,
            "measured_per_step_s": tail_step,
        },
        "predicted_turn_s_by_model": models,
        "predicted_turn_s_range": [min(models.values()), max(models.values())],
        "steps_per_turn_predicted": full_steps,
        "predicted_turn_s": predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=6)
    parser.add_argument(
        "--steps",
        type=int,
        default=UCBBlockConfig().steps_per_turn,
        help="steps per timed turn (default: a run's steps_per_turn)",
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--path", type=Path, default=sangria.DEFAULT_PATH)
    parser.add_argument(
        "--skip-gaps", action="store_true", help="leave gaps between clusters unsampled"
    )
    parser.add_argument(
        "--max-interior-bins", type=int, default=None, help="soft cap on interior width"
    )
    parser.add_argument(
        "--out",
        default="segments-dry-run.json",
        help="report name in tools/performance",
    )
    args = parser.parse_args()
    report = run(
        args.sample,
        args.steps,
        args.seed,
        args.path,
        skip_gaps=args.skip_gaps,
        max_interior_bins=args.max_interior_bins,
    )
    path = runs.write_report(args.out, report)
    print(
        f"{report['start_sources']} start sources -> {report['n_segments']} segments "
        f"in {report['n_clusters']} clusters, {report['compile_keys']} compile keys "
        f"(skip_gaps={report['skip_gaps']}, "
        f"max_interior_bins={report['max_interior_bins']})"
    )
    wd = report["widest_segment"]
    print(
        f"interior {report['interior_bins_total']} of {report['band_bins']} band bins "
        f"({100 * report['interior_fraction_of_band']:.1f}%); widest segment "
        f"{wd['segment']}: {wd['width']} bins, {wd['frequency_mhz'][0]:.2f}-"
        f"{wd['frequency_mhz'][1]:.2f} mHz, {wd['owned']} owned"
    )
    print(f"width {report['width']}  pad {report['pad']}  K {report['k_max']}")
    print(
        f"memory ~{report['memory_bytes'] / 2**30:.2f} GiB, archive "
        f"~{report['archive_bytes_per_turn'] / 2**30:.2f} GiB per turn (uncompressed)"
    )
    for t in report["timings"]:
        print(
            f"  seg {t['segment']:5d} f={t['frequency_hz'] * 1e3:.3f}mHz "
            f"BW={t['bw']:4d} K={t['k_max']:3d} "
            f"cold {t['first_turn_s']:.1f}s warm {t['steady_turn_s']:.2f}s"
        )
    lo, hi = report["predicted_turn_s_range"]
    print(
        f"predicted full-band turn at {report['steps_per_turn_predicted']} steps: "
        f"{lo / 3600:.1f}-{hi / 3600:.1f} h across models"
    )
    for name, v in report["predicted_turn_s_by_model"].items():
        print(f"  {name}: {v / 3600:.2f} h")
    tl = report["tail"]
    print(
        f"  tail (seg {tl['segment']} at {tl['frequency_mhz'][0]:.2f} mHz; "
        f"{tl['segments']} seg, n_pad {tl['n_pad']:.0f}): "
        f"{tl['predicted_turn_s'] / 60:.0f} min "
        f"per turn, {100 * tl['share_of_total']:.0f}% of the cached_nnls total"
    )
    fit = report["fit_seconds_per_step"]
    print(
        "  fit per step: "
        + ", ".join(f"{name} {value:.3g} s" for name, value in fit.items())
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
