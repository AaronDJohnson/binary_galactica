"""Full-band dry run: segment table, compile keys, memory and time per turn.

The stand-in start set is every dgb/igb catalogue row with analytic SNR > 7
(no confusion noise); the user's noise block supplies the real one. Timing runs
one cold turn (which includes compilation) and one warm turn on a few segments
spread across the band. Data: obs minus MBHB and VGB, with a running-median
covariance. The warm per-step time is extrapolated to every segment with a
least-squares fit in ntemps * K * BW.

    PYTHONPATH=src:... python tools/segments/dry_run.py --sample 6 --steps 100
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


def _stats(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
            "mean": float(v.mean())}


def run(sample: int, steps: int, seed: int, path: Path) -> dict:
    """Build the table, time sampled segments, and return the report."""
    n, dt = sangria.grid(path)
    T, n_freq = n * dt, n // 2 + 1
    rows = runs.catalogue_rows(path, T=T, snr_min=7.0)
    native = ldc_to_native(rows, T)
    config = UCBBlockConfig(steps_per_turn=steps)
    start = time.perf_counter()
    table = build_segment_table(native, T=T, n_freq=n_freq, config=config)
    table_s = time.perf_counter() - start
    segs = table.segments
    for s, seg in enumerate(segs):
        left = segs[s - 1].pad if s else 0
        right = segs[s + 1].pad if s < len(segs) - 1 else 0
        if seg.width < max(seg.pad, left + right):
            raise AssertionError(f"segment {s} violates the padding rule")
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
        raise ValueError("--sample must be at least 3 (evenly spaced picks plus the heaviest mid-band two)")
    cost = np.array([seg.k_max * quantize_bins(seg.n_bins) for seg in segs], dtype=np.float64)
    heavy = [int(i) for i in np.argsort(-cost[:-1])[:2]]  # heaviest by K * n_pad, tail excluded
    even = np.linspace(0, len(segs) - 1, sample - 2).round().astype(int)
    pick = np.unique(np.concatenate([even, heavy]))
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
        runner = SegmentRunner(seg, own, T=T, t0=0.0, config=config, outdir=work / f"seg{s}",
                               seed=seed + int(s), reference=c)
        t1 = time.perf_counter()
        runner.advance(r, c)
        first = time.perf_counter() - t1
        t2 = time.perf_counter()
        runner.advance(r, c)
        steady = time.perf_counter() - t2
        archived = sum(p.stat().st_size for p in runner.archive.files())
        rates = runner.sampler.proposal_acceptance_rates()
        timings.append({"segment": int(s), "frequency_hz": 0.5 * (seg.lo + seg.hi) / T,
                        f"measured_archive_bytes_per_turn_at_{steps}_steps": archived // 2,
                        "acceptance": {name: {"calls": r["calls"], "rate": r["rate"]}
                                       for name, r in rates.items() if r["calls"]},
                        "bw": seg.bw, "n_pad": quantize_bins(seg.n_bins),
                        "k_max": seg.k_max, "owned": len(own), "first_turn_s": first,
                        "steady_turn_s": steady, "per_step_s": steady / steps})
    from scipy.optimize import nnls

    n_pads = np.array([quantize_bins(seg.n_bins) for seg in segs], dtype=np.float64)
    ks_all = ks.astype(np.float64)
    bws = np.array([seg.bw for seg in segs], dtype=np.float64)
    y = np.array([t["per_step_s"] for t in timings])
    heavy_idx = [i for i, t in enumerate(timings) if t["segment"] in heavy]
    tk = np.array([t["k_max"] for t in timings], dtype=np.float64)
    tn = np.array([t["n_pad"] for t in timings], dtype=np.float64)
    tb = np.array([t["bw"] for t in timings], dtype=np.float64)
    # Model 1 (primary): per step = a*K + b*K*n_pad, non-negative least squares.
    coef, _ = nnls(np.stack([tk, tk * tn], axis=1), y)
    per_seg = coef[0] * ks_all + coef[1] * ks_all * n_pads
    tail = n_pads == n_pads.max()
    models = {"k_npad_nnls": float(per_seg.sum() * full_steps)}
    # Model 2: bulk cost scales with K (mean measured seconds per step per K over the
    # samples below the heavy threshold); every segment at or above the smallest heavy
    # sample's n_pad takes the measured time of the sample nearest in n_pad, scaled by K.
    thr = min(tn[heavy_idx])
    heavy_mask = n_pads >= thr
    bulk_t = [t for t in timings if t["n_pad"] < thr]
    hv_t = [t for t in timings if t["n_pad"] >= thr]
    bulk_rate = float(np.mean([t["per_step_s"] / t["k_max"] for t in bulk_t]))
    nearest = [min(hv_t, key=lambda t: abs(np.log(t["n_pad"] / n)))
               for n in n_pads[heavy_mask]]
    heavy_step = sum(t["per_step_s"] * k / t["k_max"] for t, k in zip(nearest, ks_all[heavy_mask]))
    models["bulk_k_scaled_plus_measured_heavy"] = float(
        (bulk_rate * ks_all[~heavy_mask].sum() + heavy_step) * full_steps)
    tail_t = [t for t in timings if t["n_pad"] == n_pads.max()]
    tail_step = float(np.mean([t["per_step_s"] for t in tail_t])) if tail_t else float("nan")
    # Model 3 (earlier, kept for comparison): a + b*ntemps*K*BW, unconstrained.
    x = np.stack([np.ones_like(tk), ntemps * tk * tb], axis=1)
    c3 = np.linalg.lstsq(x, y, rcond=None)[0]
    models["ntemps_k_bw_lstsq"] = float(np.sum(c3[0] + c3[1] * ntemps * ks_all * bws) * full_steps)
    tail_s = float(per_seg[tail].sum() * full_steps)
    predicted = models["k_npad_nnls"]
    return {
        "T": T,
        "start_sources": int(native.shape[0]),
        "n_segments": len(segs),
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
        "memory_counted": "chain positions+counts and Fisher bank at k_max sources per chain, "
                          "float64, summed over segments; impulse short_chain, JAX executables and the "
                          "tool's full-band spectra excluded",
        "archive_bytes_per_turn": archive,
        "archive_note": "archive_bytes_per_turn is an uncompressed upper bound at 1000 steps, "
                        "thin 1; the measured_* per-sample keys are at the sampled step count",
        "timings": timings,
        "fit_seconds_per_step": {"per_slot": float(coef[0]), "per_slot_window_bin": float(coef[1])},
        "tail": {"segments": int(tail.sum()), "n_pad": float(n_pads.max()),
                 "predicted_turn_s": tail_s, "share_of_total": tail_s / predicted,
                 "measured_per_step_s": tail_step},
        "predicted_turn_s_by_model": models,
        "predicted_turn_s_range": [min(models.values()), max(models.values())],
        "steps_per_turn_predicted": full_steps,
        "predicted_turn_s": predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=6)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--path", type=Path, default=sangria.DEFAULT_PATH)
    args = parser.parse_args()
    report = run(args.sample, args.steps, args.seed, args.path)
    path = runs.write_report("segments-dry-run.json", report)
    print(f"{report['start_sources']} start sources -> {report['n_segments']} segments, "
          f"{report['compile_keys']} compile keys")
    print(f"width {report['width']}  pad {report['pad']}  K {report['k_max']}")
    print(f"memory ~{report['memory_bytes'] / 2**30:.2f} GiB, archive "
          f"~{report['archive_bytes_per_turn'] / 2**30:.2f} GiB per turn (uncompressed)")
    for t in report["timings"]:
        print(f"  seg {t['segment']:5d} f={t['frequency_hz']*1e3:.3f}mHz BW={t['bw']:4d} K={t['k_max']:3d} "
              f"cold {t['first_turn_s']:.1f}s warm {t['steady_turn_s']:.2f}s")
    lo, hi = report["predicted_turn_s_range"]
    print(f"predicted full-band turn at {report['steps_per_turn_predicted']} steps: "
          f"{lo / 3600:.1f}-{hi / 3600:.1f} h across models")
    for name, v in report["predicted_turn_s_by_model"].items():
        print(f"  {name}: {v / 3600:.2f} h")
    tl = report["tail"]
    print(f"  tail ({tl['segments']} seg, n_pad {tl['n_pad']:.0f}): {tl['predicted_turn_s'] / 60:.0f} min "
          f"per turn, {100 * tl['share_of_total']:.0f}% of the k_npad_nnls total")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
