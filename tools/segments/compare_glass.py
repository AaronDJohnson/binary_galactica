"""Compare the block with injections and with GLASS's chains on a few segments.

GLASS's 12-month run used T = 31,457,280 s and Sangria spans 31,536,000 s, so
the comparison is made in Hz and ln amplitude, not in bins. Each GLASS segment's
detections form the core; the block's band extends SPAN bins either side of it
and starts from every catalogue source in the band with analytic SNR >= 7, so
loud neighbours are modelled instead of sitting in the core's padding.

    PYTHONPATH=src:... python tools/segments/compare_glass.py --segments 3
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import runs
import sangria

from binary_galactica.segments import UCBBlock, UCBBlockConfig

HALF_WIDTH = runs.HALF_WIDTH
NOISE_REFERENCE = runs.NOISE_REFERENCE

GLASS_DIR = Path(
    os.environ.get(
        "BG_GLASS_DIR",
        str(
            Path.home()
            / "Documents/lisa/project-catalog/data/legacy/old"
            / "glass_h5_files_sangria_12mo_UCB"
        ),
    )
)
MARGIN = runs.MARGIN
SPAN = 300


def glass_segments(count: int, fmin=3e-3, fmax=5e-3, nmin=8, nmax=20):
    """Return GLASS's detections and ``count`` segment indices.

    Each selected segment holds ``nmin..nmax`` of the detections.
    """
    import pandas as pd

    det = pd.read_hdf(GLASS_DIR / "UCB.h5", "detections")
    det = det[(det["Frequency"] >= fmin) & (det["Frequency"] < fmax)]
    sizes = det.groupby("segment").size()
    chosen = sorted(int(s) for s in sizes[(sizes >= nmin) & (sizes <= nmax)].index)[
        :count
    ]
    return det, chosen


def glass_chain(det, name):
    """Return GLASS's posterior chain of one detection."""
    import pandas as pd

    return pd.read_hdf(GLASS_DIR / det.loc[name, "chain file"], f"/{name}_chain")


def _counterpart(chain: dict, q: float, burn: float = 0.25, tol: float = 2.0):
    first = int(burn * chain["k"].size)
    comps = chain["samples"][first:].reshape(chain["samples"][first:].shape[0], -1, 8)
    d = np.abs(comps[..., 0] - q)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    rows = np.arange(comps.shape[0])
    hit = d[rows, j] < tol
    return comps[rows, j][hit], float(hit.mean()) if hit.size else 0.0


def _nearest_catalogue(path, T, q, start):
    """Return the catalogue row of any SNR nearest ``q`` (bins).

    Also return whether that row was a start source.
    """
    rows = runs.catalogue_rows(path, T=T, band=(q - 10, q + 10))
    if not len(rows):
        return None
    j = int(np.argmin(np.abs(rows["Frequency"] * T - q)))
    r = rows[j]
    return {
        "name": str(r["Name"]),
        "f": float(r["Frequency"]),
        "lnA": float(np.log(r["Amplitude"])),
        "analytic_snr": float(runs.analytic_snrs(r["Frequency"], r["Amplitude"], T)),
        "distance_bins": float(r["Frequency"] * T - q),
        "in_start_set": bool(r["Name"] in set(start["Name"])),
    }


def compare_segment(
    seg_index, det, spectra, data, *, T, turns, steps, seed, outdir, path, workers=1
):
    """Run the block on one GLASS segment's band and compare every detection."""
    rows_g = det[det["segment"] == seg_index]
    q = rows_g["Frequency"].to_numpy() * T
    core = (int(np.floor(q.min())) - 5, int(np.floor(q.max())) + 6)
    band, start = runs.target_band(path, T=T, core=core, span=SPAN)
    config = UCBBlockConfig(band_bins=band, steps_per_turn=steps)
    a, b = band[0] - MARGIN, band[1] + MARGIN
    resid, _ = runs.residual_spectra(
        spectra, path, T=T, lo=a - HALF_WIDTH, hi=b + HALF_WIDTH
    )
    cov = runs.covariance(
        data, runs.running_median_variance(resid, a, b, HALF_WIDTH), a, b
    )
    block = UCBBlock(
        grid=data,
        start_sources=start,
        outdir=outdir / f"glass_{seg_index}",
        config=config,
        reference_covariance=cov,
        workers=workers,
    )
    segs = block.table.segments
    if segs[0].bin_lo < a or segs[-1].bin_hi > b:
        raise RuntimeError(
            f"padded band {segs[0].bin_lo}..{segs[-1].bin_hi} exceeds "
            f"the running-median range {a}..{b}"
        )
    runs.inner_segments(block, core)
    t_start = time.time()
    with block:
        _, seconds = runs.run_block(block, data, cov, turns, seed)
    minutes = (time.time() - t_start) / 60.0
    chains = {s.index: block.cold_chain(s.index) for s in segs}
    owners = block.table.owners(q)
    dropped = {d["name"]: d["snr"] for d in block.dropped}
    inj_q = start["Frequency"] * T
    out = []
    for (name, row), qd, owner in zip(rows_g.iterrows(), q, owners, strict=True):
        g = glass_chain(det, name)
        gf, glnA = g["Frequency"].to_numpy(), np.log(g["Amplitude"].to_numpy())
        chain = chains[segs[owner].index] if owner >= 0 else None
        if chain is not None:
            ours, frac = _counterpart(chain, float(np.median(gf)) * T)
        else:
            ours, frac = np.empty((0, 8)), 0.0
        entry = {
            "name": name,
            "glass_snr": float(row["SNR"]),
            "found_fraction": frac,
            "owner_segment": int(segs[owner].index) if owner >= 0 else None,
            "glass_f": [float(np.median(gf)), float(np.std(gf))],
            "glass_lnA": [float(np.median(glnA)), float(np.std(glnA))],
        }
        if len(ours) > 1:
            of, olnA = ours[:, 0] / T, ours[:, 3]
            entry["block_f"] = [float(np.median(of)), float(np.std(of))]
            entry["block_lnA"] = [float(np.median(olnA)), float(np.std(olnA))]
            entry["sep_f"] = abs(entry["block_f"][0] - entry["glass_f"][0]) / np.hypot(
                entry["block_f"][1], entry["glass_f"][1]
            )
            entry["sep_lnA"] = abs(
                entry["block_lnA"][0] - entry["glass_lnA"][0]
            ) / np.hypot(entry["block_lnA"][1], entry["glass_lnA"][1])
        if len(inj_q):
            j = int(np.argmin(np.abs(inj_q - qd)))
            inj_name = str(start["Name"][j])
            entry["injection"] = {
                "name": inj_name,
                "f": float(start["Frequency"][j]),
                "lnA": float(np.log(start["Amplitude"][j])),
                "dropped": inj_name in dropped,
                "block_snr": dropped.get(inj_name),
            }
        if frac < 0.5:
            entry["nearest_catalogue"] = _nearest_catalogue(path, T, qd, start)
        entry["overlap"] = bool(
            frac >= 0.5
            and entry.get("sep_f", np.inf) < 3
            and entry.get("sep_lnA", np.inf) < 3
        )
        out.append(entry)
    return {
        "segment": seg_index,
        "band": list(band),
        "core": list(core),
        "span": SPAN,
        "edge_segments": [segs[0].index, segs[-1].index],
        "n_start": len(start),
        "n_dropped": len(block.dropped),
        "seconds_per_turn": seconds,
        "minutes": minutes,
        "detections": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--path", type=Path, default=sangria.DEFAULT_PATH)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    n, dt = sangria.grid(args.path)
    T = n * dt
    spectra = sangria.aet_spectra(*sangria.combine(["obs"], ["mbhb", "vgb"], args.path))
    data = runs.frequency_data(spectra, n, dt)
    det, chosen = glass_segments(args.segments)
    outdir = Path(tempfile.mkdtemp(prefix="glass-compare-"))
    results = [
        compare_segment(
            s,
            det,
            spectra,
            data,
            T=T,
            turns=args.turns,
            steps=args.steps,
            seed=args.seed + s,
            outdir=outdir,
            path=args.path,
            workers=args.workers,
        )
        for s in chosen
    ]
    bright = [d for r in results for d in r["detections"] if d["glass_snr"] >= 10]
    fraction = float(np.mean([d["overlap"] for d in bright])) if bright else 0.0
    report = {
        "noise_reference": NOISE_REFERENCE,
        "T_block": T,
        "T_glass": 31_457_280.0,
        "segments": results,
        "n_bright": len(bright),
        "bright_overlap_fraction": fraction,
        "pass": len(bright) >= 1 and fraction >= 0.8,
    }
    runs.write_report("segments-glass-compare.json", report)
    for r in results:
        print(
            f"GLASS segment {r['segment']} band {r['band']} core {r['core']} "
            f"start={r['n_start']} dropped={r['n_dropped']} minutes={r['minutes']:.1f}"
        )
        for d in r["detections"]:
            inj = d.get("injection", {})
            print(
                f"  {d['name']} snr={d['glass_snr']:6.1f} "
                f"found={d['found_fraction']:.2f} "
                f"sep_f={d.get('sep_f')} sep_lnA={d.get('sep_lnA')} "
                f"overlap={d['overlap']} "
                f"inj={inj.get('name')} inj_dropped={inj.get('dropped')} "
                f"inj_block_snr={inj.get('block_snr')}"
            )
            if "nearest_catalogue" in d:
                print(f"    MISS nearest catalogue: {d['nearest_catalogue']}")
    if not bright:
        print("VACUOUS: no GLASS detection has SNR >= 10")
    print(
        f"n_bright {len(bright)}; bright overlap fraction {fraction:.2f}; "
        f"run directory: {outdir}"
    )
    print("PASS" if report["pass"] else "FAIL")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
