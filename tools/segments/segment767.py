"""Crowded band: gbnoisefit's Segment767 core with a band built around it.

Data: Sangria obs minus the true MBHB and VGB streams. Noise: the residual
noise reference (``runs.NOISE_REFERENCE``): a running median of the power of
obs - mbhb - vgb minus the catalogue sources with analytic SNR >= 7. It absorbs
the unresolved galaxy, and serves as both the turn's covariance and the
SNR-cut reference. The core is the npz band
(``f_lo``..``f_hi``); the block's band extends SPAN bins either side, and the
start set is every dgb/igb catalogue row in it with analytic SNR >= 7 plus the
18 rows matching the npz's ``initial_sources_native`` frequencies (column 0, Hz).
Loud neighbours therefore get segments of their own instead of sitting
unmodelled in the core's padding.

Recovery is judged over start sources that survive the SNR-5 cut, in the
segments other than the first and last: the edge segments' outer padding holds
catalogue sources outside the band that no segment models. The 18 npz sources
are reported as a labelled subset, not as the pass criterion.

    PYTHONPATH=src:... python tools/segments/segment767.py --mode fixed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import runs
import sangria

from binary_galactica.segments import (
    SegmentLikelihood,
    UCBBlock,
    UCBBlockConfig,
    ldc_to_native,
)

NPZ = Path(os.environ.get(
    "BG_SEGMENT767",
    str(Path.home() / "Documents/lisa/gbnoisefit/vendor/sangria_segment_767/segment_767.npz"),
))
MARGIN = runs.MARGIN
SPAN = 600
HALF_WIDTH = runs.HALF_WIDTH
NOISE_REFERENCE = runs.NOISE_REFERENCE


def start_rows(path: Path, T: float) -> np.ndarray:
    """Return the catalogue rows of the npz's 18 sources."""
    freqs = np.asarray(np.load(NPZ)["initial_sources_native"][:, 0])
    band = (int(np.floor(freqs.min() * T)) - 2, int(np.floor(freqs.max() * T)) + 3)
    rows = runs.catalogue_rows(path, T=T, band=band)
    picked = []
    for f in freqs:
        j = int(np.argmin(np.abs(rows["Frequency"] - f)))
        if abs(rows["Frequency"][j] - f) > 1e-12 * f:
            raise ValueError(f"no catalogue row at {f!r} Hz")
        picked.append(j)
    if len(set(picked)) != len(freqs):
        raise ValueError("two npz sources matched the same catalogue row")
    return rows[sorted(picked)]


def run(mode: str, *, turns: int, steps: int, seed: int, outdir: Path,
        path: Path = sangria.DEFAULT_PATH) -> dict:
    """Run one mode and return its report."""
    n, dt = sangria.grid(path)
    T = n * dt
    z = np.load(NPZ)
    core = (int(np.floor(float(z["f_lo"]) * T)), int(np.ceil(float(z["f_hi"]) * T)))
    npz_rows = start_rows(path, T)
    band, rows = runs.target_band(path, T=T, core=core, span=SPAN, extra=npz_rows)
    spectra = sangria.aet_spectra(*sangria.combine(["obs"], ["mbhb", "vgb"], path))
    data = runs.frequency_data(spectra, n, dt)
    lo, hi = band[0] - MARGIN, band[1] + MARGIN
    resid, _ = runs.residual_spectra(spectra, path, T=T, lo=lo - HALF_WIDTH, hi=hi + HALF_WIDTH)
    cov = runs.covariance(data, runs.running_median_variance(resid, lo, hi, HALF_WIDTH), lo, hi)
    config = UCBBlockConfig(band_bins=band, birth_death=(mode == "rj"), steps_per_turn=steps)
    block = UCBBlock(grid=data, start_sources=rows, outdir=outdir / mode, config=config,
                     reference_covariance=cov)
    segs = block.table.segments
    if segs[0].bin_lo < lo or segs[-1].bin_hi > hi:
        raise RuntimeError(f"padded band {segs[0].bin_lo}..{segs[-1].bin_hi} exceeds "
                           f"the running-median range {lo}..{hi}")
    inner = runs.inner_segments(block, core)
    _, seconds = runs.run_block(block, data, cov, turns, seed)
    dropped_snr = {d["name"]: d["snr"] for d in block.dropped}
    npz_names = set(npz_rows["Name"])
    floor, need = (7.0, 0.9) if mode == "fixed" else (10.0, 0.8)
    report = {"noise_reference": NOISE_REFERENCE, "mode": mode, "band": band, "core": core, "span": SPAN,
              "segments": len(segs), "edge_segments": [segs[0].index, segs[-1].index],
              "dropped": block.dropped, "seconds_per_turn": seconds,
              "n_start": len(rows), "n_dropped": len(block.dropped), "sources": [],
              "count_histogram": {}}
    for seg in inner:
        lik = SegmentLikelihood.for_segment(seg, T=T, t0=0.0, config=config)
        lik.set_reference(block.c_ref[:, seg.bin_lo : seg.bin_hi])
        own = rows[block.owned_start[seg.index]]
        chain = block.cold_chain(seg.index)
        if chain is None or not len(own):
            continue
        inj = ldc_to_native(own, T)
        found, counts = runs.recovery(chain, inj)
        for entry, row, rho in zip(found, own, lik.snr(inj)):
            entry.update(snr=float(rho), npz=bool(row["Name"] in npz_names),
                         segment=seg.index, name=str(row["Name"]))
        report["sources"].extend(found)
        report["count_histogram"][str(seg.index)] = counts
    report["n_evaluated"] = len(report["sources"])
    report["n_at_floor"] = sum(s["snr"] >= floor for s in report["sources"])
    by_name = {s["name"]: s for s in report["sources"]}
    report["npz_subset"] = [
        {"q": float(q), "name": str(name), "dropped": name in dropped_snr,
         "snr": dropped_snr[name] if name in dropped_snr else by_name.get(name, {}).get("snr"),
         "recovered": by_name.get(name, {}).get("recovered")}
        for q, name in zip(npz_rows["Frequency"] * T, npz_rows["Name"])
    ]
    report["criterion"] = {"snr_at_least": floor, "recovered_at_least": need}
    report["pass"] = report["n_at_floor"] >= 1 and all(
        s["recovered"] >= need for s in report["sources"] if s["snr"] >= floor)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("fixed", "rj"), required=True)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=767)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or Path(tempfile.mkdtemp(prefix="segment767-"))
    report = run(args.mode, turns=args.turns, steps=args.steps, seed=args.seed, outdir=outdir)
    out = runs.PERFORMANCE / "segments-767.json"
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged[args.mode] = report
    runs.write_report("segments-767.json", merged)
    for s in report["sources"]:
        print(f"seg={s['segment']:2d} q={s['q']:.2f} snr={s['snr']:6.1f} "
              f"recovered={s['recovered']:.2f} dq={s['median_dq']} dlnA={s['median_dlnA']}"
              f"{' npz' if s['npz'] else ''}")
    print("counts:", report["count_histogram"], " dropped:", report["n_dropped"],
          " evaluated:", report["n_evaluated"])
    print(f"n_at_floor (snr >= {report['criterion']['snr_at_least']}): {report['n_at_floor']}")
    if report["n_at_floor"] == 0:
        print("VACUOUS: no evaluated source reaches the SNR floor")
    print(f"run directory: {outdir}")
    print("PASS" if report["pass"] else "FAIL")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
