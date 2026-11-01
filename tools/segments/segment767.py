"""Crowded band: gbnoisefit's Segment767 (18 sources) at fixed count and with RJ.

Data: Sangria obs minus the true MBHB and VGB streams. Noise: a running median
of the data's own power. It absorbs the unresolved galaxy, and serves as both
the turn's covariance and the SNR-cut reference. Start set: the 18 catalogue
rows matching the npz's ``initial_sources_native`` frequencies (column 0, Hz).

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
MARGIN = 300


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
    band = (int(np.floor(float(z["f_lo"]) * T)), int(np.ceil(float(z["f_hi"]) * T)))
    rows = start_rows(path, T)
    spectra = sangria.aet_spectra(*sangria.combine(["obs"], ["mbhb", "vgb"], path))
    data = runs.frequency_data(spectra, n, dt)
    lo, hi = band[0] - MARGIN, band[1] + MARGIN
    cov = runs.covariance(data, runs.running_median_variance(spectra, lo, hi), lo, hi)
    config = UCBBlockConfig(band_bins=band, n_target=len(rows),
                            birth_death=(mode == "rj"), steps_per_turn=steps)
    block = UCBBlock(grid=data, start_sources=rows, outdir=outdir / mode, config=config,
                     reference_covariance=cov)
    segs = block.table.segments
    if segs[0].bin_lo < lo or segs[-1].bin_hi > hi:
        raise RuntimeError(f"padded band {segs[0].bin_lo}..{segs[-1].bin_hi} exceeds "
                           f"the running-median range {lo}..{hi}")
    _, seconds = runs.run_block(block, data, cov, turns, seed)
    native = ldc_to_native(rows, T)
    report = {"mode": mode, "band": band, "segments": len(block.table.segments),
              "dropped": block.dropped, "seconds_per_turn": seconds, "sources": []}
    for seg in block.table.segments:
        lik = SegmentLikelihood.for_segment(seg, T=T, t0=0.0, config=config)
        lik.set_reference(block.c_ref[:, seg.bin_lo : seg.bin_hi])
        inj = native[(native[:, 0] >= seg.lo) & (native[:, 0] < seg.hi)]
        # ChainArchive refuses a directory that already holds turns, so read the
        # block's own runner archive instead of opening a second one.
        runner = block._runners[seg.index]
        if runner is None:
            continue
        chain = runner.archive.load()
        found, counts = runs.recovery(chain, inj)
        for entry, rho in zip(found, lik.snr(inj) if len(inj) else []):
            entry["snr"] = float(rho)
        report["sources"].extend(found)
        report.setdefault("count_histogram", []).append(counts)
    floor, need = (7.0, 0.9) if mode == "fixed" else (10.0, 0.8)
    report["criterion"] = {"snr_at_least": floor, "recovered_at_least": need}
    report["pass"] = all(s["recovered"] >= need for s in report["sources"] if s["snr"] >= floor)
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
        print(f"q={s['q']:.2f} snr={s['snr']:6.1f} recovered={s['recovered']:.2f} "
              f"dq={s['median_dq']} dlnA={s['median_dlnA']}")
    print("counts:", report["count_histogram"], " dropped:", len(report["dropped"]))
    print(f"run directory: {outdir}")
    print("PASS" if report["pass"] else "FAIL")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
