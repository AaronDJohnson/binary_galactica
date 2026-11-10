"""Time the parallel even/odd sweep on a Sangria sub-band for several worker counts.

Each configuration runs in a fresh Python process (``--one``), so its peak
memory and ``XLA_FLAGS`` are its own. A configuration builds the block on the
band with the residual noise reference (as ``segment767.py`` does), runs
``--turns`` turns of ``--steps`` steps, and records:

- ``turn_s``: every turn's wall time; the last is the steady value, because
  the first includes the compiles;
- ``steps_per_s``: active segments x steps / the last turn's time;
- ``worker_peak_rss_bytes``: each worker's peak resident memory (the main
  process's with one worker) and ``main_peak_rss_bytes``.

``single_thread`` configurations set ``XLA_FLAGS`` to ``SINGLE_THREAD`` for
the configuration's processes; the others inherit the caller's environment.

A configuration whose predicted memory (the main process plus ``workers``
times the largest worker of the latest multi-worker run at the same
threading) exceeds ``--max-total-gib`` is skipped and recorded as skipped.

    PYTHONPATH=src:... python tools/segments/parallel_timing.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import runs
import sangria

from binary_galactica.segments import UCBBlock, UCBBlockConfig
from binary_galactica.segments.executor import peak_rss_bytes

SINGLE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
MARGIN = runs.MARGIN
HALF_WIDTH = runs.HALF_WIDTH


def measure(
    *,
    workers: int,
    band_mhz: tuple[float, float],
    steps: int,
    turns: int,
    seed: int,
    path: Path,
    outdir: Path,
) -> dict:
    """Run one configuration in this process and return its record."""
    n, dt = sangria.grid(path)
    T = n * dt
    band = (int(band_mhz[0] * 1e-3 * T), int(band_mhz[1] * 1e-3 * T))
    rows = runs.catalogue_rows(path, T=T, band=band, snr_min=runs.START_SNR)
    spectra = sangria.aet_spectra(*sangria.combine(["obs"], ["mbhb", "vgb"], path))
    data = runs.frequency_data(spectra, n, dt)
    lo, hi = band[0] - MARGIN, band[1] + MARGIN
    resid, _ = runs.residual_spectra(
        spectra, path, T=T, lo=lo - HALF_WIDTH, hi=hi + HALF_WIDTH
    )
    cov = runs.covariance(
        data, runs.running_median_variance(resid, lo, hi, HALF_WIDTH), lo, hi
    )
    config = UCBBlockConfig(band_bins=band, steps_per_turn=steps)
    with UCBBlock(
        grid=data,
        start_sources=rows,
        outdir=outdir,
        config=config,
        reference_covariance=cov,
        workers=workers,
    ) as block:
        segs = block.table.segments
        if segs[0].bin_lo < lo or segs[-1].bin_hi > hi:
            raise RuntimeError(
                f"padded band {segs[0].bin_lo}..{segs[-1].bin_hi} exceeds "
                f"the running-median range {lo}..{hi}"
            )
        _, seconds = runs.run_block(block, data, cov, turns, seed)
        worker_rss = block.peak_rss_bytes()
        active = sum(
            1 for s in segs if config.birth_death or block.owned_start[s.index]
        )
    return {
        "workers": workers,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "band_mhz": list(band_mhz),
        "band_bins": list(band),
        "start_sources": int(len(rows)),
        "segments": len(segs),
        "active_segments": active,
        "steps_per_turn": steps,
        "turn_s": seconds,
        "steady_turn_s": seconds[-1],
        "steps_per_s": active * steps / seconds[-1],
        "worker_peak_rss_bytes": worker_rss,
        "main_peak_rss_bytes": peak_rss_bytes(),
    }


def _one(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="parallel-timing-") as tmp:
        record = measure(
            workers=args.workers[0],
            band_mhz=tuple(args.band_mhz),
            steps=args.steps,
            turns=args.turns,
            seed=args.seed,
            path=args.path,
            outdir=Path(tmp) / "run",
        )
    args.record.write_text(json.dumps(record, indent=1) + "\n")


def _configuration(args: argparse.Namespace, workers: int, single: bool) -> dict:
    env = dict(os.environ)
    if single:
        env["XLA_FLAGS"] = SINGLE_THREAD
    with tempfile.TemporaryDirectory(prefix="parallel-timing-record-") as tmp:
        record = Path(tmp) / "record.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--one",
            "--record",
            str(record),
            "--workers",
            str(workers),
            "--band-mhz",
            *(str(x) for x in args.band_mhz),
            "--steps",
            str(args.steps),
            "--turns",
            str(args.turns),
            "--seed",
            str(args.seed),
            "--path",
            str(args.path),
        ]
        done = subprocess.run(command, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            return {
                "workers": workers,
                "single_thread": single,
                "error": done.stderr[-4000:],
            }
        result = json.loads(record.read_text())
    result["single_thread"] = single
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--band-mhz", type=float, nargs=2, default=[2.0, 4.0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--path", type=Path, default=sangria.DEFAULT_PATH)
    parser.add_argument(
        "--threading",
        choices=("both", "inherit", "single"),
        default="both",
        help="XLA threading variants to time",
    )
    parser.add_argument(
        "--max-total-gib",
        type=float,
        default=52.0,
        help="skip a configuration whose predicted memory exceeds this",
    )
    parser.add_argument("--out", default="segments-parallel.json")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        _one(args)
        return
    variants = {"both": (False, True), "inherit": (False,), "single": (True,)}
    results = []
    budget = args.max_total_gib * 2**30
    for single in variants[args.threading]:
        last = None  # the latest successful multi-worker record at this threading
        for workers in args.workers:
            if workers > 1 and last is not None:
                predicted = last["main_peak_rss_bytes"] + workers * max(
                    last["worker_peak_rss_bytes"]
                )
                if predicted > budget:
                    note = (
                        f"predicted {predicted / 2**30:.1f} GiB exceeds "
                        f"--max-total-gib {args.max_total_gib:g}"
                    )
                    results.append(
                        {"workers": workers, "single_thread": single, "skipped": note}
                    )
                    print(f"workers={workers} single_thread={single}: SKIPPED, {note}")
                    continue
            result = _configuration(args, workers, single)
            results.append(result)
            if "error" in result:
                print(f"workers={workers} single_thread={single}: FAILED")
                print(result["error"])
                continue
            if workers > 1:
                last = result
            print(
                f"workers={workers:2d} single_thread={single!s:5s} "
                f"segments={result['active_segments']} "
                f"steady turn {result['steady_turn_s']:7.1f} s "
                f"{result['steps_per_s']:8.0f} steps/s "
                f"worker rss max {max(result['worker_peak_rss_bytes']) / 2**30:.2f} GiB"
            )
    serial = {
        r["single_thread"]: r["steps_per_s"]
        for r in results
        if r.get("workers") == 1 and "error" not in r and "skipped" not in r
    }
    for r in results:
        if "error" not in r and "skipped" not in r and r["single_thread"] in serial:
            r["speedup_vs_one_worker"] = r["steps_per_s"] / serial[r["single_thread"]]
    report = {
        "machine_cpu_count": os.cpu_count(),
        "single_thread_flags": SINGLE_THREAD,
        "steps_per_turn": args.steps,
        "turns": args.turns,
        "note": (
            "steady_turn_s is the last turn's wall time; the first turn includes "
            "the compiles. speedup_vs_one_worker compares steps_per_s with the "
            "one-worker run of the same threading."
        ),
        "results": results,
    }
    print(f"wrote {runs.write_report(args.out, report)}")


if __name__ == "__main__":
    main()
