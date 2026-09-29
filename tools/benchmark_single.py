"""Time float32 and float64 on-fly waveforms on identical sources.

Run from the repository root, serially, on an otherwise idle machine::

    PYTHONPATH=src python tools/benchmark_single.py --output /tmp/single-timing.json

Each precision runs in its own process, as an application would: the float64
``binary_galactica.ucb_waveform`` with ``jax_enable_x64`` enabled, and the
float32 ``binary_galactica.single.ucb_waveform`` with it disabled. The two
processes alternate for ``--rounds`` rounds, swapping which goes first. Both
evaluate the same seeded catalogue near 3 mHz: ``split_native`` gives the
float32 reference bins and offsets, and float64 receives the exact float64
values of those inputs.

At each duration and bandwidth three workloads are timed after warmup, one
call at a time, with every output leaf synchronized before the clock stops:

``public``
    The ordinary public call on host NumPy inputs, including validation.
``jit``
    One source through an outer ``jax.jit`` compiled ahead of time with
    ``lower(...).compile()``. Parameters, ``q_ref``, ``T`` and ``t0`` are
    device arrays passed as dynamic arguments.
``batch``
    ``--batch-size`` sources through ``jax.jit(jax.vmap(...))`` compiled
    ahead of time, vectorized over the parameters and ``q_ref``.

All calls use the automatic response grid, the analytic orbit and the Sangria
format, and return every output (X, Y, Z, A, E, qmin, freq_bin_indices).
Before timing, the float32 A/E of the first source is compared with float64
in float64 arithmetic. Each worker records the system load average at its
start and end, since concurrent work inflates latencies. These are CPU
latencies on one machine; they do not measure GPU or TPU throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np

import binary_galactica as bg
from binary_galactica import single as bg32
from binary_galactica.glass.constants import YEAR
from binary_galactica.glass.waveform import _AUTO_RESPONSE_SAMPLES

PRECISIONS = ("float64", "float32")
#: Dtypes of the bin metadata and of the channels for each precision.
DTYPES = {
    "float64": (np.dtype(np.float64), np.dtype(np.complex128)),
    "float32": (np.dtype(np.int32), np.dtype(np.complex64)),
}
WORKLOADS = ("public", "jit", "batch")
SEED = 20260929
#: About a quarter year and about four years, exact in float32 and clear of
#: the automatic grid's thresholds (32 and 512 response samples).
DURATIONS = (7_884_000.0, 126_000_000.0)
EPOCH = 12_345.0
FREQUENCY = 3e-3
FORMAT = "sangria"
#: Disjoint catalogue slices that the batch workload cycles through.
BATCH_COUNT = 2
BUDGET = {"relative_l2": 1e-4, "mismatch": 1e-8}
SETTINGS = (
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "JAX_DISABLE_JIT",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "XLA_FLAGS",
    "OMP_NUM_THREADS",
)


def automatic_samples(duration: float) -> int:
    """Return the automatic sparse response count used for ``duration``."""
    boundaries = YEAR * np.asarray(_AUTO_RESPONSE_SAMPLES[:-1]) / 128
    return _AUTO_RESPONSE_SAMPLES[int(np.sum(boundaries < duration))]


def catalogue(duration: float, size: int) -> dict[str, np.ndarray]:
    """Return seeded native sources, split for float32, rejoined for float64."""
    rng = np.random.default_rng(SEED)
    native = np.empty((size, 8), dtype=np.float64)
    # Quarter-bin offsets keep both default carriers on the same bin.
    native[:, 0] = round(FREQUENCY * duration) + rng.integers(-16, 17, size) + 0.25
    native[:, 1] = rng.uniform(-0.9, 0.9, size)
    native[:, 2] = rng.uniform(0.0, 2.0 * np.pi, size)
    native[:, 3] = rng.uniform(np.log(1e-23), np.log(1e-20), size)
    native[:, 4] = rng.uniform(-0.9, 0.9, size)
    native[:, 5] = rng.uniform(0.0, np.pi, size)
    native[:, 6] = rng.uniform(0.0, 2.0 * np.pi, size)
    native[:, 7] = rng.uniform(-0.05, 0.2, size)
    q_ref, params = bg32.split_native(native, duration)
    exact = params.astype(np.float64)
    exact[:, 0] += q_ref
    return {"q_ref": np.asarray(q_ref), "float32": params, "float64": exact}


def synchronize(result: Any) -> None:
    """Wait for every output leaf."""
    for leaf in jax.tree_util.tree_leaves(result):
        leaf.block_until_ready()


def check(result: Any, precision: str) -> None:
    """Require the precision's dtypes and finite values, outside the timing."""
    bins, channels = DTYPES[precision]
    for name, value in result.items():
        value = np.asarray(value)
        expected = bins if name in ("qmin", "freq_bin_indices") else channels
        if value.dtype != expected or not np.all(np.isfinite(value)):
            raise RuntimeError(f"{precision} {name}: {value.dtype}, or nonfinite")


def timings(
    function: Callable[..., Any],
    arguments: list[tuple[Any, ...]],
    *,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    """Time sequential synchronized calls, cycling through the inputs."""

    def call(index: int) -> int:
        start = time.perf_counter_ns()
        synchronize(function(*arguments[index % len(arguments)]))
        return time.perf_counter_ns() - start

    first = call(0)
    for index in range(warmups):
        call(index + 1)
    samples = np.asarray([call(index) for index in range(repeats)]) / 1e6
    return {
        "first_call_ms": first / 1e6,
        "median_ms": float(np.median(samples)),
        "p05_ms": float(np.percentile(samples, 5)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(np.min(samples)),
    }


def workloads(
    precision: str,
    sources: dict[str, np.ndarray],
    duration: float,
    bandwidth: int,
    batch_size: int,
) -> dict[str, tuple[Callable[..., Any], list[tuple[Any, ...]], float | None]]:
    """Build each workload's function, inputs and ahead-of-time compile time."""
    real = np.dtype(precision)
    params, q_ref = sources[precision], sources["q_ref"]

    def one(p, q, duration, epoch):
        if precision == "float32":
            return bg32.ucb_waveform(
                p, q_ref=q, T=duration, BW=bandwidth, t0=epoch, fmt=FORMAT
            )
        return bg.ucb_waveform(p, T=duration, BW=bandwidth, t0=epoch, fmt=FORMAT)

    host = [(p, int(q), duration, EPOCH) for p, q in zip(params, q_ref, strict=True)]
    clock = (jax.device_put(real.type(duration)), jax.device_put(real.type(EPOCH)))
    device = [
        (jax.device_put(p), jax.device_put(q), *clock)
        for p, q in zip(params, q_ref, strict=True)
    ]
    batches = [
        (jax.device_put(params[part]), jax.device_put(q_ref[part]), *clock)
        for part in np.split(np.arange(BATCH_COUNT * batch_size), BATCH_COUNT)
    ]
    synchronize((device, batches))
    built: dict[str, tuple[Callable[..., Any], list[tuple[Any, ...]], float | None]]
    built = {"public": (one, host, None)}
    for name, function, inputs in (
        ("jit", jax.jit(one), device),
        ("batch", jax.jit(jax.vmap(one, in_axes=(0, 0, None, None))), batches),
    ):
        start = time.perf_counter()
        compiled = function.lower(*inputs[0]).compile()
        built[name] = (compiled, inputs, time.perf_counter() - start)
    return built


def worker(precision: str, arguments: argparse.Namespace) -> None:
    """Time one precision in this process and write its record and outputs."""
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    elif jax.config.read("jax_enable_x64"):
        raise RuntimeError("the float32 worker must run with jax_enable_x64 off")
    load = os.getloadavg()
    records: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    fixture = hashlib.sha256()
    for duration in DURATIONS:
        sources = catalogue(duration, BATCH_COUNT * arguments.batch_size)
        for name in ("q_ref", "float32", "float64"):
            fixture.update(sources[name].tobytes())
        for bandwidth in arguments.bandwidths:
            built = workloads(
                precision, sources, duration, bandwidth, arguments.batch_size
            )
            function, inputs, _ = built["jit"]
            first = function(*inputs[0])
            check(first, precision)
            key = f"{duration:.0f}_{bandwidth}"
            outputs[f"AE_{key}"] = np.stack([first["A"], first["E"]])
            outputs[f"qmin_{key}"] = np.asarray(first["qmin"])
            for name, (function, inputs, compile_seconds) in built.items():
                if name == "batch":
                    check(function(*inputs[0]), precision)
                record = timings(
                    function,
                    inputs,
                    repeats=arguments.repeats,
                    warmups=arguments.warmups,
                )
                records.append(
                    {
                        "T": duration,
                        "BW": bandwidth,
                        "workload": name,
                        "compile_seconds": compile_seconds,
                        **record,
                    }
                )
                print(
                    f"  {precision} T={duration:.4g} BW={bandwidth} {name:6s} "
                    f"median {record['median_ms']:.4f} ms",
                    file=sys.stderr,
                    flush=True,
                )
    runtime = {
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_default_matmul_precision": getattr(
            jax.config, "jax_default_matmul_precision", None
        ),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "environment_variables": {name: os.environ.get(name) for name in SETTINGS},
        "load_average_1_5_15_min": {"start": load, "end": os.getloadavg()},
    }
    np.savez(arguments.worker_output.with_suffix(".npz"), **outputs)
    arguments.worker_output.write_text(
        json.dumps(
            {
                "precision": precision,
                "runtime": runtime,
                "fixture_sha256": fixture.hexdigest(),
                "records": records,
            }
        )
    )


def run(
    precision: str, arguments: argparse.Namespace, scratch: Path, round_: int
) -> dict[str, Any]:
    """Run one worker process with a clean x64 environment and load its record."""
    output = scratch / f"{precision}-{round_}.json"
    environment = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        precision,
        "--worker-output",
        str(output),
        "--repeats",
        str(arguments.repeats),
        "--warmups",
        str(arguments.warmups),
        "--batch-size",
        str(arguments.batch_size),
        "--bandwidths",
        *map(str, arguments.bandwidths),
    ]
    subprocess.run(command, env=environment, check=True)
    record = json.loads(output.read_text())
    with np.load(output.with_suffix(".npz")) as arrays:
        record["outputs"] = dict(arrays)
    return record


def agreement(
    single: dict[str, np.ndarray], double: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Compare the first source's A/E across precisions, in float64."""
    rows = []
    cases = sorted(tuple(map(int, k[3:].split("_"))) for k in double if k[:3] == "AE_")
    for duration, bandwidth in cases:
        key = f"{duration}_{bandwidth}"
        a = single[f"AE_{key}"].astype(np.complex128).ravel()
        b = double[f"AE_{key}"].astype(np.complex128).ravel()
        overlap = np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
        row = {
            "T": float(duration),
            "BW": bandwidth,
            "same_qmin": int(single[f"qmin_{key}"]) == int(double[f"qmin_{key}"]),
            "relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
            "mismatch": float(1 - overlap),
        }
        if not row["same_qmin"] or any(row[m] > BUDGET[m] for m in BUDGET):
            raise RuntimeError(f"float32 and float64 disagree: {row}")
        rows.append(row)
    return rows


def summarize(
    rounds: list[dict[str, Any]], arguments: argparse.Namespace
) -> list[dict[str, Any]]:
    """Return the median over rounds of each round's median, and the ratio."""
    rows = []
    for duration in DURATIONS:
        for bandwidth in arguments.bandwidths:
            for name in WORKLOADS:
                row: dict[str, Any] = {
                    "T": duration,
                    "response_samples": automatic_samples(duration),
                    "BW": bandwidth,
                    "workload": name,
                    "sources_per_call": arguments.batch_size if name == "batch" else 1,
                }
                for precision in PRECISIONS:
                    medians = [
                        record["median_ms"]
                        for results in rounds
                        for record in results[precision]["records"]
                        if (record["T"], record["BW"], record["workload"])
                        == (duration, bandwidth, name)
                    ]
                    row[f"{precision}_median_ms"] = float(np.median(medians))
                    row[f"{precision}_round_medians_ms"] = medians
                row["float32_over_float64"] = (
                    row["float32_median_ms"] / row["float64_median_ms"]
                )
                rows.append(row)
    return rows


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark settings; the worker options are internal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON report path")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bandwidths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--worker", choices=PRECISIONS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if min(arguments.repeats, arguments.warmups, arguments.rounds) < 1:
        parser.error("repeats, warmups and rounds must be positive")
    if arguments.batch_size < 1:
        parser.error("batch-size must be positive")
    if any(bandwidth < 2 or bandwidth % 2 for bandwidth in arguments.bandwidths):
        parser.error("bandwidths must be positive even integers")
    if arguments.worker is None and arguments.output is None:
        parser.error("--output is required")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    """Alternate the two precision processes and write the combined report."""
    arguments = parse_arguments(argv)
    if arguments.worker is not None:
        worker(arguments.worker, arguments)
        return
    # Importing the shared harness enables x64, but only in this parent
    # process, which times nothing.
    from benchmark_onfly import provenance

    before = provenance()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rounds = []
    with tempfile.TemporaryDirectory() as scratch:
        for round_ in range(arguments.rounds):
            order = PRECISIONS if round_ % 2 == 0 else PRECISIONS[::-1]
            print(f"round {round_ + 1}: {', '.join(order)}", file=sys.stderr)
            rounds.append(
                {
                    precision: run(precision, arguments, Path(scratch), round_)
                    for precision in order
                }
            )
    after = provenance()
    if after["source_tree_sha256"] != before["source_tree_sha256"]:
        raise RuntimeError("the package source changed during the benchmark")
    fixtures = {results[p]["fixture_sha256"] for results in rounds for p in PRECISIONS}
    if len(fixtures) != 1:
        raise RuntimeError("the worker processes built different catalogues")
    last = rounds[-1]
    report = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": before,
        "purpose": (
            "Warmed CPU latency of the float32 binary_galactica.single.ucb_waveform "
            "(x64 disabled) against the float64 on-fly ucb_waveform (x64 enabled) "
            "on identical sources."
        ),
        "method": {
            "processes": (
                "one process per precision and round; the order alternates each "
                "round; JAX_ENABLE_X64 is removed from each worker's environment"
            ),
            "clock": (
                "time.perf_counter_ns around one call, stopped after every output "
                "leaf blocks; the first call and the warmups are excluded"
            ),
            "public": "ordinary public call on host NumPy inputs, with validation",
            "jit": (
                "outer jax.jit, compiled ahead of time with lower().compile(); "
                "device inputs, T and t0 dynamic, q_ref a dynamic int32 scalar"
            ),
            "batch": (
                "jax.jit(jax.vmap(...)) over params and q_ref, compiled ahead of "
                "time; T and t0 shared"
            ),
            "inputs": (
                "split_native of a seeded catalogue near 3 mHz; float64 receives "
                "the exact float64 values of the float32 inputs"
            ),
            "summary": "median over rounds of each round's median latency",
            "load": "os.getloadavg() at each worker's start and end, per round",
            "settings": {
                "repeats": arguments.repeats,
                "warmups": arguments.warmups,
                "rounds": arguments.rounds,
                "batch_size": arguments.batch_size,
                "bandwidths": arguments.bandwidths,
                "durations_seconds": DURATIONS,
                "t0_seconds": EPOCH,
                "frequency_hz": FREQUENCY,
                "format": FORMAT,
                "seed": SEED,
                "response_samples": "automatic",
            },
        },
        "fixture_sha256": fixtures.pop(),
        "agreement": agreement(last["float32"]["outputs"], last["float64"]["outputs"]),
        "summary": summarize(rounds, arguments),
        "rounds": [
            {
                p: {"runtime": results[p]["runtime"], "records": results[p]["records"]}
                for p in PRECISIONS
            }
            for results in rounds
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in report["summary"]:
        print(
            f"T={row['T']:.4g} BW={row['BW']:5d} {row['workload']:6s} "
            f"float64 {row['float64_median_ms']:8.4f} ms  "
            f"float32 {row['float32_median_ms']:8.4f} ms  "
            f"ratio {row['float32_over_float64']:.2f}"
        )


if __name__ == "__main__":
    main()
