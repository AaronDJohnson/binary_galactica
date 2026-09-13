"""Compare zero refiners using an explicitly selected local LAT JAX response.

Run ``python -m tools.tdi_zero_crossings.benchmark_lat --lat-checkout PATH``.
The callback uses LAT's unchanged projection with a frozen constellation and
chirping source fixture. It is not a production waveform validation.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from .lat_reference import make_frozen_chirp_callback
from .solver import refine_candidate

METHODS = ("bounded", "brentq", "toms748")
INTERVAL = (-100.0, 100.0)
CONTROLS = {
    "amplitude_atol": 0.0,
    "amplitude_rtol": 1e-8,
    "time_atol": 1e-8,
    "maxiter": 100,
}
LAT_FILES = (
    "src/lisatools/jax/response/base.py",
    "src/lisatools/jax/response/tdi_config.py",
    "src/lisatools/jax/orbits.py",
    "src/lisatools/jax/response/projection.py",
)


def _hash_sources(checkout: Path) -> dict[str, str]:
    """Fingerprint the loaded modules and available TDI combination definition."""
    files = list(LAT_FILES)
    configuration_source = "src/lisatools/response/tdiconfig.py"
    if (checkout / configuration_source).is_file():
        files.append(configuration_source)
    return {
        relative: hashlib.sha256((checkout / relative).read_bytes()).hexdigest()
        for relative in files
    }


def _measure(response, method: str) -> dict:
    """Count all real callback evaluations and time a complete candidate solve."""
    observed_evaluations = 0

    def counted(t):
        nonlocal observed_evaluations
        observed_evaluations += 1
        return response(t)

    wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
    result = refine_candidate(counted, INTERVAL, method=method, **CONTROLS)
    cpu_seconds = (time.process_time_ns() - cpu_start) * 1e-9
    wall_seconds = (time.perf_counter_ns() - wall_start) * 1e-9
    if result.evaluations != observed_evaluations:
        raise RuntimeError("Solver callback count does not match observed evaluations")
    return {
        **asdict(result),
        "value": [result.value.real, result.value.imag],
        "method_requested": method,
        "expected_kind": "near_miss",
        "classification_correct": result.kind == "near_miss",
        "relative_residual": result.residual / result.scale if result.scale else None,
        "observed_evaluations": observed_evaluations,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
    }


def benchmark(lat_checkout: str | Path, *, repeats: int = 5) -> dict:
    """Measure the controlled reference minimum with compilation excluded.

    Enable JAX float64 before calling this function; the CLI does so explicitly.
    A warm scalar callback is shared by every method. Solves include their
    endpoint checks and complex residual checks, but candidate discovery and
    source loading are excluded from timings.
    """
    import jax
    import jaxlib

    if repeats < 1:
        raise ValueError("repeats must be positive")
    checkout = Path(lat_checkout).expanduser().resolve()
    hashes = _hash_sources(checkout)
    response = make_frozen_chirp_callback(checkout)
    response(0.0)  # Python complex conversion waits for compiled CPU evaluation.
    records = {method: [] for method in METHODS}
    for repeat in range(repeats):
        order = METHODS[repeat % len(METHODS) :] + METHODS[: repeat % len(METHODS)]
        for method in order:
            row = _measure(response, method)
            row["repeat"] = repeat
            records[method].append(row)
    if hashes != _hash_sources(checkout):
        raise RuntimeError("LAT reference source files changed during the benchmark")
    return {
        "schema_version": 1,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "jax_enable_x64": jax.config.x64_enabled,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_devices": [str(device) for device in jax.devices("cpu")],
            "wall_clock": "time.perf_counter_ns",
            "cpu_clock": "time.process_time_ns",
        },
        "reference": {
            "checkout": str(checkout),
            "loaded_modules": LAT_FILES,
            "sha256": hashes,
        },
        "configuration": {
            "interval_seconds": INTERVAL,
            "repeats": repeats,
            **CONTROLS,
            "fixture": {
                key: parameter.default
                for key, parameter in inspect.signature(
                    make_frozen_chirp_callback
                ).parameters.items()
                if key != "lat_checkout"
            },
            "expected_kind": "near_miss",
            "timed_callback_warmups": 0,
            "untimed_callback_warmups": 1,
        },
        "limitations": [
            "The unchanged LAT JAX projection is evaluated with a frozen "
            "constellation and a constant-amplitude chirping source fixture.",
            "This verifies a nonzero response minimum, not a true crossing "
            "or a realistic orbital waveform.",
            "This does not establish compiled C++ parity or production speedup.",
            "Loading, JIT compilation, candidate discovery, and reporting are "
            "excluded; each solve includes its full response evaluation costs.",
            "Projected scalar roots require the full complex residual check; "
            "a failed residual check triggers bounded magnitude refinement, "
            "whose complete additional cost is included.",
            "The reported requested method can differ from the final method. "
            "Bounded refinement includes a second, recentered minimization.",
            "Callback cancellation and the flat nonzero minimum limit time "
            "accuracy; the time tolerance does not certify the minimum's location.",
        ],
        "summary": {
            method: {
                "runs": len(rows),
                "near_misses": sum(row["kind"] == "near_miss" for row in rows),
                "methods_used": dict(Counter(row["method"] for row in rows)),
                "fallbacks": sum(
                    row["method"] != row["method_requested"] for row in rows
                ),
                "evaluations_median": statistics.median(
                    row["evaluations"] for row in rows
                ),
                "wall_seconds_median": statistics.median(
                    row["wall_seconds"] for row in rows
                ),
                "cpu_seconds_median": statistics.median(
                    row["cpu_seconds"] for row in rows
                ),
            }
            for method, rows in records.items()
        },
        "results": records,
    }


def main() -> None:
    """Write every solve and its source fingerprints to a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lat-checkout",
        required=True,
        type=Path,
        help="Trusted local LISAanalysistools checkout whose Python source is executed",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/tdi_zero_crossings_lat.json")
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    import jax

    jax.config.update("jax_enable_x64", True)
    report = benchmark(args.lat_checkout, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Frozen-constellation LAT JAX fixture; expected nonzero minimum.")
    print(
        "requested used       near misses     calls    "
        "wall median (s)     CPU median (s)"
    )
    for method, row in report["summary"].items():
        print(
            f"{method:9} {','.join(row['methods_used']):9} "
            f"{row['near_misses']:3}/{row['runs']:<3} "
            f"{row['evaluations_median']:9g} "
            f"{row['wall_seconds_median']:18.6g} {row['cpu_seconds_median']:18.6g}"
        )
    print("These fixture timings do not establish production speedup.")
    print(f"JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
