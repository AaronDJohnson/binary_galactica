"""Compare native GLASS and JAX costs on the same UCB inputs.

Build tools/native_glass_timing first, then run this script serially without
competing benchmarks. Native signed-polar response, JAX complex response,
native WDM waveforms, and JAX Fourier waveforms have distinct output contracts;
only the legacy Fourier comparison is an equivalent-output comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_onfly import automatic_response_samples, provenance

import binary_galactica as bg
from binary_galactica.glass.onfly import tdi_response

CHANNELS = ("X", "Y", "Z", "A", "E")


def native(executable, *arguments):
    """Run one isolated native block or dump, parsing its JSON records."""
    result = subprocess.run(
        [str(executable), *map(str, arguments)],
        text=True,
        capture_output=True,
        check=True,
    )
    return [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]


def complex_pairs(values):
    """Decode the native JSON representation without changing normalization."""
    values = np.asarray(values, dtype=np.float64)
    return values[..., 0] + 1j * values[..., 1]


def relative_error(actual, reference):
    """Return a scale-independent complex L2 discrepancy."""
    return float(
        np.linalg.norm(np.asarray(actual) - reference) / np.linalg.norm(reference)
    )


def summary(blocks):
    """Keep native and Python summaries on the same block-mean basis."""
    return {
        "block_means_ms": blocks,
        "median_block_mean_ms": statistics.median(blocks),
        "min_block_mean_ms": min(blocks),
        "max_block_mean_ms": max(blocks),
    }


def compare_case(executable, frequency, years, bandwidth, repeats, blocks):
    """Validate matching inputs, then time each complete call synchronously."""
    response_dump = next(
        row
        for row in native(executable, "--dump-response", frequency, years, bandwidth)
        if row.get("kind") == "response"
    )
    fourier_dump = next(
        row
        for row in native(executable, "--dump-fourier", frequency, years, bandwidth)
        if row.get("kind") == "fourier"
    )
    host_params = np.asarray(response_dump["params"])
    duration, epoch = response_dump["T"], response_dump["t0"]
    params = jnp.asarray(host_params)
    t, e = jnp.asarray(duration), jnp.asarray(epoch)
    times = jnp.asarray(response_dump["times"])
    positions = jnp.asarray(response_dump["positions"])
    count = times.shape[0]
    np.testing.assert_array_equal(host_params, fourier_dump["params"])

    def analytic_response(p, times, t):
        return tdi_response(p, times, t, reference_phase=True)

    def cached_response(p, times, t, positions):
        return tdi_response(p, times, t, positions, reference_phase=True)

    functions = {
        "response_analytic": (jax.jit(analytic_response), (params, times, t)),
        "response_supplied_geometry": (
            jax.jit(cached_response),
            (params, times, t, positions),
        ),
    }
    for name, backend, samples in (
        ("fourier", "fourier", None),
        ("onfly_default", "onfly", None),
        ("onfly_equal_samples", "onfly", count),
    ):

        def complete(p, t, e, backend=backend, samples=samples):
            result = bg.ucb_waveform(
                p,
                T=t,
                t0=e,
                BW=bandwidth,
                backend=backend,
                response_samples=samples,
            )
            return tuple(result[channel] for channel in CHANNELS)

        functions[name + "_outer_jit"] = jax.jit(complete), (params, t, e)
        functions[name + "_public"] = complete, (host_params, duration, epoch)

    # Warm and validate before any timing. Passing all arrays dynamically keeps
    # orbit evaluation inside the analytic kernels rather than constant-folded.
    outputs = {}
    for name, (function, args) in functions.items():
        for _ in range(10):
            outputs[name] = jax.block_until_ready(function(*args))
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(outputs[name])), name

    c_response = complex_pairs(response_dump["xyz"])
    errors = {
        "response_supplied_geometry": relative_error(
            outputs["response_supplied_geometry"], c_response
        ),
        "response_analytic": relative_error(outputs["response_analytic"], c_response),
        "fourier_by_channel": {},
    }
    for index, channel in enumerate(CHANNELS):
        reference = complex_pairs(fourier_dump["channels"][channel])
        errors["fourier_by_channel"][channel] = relative_error(
            outputs["fourier_outer_jit"][index], reference
        )
    assert max(errors["fourier_by_channel"].values()) < 1e-8, errors
    assert errors["response_supplied_geometry"] < 1e-4, errors
    actual = bg.ucb_waveform(
        host_params, T=duration, t0=epoch, BW=bandwidth, backend="fourier"
    )
    assert float(actual["qmin"]) == fourier_dump["qmin"]

    native_blocks = []
    measurements = {name: [] for name in functions}
    generator = np.random.default_rng(1909)
    for _ in range(blocks):
        native_blocks.append(
            next(
                row
                for row in native(
                    executable, "--case", frequency, years, bandwidth, repeats
                )
                if "wavelet_xyz_seconds" in row
            )
        )
        for name in generator.permutation(list(functions)):
            function, args = functions[name]
            start = time.perf_counter_ns()
            for _ in range(repeats):
                jax.block_until_ready(function(*args))
            measurements[name].append((time.perf_counter_ns() - start) / repeats / 1e6)
    return {
        "frequency_hz": frequency,
        "binary_years": years,
        "T": duration,
        "t0": epoch,
        "params": host_params.tolist(),
        "BW": bandwidth,
        "native_response_samples": count,
        "jax_default_response_samples": automatic_response_samples(duration),
        "numerical_checks": errors,
        "jax": {name: summary(values) for name, values in measurements.items()},
        "glass": {
            name: summary([row[name + "_seconds"] * 1000 for row in native_blocks])
            for name in ("response", "fourier_xyzae", "wavelet_xyz")
        },
        "native_blocks": native_blocks,
    }


def main():
    """Record matched-source CPU costs with explicit output-contract limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glass-executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=3)
    args = parser.parse_args()
    if min(args.repeats, args.blocks) < 1:
        parser.error("repeats and blocks must be positive")
    report = {
        "provenance": provenance(),
        "native_executable": str(args.glass_executable.resolve()),
        "native_executable_sha256": hashlib.sha256(
            args.glass_executable.read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "protocol": (
            "Same sources, duration, epoch and BW512. Three blocks by default; "
            "each JAX call synchronized, native C averages same repeat count. "
            "JAX compilation and native orbit/basis setup excluded. Public JAX "
            "uses NumPy inputs; outer jit uses resident arrays, dynamic T/epoch."
        ),
        "limitations": (
            "Only legacy Fourier has matching output contracts. GLASS response "
            "includes orbit spline lookup and signed-polar extraction; JAX returns "
            "complex XYZ with analytic or supplied geometry. Native source splines "
            "and reference phases are prepared outside the response timer. "
            "Full onfly costs compare JAX Fourier output with native WDM output."
        ),
        "repeats": args.repeats,
        "blocks": args.blocks,
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for years in (0.25, 1.0, 4.0):
        result = compare_case(
            args.glass_executable, 0.005, years, 512, args.repeats, args.blocks
        )
        report["runs"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            years,
            "JAX",
            {k: v["median_block_mean_ms"] for k, v in result["jax"].items()},
            flush=True,
        )
        print(
            years,
            "GLASS",
            {k: v["median_block_mean_ms"] for k, v in result["glass"].items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
