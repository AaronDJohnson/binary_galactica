"""Reproducible timings of the package's public detector-waveform APIs.

Run from the repository root::

    PYTHONPATH=src python tools/benchmark_waveform.py --output /tmp/waveform.json

Point PYTHONPATH at another source tree to benchmark that implementation with
the same harness. Reports identify the imported package, not the current
working directory. Run comparisons serially, with the same Python environment,
device settings, fixture, and arguments. These timings cover waveform generation,
window placement, batching, and derivatives; there is no likelihood calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import binary_galactica as bg

# The float64 waveform API requires JAX double precision; importing
# binary_galactica does not enable it.
jax.config.update("jax_enable_x64", True)

SEED = 20260913
DURATION = 7_884_000.0
EPOCH = 12_345.0
FORMAT = "sangria"
NATIVE_NAMES = (
    "f0*T",
    "cos(theta)",
    "longitude",
    "log(amplitude)",
    "cos(iota)",
    "psi",
    "phi0",
    "fdot*T**2",
)
FULL_OUTPUTS = "X, Y, Z, A, E, qmin, freq_bin_indices (raw GLASS units)"


def source_snapshot() -> dict[str, Any]:
    """Hash the resolved imported source, including untracked Python modules."""
    package_root = Path(bg.__file__).resolve().parent
    files = sorted(package_root.rglob("*.py"))
    tree_hash = hashlib.sha256()
    hashes = {}
    for path in files:
        relative = path.relative_to(package_root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative] = digest
        tree_hash.update(f"{relative}\0{digest}\n".encode())

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(package_root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "package_directory": str(package_root),
        "package_version": bg.__version__,
        "python_source_tree_sha256": tree_hash.hexdigest(),
        "python_source_files_sha256": hashes,
        "git_root": git("rev-parse", "--show-toplevel"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
    }


def environment() -> dict[str, Any]:
    """Describe the runtime and explicit JAX/compiler/thread settings."""
    setting_names = (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_ENABLE_X64",
        "JAX_DISABLE_JIT",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_ENABLE_COMPILATION_CACHE",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "XLA_FLAGS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "CUDA_VISIBLE_DEVICES",
    )
    devices = [
        {
            "id": device.id,
            "platform": device.platform,
            "device_kind": device.device_kind,
            "description": str(device),
        }
        for device in jax.devices()
    ]
    config_names = (
        "jax_enable_x64",
        "jax_disable_jit",
        "jax_enable_compilation_cache",
        "jax_compilation_cache_dir",
        "jax_default_matmul_precision",
    )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "jax": jax.__version__,
        "jaxlib": metadata.version("jaxlib"),
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "devices": devices,
        "jax_config": {name: getattr(jax.config, name, None) for name in config_names},
        "environment_variables": {name: os.environ.get(name) for name in setting_names},
    }


def make_catalogue(size: int) -> tuple[np.ndarray, list[bg.GBParams]]:
    """Draw a fixed local source population, safely inside coordinate domains."""
    rng = np.random.default_rng(SEED)
    native = np.empty((size, 8), dtype=np.float64)
    # Stay near 3 mHz so one fixed analysis window includes every source band.
    # Quarter-bin offsets avoid the carrier discontinuity in the Jacobian case.
    native[:, 0] = 23_652.0 + rng.integers(-16, 17, size=size) + 0.25
    native[:, 1] = rng.uniform(-0.9, 0.9, size)
    native[:, 2] = rng.uniform(0.0, 2.0 * np.pi, size)
    native[:, 3] = rng.uniform(np.log(1e-23), np.log(1e-20), size)
    native[:, 4] = rng.uniform(-0.9, 0.9, size)
    native[:, 5] = rng.uniform(0.0, np.pi, size)
    native[:, 6] = rng.uniform(0.0, 2.0 * np.pi, size)
    native[:, 7] = rng.uniform(-0.05, 0.2, size)
    physical = [
        bg.GBParams(
            amp=float(np.exp(p[3])),
            f0=float(p[0] / DURATION),
            fdot=float(p[7] / DURATION**2),
            phi0=float(-p[6]),
            iota=float(np.arccos(p[4])),
            psi=float(p[5]),
            lam=float(p[2]),
            beta=float(np.arcsin(p[1])),
        )
        for p in native
    ]
    return native, physical


def synchronize(result: Any) -> None:
    """Wait for every array leaf, retaining full waveform channel computation."""
    for leaf in jax.tree_util.tree_leaves(result):
        leaf.block_until_ready()


def output_description(result: Any) -> list[dict[str, Any]]:
    """Validate and fingerprint the final output outside the timed interval."""
    descriptions = []
    for leaf in jax.tree_util.tree_leaves(result):
        array = np.asarray(leaf)
        if not np.all(np.isfinite(array)):
            raise RuntimeError("Benchmark produced a nonfinite output")
        descriptions.append(
            {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "max_abs": float(np.max(np.abs(array))),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    return descriptions


def summarize(samples_ns: list[int], sources_per_call: int) -> dict[str, float]:
    """Summarize synchronized wall latency, with explicit batch amortization."""
    values = np.asarray(samples_ns, dtype=np.float64) / 1e6
    median = float(np.median(values))
    return {
        "median_ms": median,
        "mean_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "p05_ms": float(np.percentile(values, 5)),
        "p95_ms": float(np.percentile(values, 95)),
        "stddev_ms": float(np.std(values)),
        "amortized_median_ms_per_source": median / sources_per_call,
        "sources_per_second_at_median": 1000.0 * sources_per_call / median,
    }


def measure(
    name: str,
    function: Callable[..., Any],
    arguments: list[tuple[Any, ...]],
    *,
    repeats: int,
    warmups: int,
    outputs: str,
    input_description: str,
    sources_per_call: int = 1,
) -> dict[str, Any]:
    """Time a cache-cleared first call, untallied warmups, and varied repeats."""
    jax.clear_caches()

    def timed(index: int) -> tuple[int, Any]:
        args = arguments[index % len(arguments)]
        start = time.perf_counter_ns()
        result = function(*args)
        synchronize(result)
        return time.perf_counter_ns() - start, result

    first_ns, result = timed(0)
    first_output = output_description(result)
    warmup_ns = [timed(index + 1)[0] for index in range(warmups)]
    samples_ns = []
    indices = []
    for index in range(repeats):
        # Start measured calls at fixture zero in every workload for pairing.
        elapsed, result = timed(index)
        samples_ns.append(elapsed)
        indices.append(index % len(arguments))
    summary = summarize(samples_ns, sources_per_call)
    print(
        f"  {name:36s} first {first_ns / 1e6:9.2f} ms; "
        f"warm median {summary['median_ms']:9.4f} ms",
        file=sys.stderr,
        flush=True,
    )
    return {
        "name": name,
        "inputs": input_description,
        "outputs": outputs,
        "sources_per_call": sources_per_call,
        "first_call_after_clear_caches_ns": first_ns,
        "warmup_ns": warmup_ns,
        "raw_repeat_ns": samples_ns,
        "repeat_fixture_indices": indices,
        "summary": summary,
        "first_output_leaves": first_output,
        "last_output_leaves": output_description(result),
    }


def benchmark_bandwidth(
    bandwidth: int,
    native: np.ndarray,
    physical: list[bg.GBParams],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Exercise eager and transformed public APIs at one fixed bandwidth."""
    print(f"BW={bandwidth}", file=sys.stderr, flush=True)
    positions = bg.orbit_cache(DURATION, EPOCH, bandwidth)
    synchronize(positions)
    duration = np.float64(DURATION)
    epoch = np.float64(EPOCH)
    # Single-source transformed inputs and batches are already on the device;
    # eager native paths deliberately include NumPy input handling/transfers.
    device_params = [jax.device_put(p) for p in native]
    batch_count = len(native) // args.batch_size
    batches = [
        jax.device_put(native[i * args.batch_size : (i + 1) * args.batch_size])
        for i in range(batch_count)
    ]
    synchronize((device_params, batches))
    native_args = [(p, duration, positions, epoch) for p in native]
    physical_args = [(p, duration, positions, epoch) for p in physical]
    device_args = [(p, duration, positions, epoch) for p in device_params]
    batch_args = [(p, duration, positions, epoch) for p in batches]

    def native_call(p, duration, cached_positions, epoch):
        return bg.ucb_waveform(
            p,
            T=duration,
            BW=bandwidth,
            t0=epoch,
            fmt=FORMAT,
            positions=cached_positions,
            backend=args.backend,
            response_samples=args.response_samples,
        )

    def uncached_call(p, duration, cached_positions, epoch):
        return bg.ucb_waveform(
            p,
            T=duration,
            BW=bandwidth,
            t0=epoch,
            fmt=FORMAT,
            backend=args.backend,
            response_samples=args.response_samples,
        )

    def physical_call(p, duration, cached_positions, epoch):
        return bg.tdi_waveform(
            p,
            T=duration,
            BW=bandwidth,
            t0=epoch,
            fmt=FORMAT,
            positions=cached_positions,
            backend=args.backend,
            response_samples=args.response_samples,
        )

    bin_lo = 23_652 - bandwidth // 2 - 32
    n_bins = bandwidth + 64

    def window_call(p, duration, cached_positions, epoch):
        return bg.template_on_window(
            p,
            T=duration,
            BW=bandwidth,
            t0=epoch,
            fmt=FORMAT,
            positions=cached_positions,
            backend=args.backend,
            response_samples=args.response_samples,
            bin_lo=bin_lo,
            n_bins=n_bins,
        )

    def uncached_window_call(p, duration, cached_positions, epoch):
        return bg.template_on_window(
            p,
            T=duration,
            BW=bandwidth,
            t0=epoch,
            fmt=FORMAT,
            backend=args.backend,
            response_samples=args.response_samples,
            bin_lo=bin_lo,
            n_bins=n_bins,
        )

    def ae_vector(p, duration, cached_positions, epoch):
        waveform = native_call(p, duration, cached_positions, epoch)
        channels = jnp.stack((waveform["A"], waveform["E"]))
        return jnp.stack((channels.real, channels.imag))

    def uncached_ae_vector(p, duration, cached_positions, epoch):
        waveform = uncached_call(p, duration, cached_positions, epoch)
        channels = jnp.stack((waveform["A"], waveform["E"]))
        return jnp.stack((channels.real, channels.imag))

    host_input = "NumPy float64 native vector; cached JAX positions"
    device_input = "device float64 native vector; cached JAX positions"
    uncached_device_input = "device float64 native vector; analytic orbit in waveform"
    # T is an explicit dynamic argument throughout; closing over a Python T
    # can change float64 carrier rounding when XLA rewrites division.
    workloads = [
        (
            "native_eager_cached_numpy",
            native_call,
            native_args,
            FULL_OUTPUTS,
            host_input,
            1,
        ),
        (
            "physical_eager_cached_gbparams",
            physical_call,
            physical_args,
            FULL_OUTPUTS,
            "GBParams with Python float fields; cached JAX positions",
            1,
        ),
        (
            "native_outer_jit_cached",
            jax.jit(native_call),
            device_args,
            FULL_OUTPUTS,
            device_input,
            1,
        ),
        (
            "fixed_window_ae_outer_jit_cached",
            jax.jit(window_call),
            device_args,
            "A/E, shape (2, BW+64), continuous Fourier units",
            device_input,
            1,
        ),
        (
            "batch_native_outer_jit_cached",
            jax.jit(jax.vmap(native_call, in_axes=(0, None, None, None))),
            batch_args,
            FULL_OUTPUTS + "; leading batch axis on every leaf",
            "device float64 native batch; shared cached JAX positions",
            args.batch_size,
        ),
        (
            "jacfwd_ae_outer_jit_cached",
            jax.jit(jax.jacfwd(ae_vector, argnums=0)),
            device_args,
            "Jacobian (real/imag, A/E, BW, 8 native parameters), raw GLASS units",
            device_input,
            1,
        ),
        (
            "native_eager_uncached_numpy",
            uncached_call,
            native_args,
            FULL_OUTPUTS,
            "NumPy float64 native vector; analytic orbit in waveform",
            1,
        ),
        (
            "native_outer_jit_uncached",
            jax.jit(uncached_call),
            device_args,
            FULL_OUTPUTS,
            uncached_device_input,
            1,
        ),
        (
            "fixed_window_ae_outer_jit_uncached",
            jax.jit(uncached_window_call),
            device_args,
            "A/E, shape (2, BW+64), continuous Fourier units",
            uncached_device_input,
            1,
        ),
        (
            "batch_native_outer_jit_uncached",
            jax.jit(jax.vmap(uncached_call, in_axes=(0, None, None, None))),
            batch_args,
            FULL_OUTPUTS + "; leading batch axis on every leaf",
            "device float64 native batch; analytic orbit in waveform",
            args.batch_size,
        ),
        (
            "jacfwd_ae_outer_jit_uncached",
            jax.jit(jax.jacfwd(uncached_ae_vector, argnums=0)),
            device_args,
            "Jacobian (real/imag, A/E, BW, 8 native parameters), raw GLASS units",
            uncached_device_input,
            1,
        ),
    ]
    records = [
        measure(
            name,
            function,
            inputs,
            repeats=args.repeats,
            warmups=args.warmups,
            outputs=outputs,
            input_description=description,
            sources_per_call=count,
        )
        for name, function, inputs, outputs, description, count in workloads
    ]
    return {
        "bandwidth": bandwidth,
        "cached_positions_shape": list(positions.shape),
        "cached_positions_sha256": hashlib.sha256(
            np.asarray(positions).tobytes()
        ).hexdigest(),
        "window": {"bin_lo": bin_lo, "n_bins": n_bins},
        "batch_fixture_count": batch_count,
        "workloads": records,
    }


def parse_response_samples(value: str) -> int | None:
    """Accept an explicit sparse grid size or the automatic ``none`` policy."""
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "response-samples must be an integer at least 4 or 'none'"
        ) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse reproducibility settings and reject empty timing runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Write JSON report (default: stdout)"
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--bandwidths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backend", choices=bg.BACKENDS, default="onfly")
    parser.add_argument(
        "--response-samples",
        type=parse_response_samples,
        default=bg.DEFAULT_RESPONSE_SAMPLES,
        metavar="INTEGER|none",
        help="sparse grid size, or none for the package's automatic default",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmups < 1 or args.batch_size < 1:
        parser.error("repeats, warmups, and batch-size must be positive")
    if any(bandwidth < 2 or bandwidth % 2 for bandwidth in args.bandwidths):
        parser.error("bandwidths must be positive even integers")
    if len(set(args.bandwidths)) != len(args.bandwidths):
        parser.error("bandwidths must be distinct")
    if args.response_samples is not None and args.response_samples < 4:
        parser.error("response-samples must be an integer at least 4")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run all waveform workloads and emit a self-contained JSON report."""
    args = parse_args(argv)
    snapshot = source_snapshot()
    runtime = environment()
    if not jax.config.jax_enable_x64 or jax.config.jax_disable_jit:
        raise RuntimeError("This benchmark requires float64 and enabled JIT")
    size = max(64, 2 * args.batch_size)
    native, physical = make_catalogue(size)
    script = Path(__file__).resolve()
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "harness_path": str(script),
        "harness_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "source": snapshot,
        "environment": runtime,
        "settings": {
            "bandwidths": args.bandwidths,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "batch_size": args.batch_size,
            "backend": args.backend,
            "response_samples": args.response_samples,
        },
        "methodology": {
            "clock": "time.perf_counter_ns; synchronized wall-clock latency",
            "first_call": (
                "After jax.clear_caches(); includes tracing/compilation and execution. "
                "Existing persistent disk caches may still be reused."
            ),
            "warm_latency": (
                "Sequential calls; every output leaf blocks before stopping the clock. "
                "Warmup calls are excluded from repeat summaries."
            ),
            "included": (
                "Public API validation, dispatch, device execution, synchronization; "
                "NumPy input handling/transfers in eager paths."
            ),
            "excluded": (
                "Imports, backend initialization, fixture creation, orbit caching, "
                "device-input preparation, output validation and report serialization."
            ),
            "dynamic_arguments": (
                "Observation duration T, epoch t0, parameters, cached positions; "
                "BW, backend, response_samples, format and window shape are static."
            ),
            "batch_order": (
                "Contiguous disjoint catalogue slices, cycling across repeats; "
                "amortized per-source latency is batch latency divided by batch size."
            ),
            "scope": (
                "Waveform APIs and A/E derivatives; "
                "no likelihood, sampler, or C baseline."
            ),
        },
        "fixture": {
            "seed": SEED,
            "rng": "numpy.default_rng (PCG64)",
            "T_seconds": DURATION,
            "t0_seconds": EPOCH,
            "format": FORMAT,
            "catalogue_size": size,
            "native_parameter_names": NATIVE_NAMES,
            "native_catalogue": native.tolist(),
            "physical_parameter_names": bg.PARAMETER_NAMES,
            "physical_catalogue": [list(p) for p in physical],
            "native_catalogue_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
        },
        "results": [
            benchmark_bandwidth(bw, native, physical, args) for bw in args.bandwidths
        ],
    }
    if (
        source_snapshot()["python_source_tree_sha256"]
        != snapshot["python_source_tree_sha256"]
    ):
        raise RuntimeError(
            "Imported source changed during benchmark; rerun on a stable tree"
        )
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Report: {args.output.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
