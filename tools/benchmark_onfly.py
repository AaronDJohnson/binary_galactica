"""Measure sparse TDI-on-the-fly interpolation accuracy and warmed latency.

Run from the repository root::

    PYTHONPATH=src python tools/benchmark_onfly.py --output /tmp/onfly.json

The dense calculation is an internal numerical comparison: it evaluates the
public arbitrary-time response at every FFT time and applies the heterodyne and
``sqrt(T) / (2*BW)`` normalization explicitly. The public response is the raw
GLASS spline response, in physical Sangria units; production multiplies it by
the same ``sqrt(T) / (2*BW)``. It is not a physical C-reference test; pinned
spline fixtures cover that separately.
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
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import binary_galactica as bg
from binary_galactica.glass.constants import YEAR as RESPONSE_YEAR

# The float64 waveform API requires JAX double precision; importing
# binary_galactica does not enable it.
jax.config.update("jax_enable_x64", True)

BW = 2048
EPOCH = 12_345.0
EXPLICIT_SAMPLE_COUNTS = (32, 64, 128, 256, 512)
YEAR = 31_557_600.0


def sources() -> list[dict[str, Any]]:
    """Return fixed native sources spanning duration, frequency, and chirp."""
    definitions = (
        # Frequency, duration in years, sky cosine, inclination cosine, chirps.
        ("low_frequency_edgeon", 0.001, 0.25, -0.62, 0.02, 0.20, None),
        ("mid_frequency", 0.005, 1.0, 0.27, -0.58, -0.08, None),
        ("mid_frequency_fddot", 0.005, 4.0, 0.81, 0.34, 0.20, 2.0),
        ("high_frequency", 0.030, 1.0, -0.17, 0.91, 0.05, None),
    )
    result = []
    for index, (name, frequency, years, sky, inclination, fdot, fddot) in enumerate(
        definitions
    ):
        duration = years * YEAR
        vector = [
            frequency * duration + 0.25,
            sky,
            0.37 + 1.11 * index,
            np.log(1e-22),
            inclination,
            0.21 + 0.43 * index,
            0.17 + 0.31 * index,
            fdot,
        ]
        if fddot is not None:
            vector.append(fddot)
        result.append(
            {
                "name": name,
                "frequency_hz": frequency,
                "duration_seconds": duration,
                "duration_years": years,
                "native_parameters": vector,
            }
        )
    return result


def wait(result: Any) -> None:
    """Synchronize every output leaf before a timing boundary closes."""
    for leaf in jax.tree_util.tree_leaves(result):
        leaf.block_until_ready()


def public_response_fft(
    response: np.ndarray, phase: np.ndarray, duration: float
) -> np.ndarray:
    """Transform a public ``tdi_response`` envelope into raw spectrum units.

    ``tdi_response`` is the raw GLASS spline response. Production transforms
    it with ``sqrt(T)/(2*BW)``, the factor applied here.
    """
    return np.fft.fftshift(np.fft.fft(response * phase[None, :], axis=-1), axes=-1) * (
        np.sqrt(duration) / (2.0 * BW)
    )


def normalization_sanity_check() -> None:
    """Check the public-response factor using a constant analytic envelope."""
    duration = YEAR
    response = np.ones((3, BW), dtype=np.complex128)
    spectrum = public_response_fft(response, np.ones(BW), duration)
    expected = np.zeros((3, BW), dtype=np.complex128)
    expected[:, BW // 2] = np.sqrt(duration) / 2.0
    np.testing.assert_allclose(spectrum, expected, rtol=0.0, atol=0.0)


def dense_spectrum(params: np.ndarray, duration: float) -> dict[str, np.ndarray]:
    """Construct a dense on-fly reference on the same FFT grid as the waveform."""
    times = EPOCH + duration * np.arange(BW, dtype=np.float64) / BW
    carrier = np.trunc((params[0] / duration) * duration)
    u = times / duration
    cycles = (params[0] - carrier) * u + 0.5 * params[7] * u**2
    if params.shape[0] == 9:
        cycles = cycles + params[8] * u**3 / 6.0
    phase = 2.0 * np.pi * cycles - params[6]
    response = np.asarray(bg.tdi_response(params, jnp.asarray(times), T=duration))
    xyz = public_response_fft(response, np.exp(1j * phase), duration)
    a, e = bg.xyz_to_ae(jnp.asarray(xyz[0]), jnp.asarray(xyz[1]), jnp.asarray(xyz[2]))
    return {
        "X": xyz[0],
        "Y": xyz[1],
        "Z": xyz[2],
        "A": np.asarray(a),
        "E": np.asarray(e),
    }


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    """Return a complex relative L2 error, including an exact-zero fallback."""
    denominator = np.linalg.norm(reference)
    numerator = np.linalg.norm(value - reference)
    return float(numerator if denominator == 0 else numerator / denominator)


def automatic_response_samples(duration: float) -> int:
    """Mirror the package's duration-density automatic sparse-grid policy."""
    target = max(32.0, 128.0 * duration / RESPONSE_YEAR)
    for count in (32, 64, 128, 256, 512, 1024, 2048):
        if count >= target:
            return count
    return 2048


def accuracy_case(
    source: dict[str, Any], response_samples: int | None
) -> dict[str, Any]:
    """Compare sparse public spectra with the direct dense response FFT."""
    params = np.asarray(source["native_parameters"], dtype=np.float64)
    duration = float(source["duration_seconds"])
    reference = dense_spectrum(params, duration)
    sparse = bg.ucb_waveform(
        params,
        T=duration,
        BW=BW,
        t0=EPOCH,
        backend="onfly",
        response_samples=response_samples,
    )
    errors = {
        name: relative_l2(np.asarray(sparse[name]), reference[name])
        for name in reference
    }
    return {
        "response_samples": response_samples,
        "resolved_response_samples": (
            automatic_response_samples(duration)
            if response_samples is None
            else response_samples
        ),
        "relative_l2_by_channel": errors,
        "max_relative_l2": max(errors.values()),
    }


def fourier_baseline(source: dict[str, Any]) -> dict[str, Any]:
    """Report the expected-different legacy response against the dense spline FFT."""
    params = np.asarray(source["native_parameters"], dtype=np.float64)
    duration = float(source["duration_seconds"])
    reference = dense_spectrum(params, duration)
    legacy = bg.ucb_waveform(params, T=duration, BW=BW, t0=EPOCH, backend="fourier")
    errors = {
        name: relative_l2(np.asarray(legacy[name]), reference[name])
        for name in reference
    }
    return {
        "backend": "fourier",
        "meaning": "legacy response difference; not an interpolation-accuracy metric",
        "relative_l2_by_channel": errors,
        "max_relative_l2": max(errors.values()),
    }


def timing(
    name: str, function: Callable[[jnp.ndarray], Any], params: np.ndarray, repeats: int
) -> dict[str, Any]:
    """Record synchronized calls; earlier accuracy checks may have compiled them.

    ``first_call_ms`` is the first call in this timing section, not a cold
    compilation measurement. Use the repeated medians to compare warm latency.
    """
    values = jnp.asarray(params)
    start = time.perf_counter_ns()
    result = function(values)
    wait(result)
    first = time.perf_counter_ns() - start
    for _ in range(2):
        wait(function(values))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function(values)
        wait(result)
        samples.append(time.perf_counter_ns() - start)
    milliseconds = np.asarray(samples, dtype=np.float64) / 1e6
    return {
        "name": name,
        "first_call_ms": first / 1e6,
        "warm_repeat_ms": [value / 1e6 for value in samples],
        "summary": {
            "median_ms": float(np.median(milliseconds)),
            "mean_ms": float(np.mean(milliseconds)),
            "min_ms": float(np.min(milliseconds)),
            "max_ms": float(np.max(milliseconds)),
        },
    }


def provenance() -> dict[str, Any]:
    """Capture the imported source and enough platform data to reproduce a run."""
    package = Path(bg.__file__).resolve().parent

    def git(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(package), *arguments],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    tree = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        tree.update(path.relative_to(package).as_posix().encode())
        tree.update(path.read_bytes())
    return {
        "package_directory": str(package),
        "package_version": bg.__version__,
        "source_tree_sha256": tree.hexdigest(),
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "jax": jax.__version__,
        "jaxlib": metadata.version("jaxlib"),
        "numpy": np.__version__,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }


def parse_arguments() -> argparse.Namespace:
    """Parse the deliberately small benchmark interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument(
        "--repeats", type=int, default=10, help="warmed timing repetitions"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="measure one fiducial source at automatic and explicit 512 samples",
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    return arguments


def main() -> None:
    """Write accuracy and performance results without making physical claims."""
    arguments = parse_arguments()
    normalization_sanity_check()
    cases = sources()
    sample_counts: tuple[int | None, ...] = (None, *EXPLICIT_SAMPLE_COUNTS)
    if arguments.quick:
        cases = cases[:1]
        sample_counts = (None, 512)

    results = []
    for source in cases:
        print(f"checking {source['name']}", file=sys.stderr, flush=True)
        samples = [accuracy_case(source, count) for count in sample_counts]
        source_maximum = max(item["max_relative_l2"] for item in samples)
        results.append(
            {
                **source,
                "accuracy": samples,
                "source_max_relative_l2": source_maximum,
                "fourier_baseline": fourier_baseline(source),
            }
        )

    timing_source = np.asarray(cases[0]["native_parameters"], dtype=np.float64)
    timing_duration = float(cases[0]["duration_seconds"])
    onfly_default = lambda p: bg.ucb_waveform(  # noqa: E731 -- compact timing fixture
        p, T=timing_duration, BW=BW, t0=EPOCH, backend="onfly", response_samples=None
    )
    onfly_default_outer_jit = jax.jit(onfly_default)
    onfly_fixed_512 = lambda p: bg.ucb_waveform(  # noqa: E731 -- compact timing fixture
        p, T=timing_duration, BW=BW, t0=EPOCH, backend="onfly", response_samples=512
    )
    onfly_fixed_512_outer_jit = jax.jit(onfly_fixed_512)
    fourier = lambda p: bg.ucb_waveform(  # noqa: E731 -- compact timing fixture
        p, T=timing_duration, BW=BW, t0=EPOCH, backend="fourier", response_samples=None
    )
    knots = EPOCH + timing_duration * (1 - 1 / BW) * jnp.linspace(0.0, 1.0, 512)
    response_only = jax.jit(lambda p: bg.tdi_response(p, knots, T=timing_duration))
    response_with_derivative = jax.jit(
        lambda p: jax.jvp(
            lambda at: bg.tdi_response(p, at, T=timing_duration),
            (knots,),
            (jnp.ones_like(knots),),
        )
    )
    timing_results = [
        timing(
            "onfly_automatic_native_public_call",
            onfly_default,
            timing_source,
            arguments.repeats,
        ),
        timing(
            "onfly_automatic_outer_jit",
            onfly_default_outer_jit,
            timing_source,
            arguments.repeats,
        ),
        timing(
            "onfly_explicit_512_native_public_call",
            onfly_fixed_512,
            timing_source,
            arguments.repeats,
        ),
        timing(
            "onfly_explicit_512_outer_jit",
            onfly_fixed_512_outer_jit,
            timing_source,
            arguments.repeats,
        ),
        timing("fourier_native_public_call", fourier, timing_source, arguments.repeats),
        # These separately compiled stages diagnose response cost. Their times
        # are not additive components of the jointly compiled waveform.
        timing(
            "onfly_response_only_explicit_512_knots",
            response_only,
            timing_source,
            arguments.repeats,
        ),
        timing(
            "onfly_response_and_time_derivative_explicit_512_knots",
            response_with_derivative,
            timing_source,
            arguments.repeats,
        ),
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "dense-grid interpolation convergence and warmed waveform latency",
        "physical_validity": (
            "not assessed; use pinned spline C fixtures for response validity"
        ),
        "configuration": {
            "BW": BW,
            "epoch_seconds": EPOCH,
            "accuracy_response_samples": list(sample_counts),
            "timing_response_samples": {
                "automatic": None,
                "explicit_comparison": 512,
                "diagnostic_knots": 512,
            },
            "automatic_response_sampling": {
                "argument": None,
                "density_per_year": 128,
                "year_seconds": RESPONSE_YEAR,
                "buckets": [32, 64, 128, 256, 512, 1024, 2048],
                "cap": 2048,
            },
            "repeats": arguments.repeats,
            "quick": arguments.quick,
        },
        "provenance": provenance(),
        "sources": results,
        "timing_source": cases[0]["name"],
        "timing": timing_results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {arguments.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
