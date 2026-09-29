"""Diagnose sparse-response compilation costs without changing production code.

Run serially, without competing benchmarks or tests::

    PYTHONPATH=src python tools/diagnose_onfly.py --output /tmp/diagnosis.json

All waveform timings return XYZ from precompiled executables with dynamic source,
duration and epoch inputs. They exclude validation, compilation and output packing
in the ordinary public API. Interpolation variants are diagnostic experiments,
not replacements validated for AD, invalid domains or arbitrary source grids.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_onfly import provenance

import binary_galactica as bg
import binary_galactica.glass.waveform as waveform
from binary_galactica.glass.onfly import reference_phase_shift, tdi_response
from binary_galactica.glass.orbit import slow_time_grid
from binary_galactica.glass.sparse import _coefficient_matrix, interpolate_uniform

# The float64 waveform API requires JAX double precision; importing
# binary_galactica does not enable it.
jax.config.update("jax_enable_x64", True)

SAMPLES = 32
DURATION = 7_884_000.0
EPOCH = 12_345.0
PARAMETERS = [
    0.003 * DURATION + 0.25,
    -0.62,
    0.37,
    np.log(1e-22),
    0.02,
    0.21,
    0.17,
    0.2,
]


def gathered_interpolation(values, coordinates):
    """Reproduce the pre-fix sampler: windows gathered directly from the response."""
    values = jnp.asarray(values, dtype=jnp.complex128)
    count, channels = values.shape
    points = min(8, count)
    center = points // 2 - 1
    windows = jnp.arange(count - points + 1)[:, None] + jnp.arange(points)[None, :]
    samples = values[windows]
    anchor = samples[:, center, :]
    matrix = jnp.asarray(_coefficient_matrix(points), dtype=jnp.float64)
    coefficients = jnp.einsum("kj,wjc->wck", matrix, samples - anchor[:, None, :])
    coefficients = coefficients.at[..., 0].add(anchor)
    sample_coordinate = coordinates.ravel() * (count - 1)
    cell = jnp.floor(sample_coordinate).astype(jnp.int32)
    window = jnp.clip(cell - center, 0, count - points)
    local = sample_coordinate - (window + center)
    selected = coefficients[window]
    result = selected[..., -1]
    for degree in range(points - 2, -1, -1):
        result = result * local[:, None] + selected[..., degree]
    return result.reshape(coordinates.shape + (channels,))


def conditional_interpolation(values, coordinates):
    """Introduce a runtime boundary after the response, for finite test inputs."""
    values = jax.lax.cond(
        jnp.isfinite(values[0, 0].real),
        lambda x: x,
        lambda x: jnp.full_like(x, jnp.nan),
        values,
    )
    return interpolate_uniform(values, coordinates)


def barrier_interpolation(values, coordinates):
    """Test whether the nominal optimization barrier prevents gather fusion."""
    return interpolate_uniform(jax.lax.optimization_barrier(values), coordinates)


def matrix_interpolation(bandwidth: int) -> Callable:
    """Precompute the same linear interpolation operator as a diagnostic."""
    matrix = np.asarray(
        jax.jit(
            lambda values: interpolate_uniform(
                values, jnp.arange(bandwidth) / (bandwidth - 1)
            )
        )(jnp.eye(SAMPLES, dtype=jnp.complex128))
    ).real

    def interpolate(values, coordinates):
        del coordinates
        weights = jnp.asarray(matrix)
        return weights @ values.real + 1j * (weights @ values.imag)

    return interpolate


def measure(functions: dict[str, tuple[Any, tuple]], repeats: int) -> dict:
    """Interleave warmed, synchronized executions to reduce ordering effects."""
    for function, arguments in functions.values():
        for _ in range(5):
            jax.block_until_ready(function(*arguments))
    samples: dict[str, list[float]] = {name: [] for name in functions}
    generator = np.random.default_rng(1837)
    for _ in range(repeats):
        for name in generator.permutation(list(functions)):
            function, arguments = functions[name]
            start = time.perf_counter_ns()
            jax.block_until_ready(function(*arguments))
            samples[name].append((time.perf_counter_ns() - start) / 1e3)
    return {
        name: {
            "median_us": float(np.median(values)),
            "p10_us": float(np.percentile(values, 10)),
            "p90_us": float(np.percentile(values, 90)),
            "samples_us": values,
        }
        for name, values in samples.items()
    }


def compile_waveform(bandwidth, backend, count, method, arguments):
    """Temporarily select an interpolation experiment, restoring the module."""
    original = waveform.interpolate_uniform
    try:
        waveform.interpolate_uniform = method
        waveform._ucb_waveform.clear_cache()

        def evaluate(params, duration, epoch):
            result = bg.ucb_waveform(
                params,
                T=duration,
                t0=epoch,
                BW=bandwidth,
                backend=backend,
                response_samples=count,
            )
            return jnp.stack([result[name] for name in ("X", "Y", "Z")])

        start = time.perf_counter()
        compiled = jax.jit(evaluate).lower(*arguments).compile()
        return compiled, time.perf_counter() - start
    finally:
        waveform.interpolate_uniform = original
        waveform._ucb_waveform.clear_cache()


def diagnose(bandwidth: int, repeats: int, hlo_directory: Path | None) -> dict:
    """Compare equivalent sparse kernels and their isolated component costs."""
    params = jnp.asarray(PARAMETERS)
    duration, epoch = jnp.asarray(DURATION), jnp.asarray(EPOCH)
    arguments = (params, duration, epoch)
    functions = {}
    compile_seconds = {}
    variants = (
        ("fourier", "fourier", None, interpolate_uniform),
        ("automatic", "onfly", None, interpolate_uniform),
        ("fixed32", "onfly", SAMPLES, interpolate_uniform),
        ("gathered", "onfly", SAMPLES, gathered_interpolation),
        ("barrier", "onfly", SAMPLES, barrier_interpolation),
        ("materialized", "onfly", SAMPLES, conditional_interpolation),
        ("matrix", "onfly", SAMPLES, matrix_interpolation(bandwidth)),
    )
    for name, backend, count, method in variants:
        compiled, elapsed = compile_waveform(
            bandwidth, backend, count, method, arguments
        )
        functions[name] = compiled, arguments
        compile_seconds[name] = elapsed
        if hlo_directory is not None:
            (hlo_directory / f"{name}-{bandwidth}.hlo").write_text(compiled.as_text())
    timings = measure(functions, repeats)
    reference = np.asarray(functions["fixed32"][0](*arguments))
    for name, (function, args) in functions.items():
        actual = np.asarray(function(*args))
        assert np.isfinite(actual).all(), name
        timings[name]["compile_seconds"] = compile_seconds[name]
        if name != "fourier":
            error = float(
                np.linalg.norm(actual - reference) / np.linalg.norm(reference)
            )
            timings[name]["relative_l2_to_fixed32"] = error
            assert error < 1e-12, (name, error)

    def raw(params, duration, epoch):
        knots = epoch + duration * (1 - 1 / bandwidth) * jnp.linspace(0.0, 1.0, SAMPLES)
        return tdi_response(params, knots, duration, reference_phase=True).T

    raw_compiled = jax.jit(raw).lower(*arguments).compile()
    values = raw_compiled(*arguments)
    dense = jnp.ones((3, bandwidth), dtype=jnp.complex128)
    components = {"raw32": (raw_compiled, arguments)}
    for name, function, args in (
        (
            "interpolation32",
            lambda v: interpolate_uniform(v, jnp.arange(bandwidth) / (bandwidth - 1)),
            (values,),
        ),
        (
            "response_and_interpolation32",
            lambda p, t, e: waveform._response_envelope(
                p, t, bandwidth, e, None, SAMPLES
            ),
            arguments,
        ),
        (
            "dense_reference_phase",
            lambda p, t, e: reference_phase_shift(
                p, slow_time_grid(t, e, bandwidth), t
            ),
            arguments,
        ),
        ("fft3", lambda v: jnp.fft.fft(v, axis=-1), (dense,)),
        ("fft6", lambda v: jnp.fft.fft(v, axis=-1), (jnp.concatenate((dense, dense)),)),
    ):
        components[name] = jax.jit(function).lower(*args).compile(), args
    return {
        "BW": bandwidth,
        "waveforms": timings,
        "standalone_components_not_additive": measure(components, repeats),
    }


def main() -> None:
    """Record synchronized timings and optional optimized compiler output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--hlo-directory", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if arguments.hlo_directory is not None:
        arguments.hlo_directory.mkdir(parents=True, exist_ok=True)
    report = {
        "provenance": provenance(),
        "description": (
            "Precompiled XYZ kernels; dynamic source, duration and epoch; "
            "serial interleaved synchronized timings"
        ),
        "limitations": (
            "Finite single-source fixture only. Conditional and matrix variants "
            "are diagnostics, not validated production replacements. Component "
            "times are not additive. Fourier has a different response convention."
        ),
        "T": DURATION,
        "t0": EPOCH,
        "native_parameters": PARAMETERS,
        "response_samples": SAMPLES,
        "repeats": arguments.repeats,
        "runs": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for bandwidth in (128, 512, 2048):
        result = diagnose(bandwidth, arguments.repeats, arguments.hlo_directory)
        report["runs"].append(result)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            bandwidth,
            {name: row["median_us"] for name, row in result["waveforms"].items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
