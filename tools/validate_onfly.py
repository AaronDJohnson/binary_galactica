"""Validate production sparse on-fly spectra against a dense public response.

Run from the repository root::

    PYTHONPATH=src python tools/validate_onfly.py

This is a numerical interpolation-convergence check over a deterministic
90-case source grid. It is not a replacement for the pinned C spline-response
fixtures, and intentionally does not assert a universal error threshold.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from benchmark_onfly import provenance

import binary_galactica as bg
from binary_galactica.glass.constants import YEAR as RESPONSE_YEAR

BW = 2048
EPOCHS = (12_345.0, -0.2 * 31_557_600.0, 2.1 * 31_557_600.0)
FREQUENCIES = (0.0001, 0.001, 0.005, 0.03, 0.1)
INCLINATIONS = (0.0, 1.0, -1.0, 0.0001, 0.3, -0.8)
SKY_COSINES = (-1.0, 1.0, 0.0, 0.99999, -0.65, 0.38)
YEAR = 31_557_600.0
YEARS = (0.25, 1.0, 4.0)
RESPONSE_SAMPLES = (None, 256, 512)


def automatic_response_samples(duration: float) -> int:
    """Return the package's documented automatic duration bucket."""
    target = max(32.0, 128.0 * duration / RESPONSE_YEAR)
    for count in (32, 64, 128, 256, 512, 1024, 2048):
        if count >= target:
            return count
    return 2048


def public_response_fft(
    response: np.ndarray, phase: np.ndarray, duration: float
) -> np.ndarray:
    """Transform public (raw-spline/2) response values into spectrum units."""
    return np.fft.fftshift(np.fft.fft(response * phase[None, :], axis=-1), axes=-1) * (
        np.sqrt(duration) / (2.0 * BW)
    )


def dense_reference(
    params: np.ndarray, duration: float, epoch: float
) -> dict[str, np.ndarray]:
    """Evaluate public XYZ at every FFT time and restore its analytic phase."""
    times = epoch + duration * np.arange(BW, dtype=np.float64) / BW
    carrier = np.trunc((params[0] / duration) * duration)
    u = times / duration
    cycles = (params[0] - carrier) * u
    if params.shape[0] > 7:
        cycles = cycles + 0.5 * params[7] * u**2
    if params.shape[0] > 8:
        cycles = cycles + params[8] * u**3 / 6.0
    response = np.asarray(bg.tdi_response(params, jnp.asarray(times), T=duration))
    xyz = public_response_fft(
        response, np.exp(1j * (2.0 * np.pi * cycles - params[6])), duration
    )
    a, e = bg.xyz_to_ae(jnp.asarray(xyz[0]), jnp.asarray(xyz[1]), jnp.asarray(xyz[2]))
    return {
        "X": xyz[0],
        "Y": xyz[1],
        "Z": xyz[2],
        "A": np.asarray(a),
        "E": np.asarray(e),
    }


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    """Calculate relative complex L2 error, retaining the exact-zero case."""
    denominator = np.linalg.norm(reference)
    numerator = np.linalg.norm(value - reference)
    return float(numerator if denominator == 0 else numerator / denominator)


def cases() -> list[dict[str, Any]]:
    """Reproduce the deterministic 90-case review grid without prototype imports."""
    rng = np.random.default_rng(731)
    result = []
    for index in range(90):
        frequency = FREQUENCIES[index % len(FREQUENCIES)]
        years = YEARS[(index // len(FREQUENCIES)) % len(YEARS)]
        duration = years * YEAR
        sky = SKY_COSINES[(index // 15) % len(SKY_COSINES)]
        inclination = INCLINATIONS[index % len(INCLINATIONS)]
        count = 7 + (index // 3) % 3
        epoch = EPOCHS[index % len(EPOCHS)]
        params = np.array(
            [
                frequency * duration + 0.25,
                sky,
                rng.uniform(0.0, 2.0 * np.pi),
                np.log(1e-22),
                inclination,
                rng.uniform(0.0, np.pi),
                0.41,
                0.2 * (-1) ** index,
                2.0 * (-1) ** index,
            ],
            dtype=np.float64,
        )[:count]
        result.append(
            {
                "index": index,
                "frequency_hz": frequency,
                "duration_years": years,
                "duration_seconds": duration,
                "sky_cosine": sky,
                "inclination_cosine": inclination,
                "parameter_count": count,
                "epoch_seconds": epoch,
                "native_parameters": params.tolist(),
            }
        )
    return result


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Compare each requested production grid to its dense public-response FFT."""
    params = np.asarray(case["native_parameters"], dtype=np.float64)
    duration = float(case["duration_seconds"])
    epoch = float(case["epoch_seconds"])
    reference = dense_reference(params, duration, epoch)
    comparisons = []
    for response_samples in RESPONSE_SAMPLES:
        waveform = bg.ucb_waveform(
            params,
            T=duration,
            BW=BW,
            t0=epoch,
            backend="onfly",
            response_samples=response_samples,
        )
        errors = {
            channel: relative_l2(np.asarray(waveform[channel]), reference[channel])
            for channel in reference
        }
        comparisons.append(
            {
                "response_samples": response_samples,
                "resolved_response_samples": (
                    automatic_response_samples(duration)
                    if response_samples is None
                    else response_samples
                ),
                "relative_l2_by_channel": errors,
                "max_relative_l2": max(errors.values()),
            }
        )
    return {**case, "comparisons": comparisons}


def largest(
    records: list[dict[str, Any]], response_samples: int | None
) -> dict[str, Any]:
    """Summarize the worst source, globally and below the 30 mHz boundary."""
    values = []
    for record in records:
        comparison = next(
            item
            for item in record["comparisons"]
            if item["response_samples"] == response_samples
        )
        values.append((comparison["max_relative_l2"], record, comparison))

    def summary(candidates: list[tuple[float, dict[str, Any], dict[str, Any]]]):
        error, record, comparison = max(candidates, key=lambda item: item[0])
        return {
            "max_relative_l2": error,
            "case_index": record["index"],
            "frequency_hz": record["frequency_hz"],
            "duration_years": record["duration_years"],
            "parameter_count": record["parameter_count"],
            "resolved_response_samples": comparison["resolved_response_samples"],
        }

    return {
        "response_samples": response_samples,
        "all_frequencies": summary(values),
        "frequencies_up_to_30_mhz": summary(
            [item for item in values if item[1]["frequency_hz"] <= 0.03]
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Accept an output path while retaining a useful durable default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/performance/onfly-validation.json"),
        help="JSON report path",
    )
    return parser.parse_args()


def main() -> None:
    """Run the deterministic correctness validation and write its full evidence."""
    arguments = parse_arguments()
    records = []
    for case in cases():
        print(f"case {case['index'] + 1}/90", flush=True)
        records.append(validate_case(case))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "production sparse-spectrum convergence against dense public response"
        ),
        "physical_validity": (
            "not assessed; pinned spline C fixtures cover that separately"
        ),
        "settings": {
            "BW": BW,
            "response_samples": list(RESPONSE_SAMPLES),
            "automatic_policy": {
                "density_per_year": 128,
                "year_seconds": RESPONSE_YEAR,
                "buckets": [32, 64, 128, 256, 512, 1024, 2048],
                "cap": 2048,
            },
            "dense_reference": (
                "public tdi_response plus explicit phase and sqrt(T)/(2*BW) FFT"
            ),
        },
        "provenance": provenance(),
        "largest_errors": [largest(records, count) for count in RESPONSE_SAMPLES],
        "cases": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
